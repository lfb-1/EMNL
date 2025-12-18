from sklearn.mixture import GaussianMixture
import torch
import numpy as np
from ResNet import resnet_cifar34
from PreResNet import ResNet18
import torch.optim as optim
import torch.nn as nn
from dataloader_cifar import cifar_dataloader
import wandb
import pandas as pd
from helper import AverageMeter
import torchmetrics as tm
from tqdm import tqdm
import torch.nn.functional as F
from dynamic_partial import DynamicPartial, sample_neg, prior_loss, pxy_kl, pyx_kl

# Map friendly backbone names to constructors so Hydra configs can switch models.
BACKBONES = {
    "resnet34": resnet_cifar34,
    "resnet_cifar34": resnet_cifar34,
    "resnet18": ResNet18,
    "preresnet18": ResNet18,
}


def build_backbone(name: str, num_classes: int) -> nn.Module:
    key = name.lower()
    if key not in BACKBONES:
        raise ValueError(f"Unsupported backbone '{name}'. Available options: {sorted(BACKBONES.keys())}")
    return BACKBONES[key](num_classes).cuda()


# from torchsort import soft_rank, soft_sort
from collections import deque


class CIFAR_Trainer:
    def __init__(self, config, name: str):
        self.warmup_epochs = config.warmup_epochs
        self.total_epochs = config.total_epochs
        self.num_classes = config.num_classes
        self.num_pri = config.num_prior
        self.beta = config.beta
        self.reg_kl = pxy_kl if config.optim_goal == "pxy" else pyx_kl
        self.cot = getattr(config, "cot", 1)

        self.net2 = None
        self.optim2 = None
        self.latent2 = None
        self.scheduler2 = None

        backbone_name = getattr(config, "backbone", "resnet34")
        self.net = build_backbone(backbone_name, self.num_classes)
        self.optim = optim.SGD(
            self.net.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.wd,
            nesterov=config.nesterov,
        )
        self.latent = DynamicPartial(50000, config.beta, config.num_classes)
        self.memory_queue_len = config.memory_queue_len
        self.loss_weight = getattr(config, "loss_weight", [1.0, 1.0, 0.2])

        if self.cot == 2:
            self.net2 = build_backbone(backbone_name, self.num_classes)
            self.optim2 = optim.SGD(
                self.net2.parameters(),
                lr=config.lr,
                momentum=0.9,
                weight_decay=config.wd,
                nesterov=config.nesterov,
            )
            self.latent2 = DynamicPartial(50000, config.beta, config.num_classes)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=config.lr_decay, gamma=0.1)
        if self.cot == 2:
            self.scheduler2 = optim.lr_scheduler.MultiStepLR(self.optim2, milestones=config.lr_decay, gamma=0.1)

        self.criterion = nn.CrossEntropyLoss(reduction="none").cuda()
        loader = cifar_dataloader(
            config.dataset,
            config.r,
            config.noise_mode,
            config.batch_size,
            config.num_workers,
            config.root_dir,
        )

        self.train_loader, self.eval_loader = loader.run("train")
        self.test_loader = loader.run("test")
        if config.wandb:
            self.use_wandb = True
            wandb.login()
            wandb.init(project="GNLv2", config=config, name=name)
        else:
            self.use_wandb = False

        self.logger = pd.DataFrame(
            # columns=["train acc", "train cov", "train ineff", "test acc", "test cov", "test ineff"]
            columns=["train acc", "test acc", "clean_cov", "noisy_cov", "clean_unc", "clean_unc"]
        )

        self.train_acc = AverageMeter()
        self.m_cov = AverageMeter()
        self.m_unc_clean = AverageMeter()
        self.m_unc_noisy = AverageMeter()
        self.l_ce = AverageMeter()
        self.l_pri = AverageMeter()
        self.l_kl = AverageMeter()
        self.test_acc = AverageMeter()
        self.calc_acc = tm.Accuracy(task="multiclass", num_classes=config.num_classes).cuda()
        self.queue_depth = AverageMeter()

    def pipeline(self, train_func):
        if self.cot == 2:
            self.pipeline_cot()
            return

        if self.memory_queue_len > 0:
            memory_queue = deque(maxlen=self.memory_queue_len)
        else:
            memory_queue = None
        for epoch in range(self.total_epochs):
            if epoch < self.warmup_epochs:
                self.train(epoch, self.net, self.optim, self.latent)
            else:
                probs = self.eval_train(self.net)
                self.train(epoch, self.net, self.optim, self.latent, probs, memory_queue=memory_queue)

            self.test(self.net)
            self.wandb_update(epoch)
            self.scheduler.step()

    def pipeline_cot(self):
        if self.memory_queue_len > 0:
            memory_queue_1 = deque(maxlen=self.memory_queue_len)
            memory_queue_2 = deque(maxlen=self.memory_queue_len)
        else:
            memory_queue_1 = None
            memory_queue_2 = None

        for epoch in range(self.total_epochs):
            if epoch < self.warmup_epochs:
                self.train_cot(epoch, self.net, self.optim, self.latent, self.latent2, memory_queue_1)
                self.train_cot(epoch, self.net2, self.optim2, self.latent2, self.latent, memory_queue_2)
            else:
                probs1 = self.eval_train(self.net)
                probs2 = self.eval_train(self.net2)
                self.train_cot(
                    epoch,
                    self.net,
                    self.optim,
                    self.latent,
                    self.latent2,
                    memory_queue_1,
                    probs_peer=probs2,
                )
                self.train_cot(
                    epoch,
                    self.net2,
                    self.optim2,
                    self.latent2,
                    self.latent,
                    memory_queue_2,
                    probs_peer=probs1,
                )

            self.test_cot()
            self.wandb_update(epoch)
            self.scheduler.step()
            self.scheduler2.step()

    def train(
        self, epoch: int, net: nn.Module, optimizer: optim.SGD, mov: DynamicPartial, probs=None, memory_queue=None
    ):
        net.train()
        for batch_idx, (inputs, targets, clean, idx) in enumerate(tqdm(self.train_loader, desc=f"Epoch: {epoch}")):
            inputs, targets, clean = inputs.cuda(), targets.cuda(), clean.cuda().to(torch.int64)
            onehot_labels = F.one_hot(targets, self.num_classes).cuda()

            optimizer.zero_grad()
            outputs, tildey, _ = net(inputs)

            pred = [F.one_hot(mov.sample_latent(idx).sample(), self.num_classes).float() for i in range(self.num_pri)]
            prior_cov = [(pred[i] + onehot_labels).clamp(max=1.0) for i in range(self.num_pri)]
            prior = [
                sample_neg(
                    prior_cov[i],
                    self.num_classes,
                    probs[idx] if probs is not None else None,
                )
                for i in range(self.num_pri)
            ]
            prior = [prior[i] / prior[i].sum(1, keepdim=True) for i in range(self.num_pri)]

            mov.update_hist(outputs.softmax(1), idx)

            # Pass raw logits to loss functions so memory-extended normalization is used
            log_outputs = outputs.log_softmax(1)
            log_prior = [prior[i].clamp(min=1e-9, max=1.0).log() for i in range(self.num_pri)]

            if memory_queue is None:
                extended_log_outputs = log_outputs
                queue_depth = 0
            else:
                history_depth = len(memory_queue)
                if history_depth == 0:
                    extended_log_outputs = log_outputs
                else:
                    # Include all stored history so memory_queue_len controls context span.
                    extended_log_outputs = torch.cat([log_outputs] + list(memory_queue), dim=0)
                # deque(maxlen=...) drops the oldest entry automatically, so append after use.
                memory_queue.append(log_outputs.detach())
                queue_depth = history_depth

            self.queue_depth.update(queue_depth)

            ce = self.criterion(tildey, targets).mean()
            pri = (
                sum([prior_loss([log_outputs, extended_log_outputs], log_prior[i]) for i in range(self.num_pri)])
                / self.num_pri
            )
            reg_kl = (
                sum(
                    [
                        self.reg_kl([log_outputs, extended_log_outputs], tildey, log_prior[i])
                        for i in range(self.num_pri)
                    ]
                )
                / self.num_pri
            )
            a, b, c = self.loss_weight
            l = a * ce + b * pri + c * reg_kl

            l.backward()
            optimizer.step()

            # MoCo: Update momentum encoder and memory queue for the next iteration.
            # This follows the standard MoCo procedure.

            self.metrics_update(inputs, clean, targets, prior[0], prior_cov[0], ce, pri, reg_kl)
            self.train_acc.update(self.calc_acc(outputs, clean.int()).item() * 100.0)

    def train_cot(
        self,
        epoch: int,
        net: nn.Module,
        optimizer: optim.SGD,
        mov: DynamicPartial,
        peer_mov: DynamicPartial,
        memory_queue=None,
        probs_peer=None,
    ):
        net.train()
        for batch_idx, (inputs, targets, clean, idx) in enumerate(tqdm(self.train_loader, desc=f"Epoch: {epoch}")):
            inputs, targets, clean = inputs.cuda(), targets.cuda(), clean.cuda().to(torch.int64)
            onehot_labels = F.one_hot(targets, self.num_classes).float().cuda()

            optimizer.zero_grad()

            lam = np.random.beta(0.5, 0.5)
            lam = max(lam, 1 - lam)
            mix_idx = torch.randperm(inputs.shape[0], device=inputs.device)
            mix_inputs = lam * inputs + (1 - lam) * inputs[mix_idx]
            mix_targets = lam * onehot_labels + (1 - lam) * onehot_labels[mix_idx]

            outputs, tildey, _ = net(mix_inputs)

            pred = [
                F.one_hot(peer_mov.sample_latent(idx).sample(), self.num_classes).float() for _ in range(self.num_pri)
            ]
            prior_cov = [(pred[i] + onehot_labels).clamp(max=1.0) for i in range(self.num_pri)]
            prior = [
                sample_neg(
                    prior_cov[i],
                    self.num_classes,
                    probs_peer[idx] if probs_peer is not None else None,
                )
                for i in range(self.num_pri)
            ]
            prior = [torch.clamp(p, max=1.0) / torch.clamp(p.sum(1, keepdim=True), min=1e-8) for p in prior]

            mov.update_hist(outputs.softmax(1), idx)

            log_outputs = outputs.log_softmax(1)
            log_prior = [prior[i].clamp(min=1e-9, max=1.0).log() for i in range(self.num_pri)]

            if memory_queue is None:
                extended_log_outputs = log_outputs
                queue_depth = 0
            else:
                history_depth = len(memory_queue)
                if history_depth == 0:
                    extended_log_outputs = log_outputs
                else:
                    extended_log_outputs = torch.cat([log_outputs] + list(memory_queue), dim=0)
                memory_queue.append(log_outputs.detach())
                queue_depth = history_depth
            self.queue_depth.update(queue_depth)

            ce = -torch.mean(torch.sum(F.log_softmax(tildey, dim=1) * mix_targets, dim=1))
            pri = (
                sum([prior_loss([log_outputs, extended_log_outputs], log_prior[i]) for i in range(self.num_pri)])
                / self.num_pri
            )
            reg_kl = (
                sum(
                    [
                        self.reg_kl([log_outputs, extended_log_outputs], tildey, log_prior[i])
                        for i in range(self.num_pri)
                    ]
                )
                / self.num_pri
            )
            a, b, c = self.loss_weight
            loss = a * ce + b * pri + c * reg_kl

            loss.backward()
            optimizer.step()

            self.metrics_update(inputs, clean, targets, prior[0], prior_cov[0], ce, pri, reg_kl)
            self.train_acc.update(self.calc_acc(outputs, clean.int()).item() * 100.0)

    @torch.no_grad()
    def eval_train(self, net: nn.Module, num_classes=100):
        net.eval()
        losses = torch.zeros(50000)
        for batch_idx, (inputs, targets, clean, index) in enumerate(self.eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net.forward_test(inputs)
            # loss = F.kl_div(outputs.log_softmax(1),tildey.log_softmax(1),reduction='none',log_target=True).sum(1)
            loss = F.cross_entropy(outputs, targets, reduction="none")
            # loss = -torch.sum(
            #     outputs.softmax(1) * F.one_hot(targets, num_classes).float().clamp(min=1e-9).log(), dim=1
            # )
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
        losses = ((losses - losses.min()) / (losses.max() - losses.min())).unsqueeze(1)
        input_loss = losses.reshape(-1, 1)
        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return 1 - torch.from_numpy(prob).cuda()

    @torch.no_grad()
    def test(self, net):
        if self.cot == 2:
            self.test_cot()
            return
        net.eval()
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # outputs = net.forward_test(inputs)
            outputs, _, _ = net(inputs)
            self.test_acc.update(self.calc_acc(outputs, targets.int()).item() * 100.0)

    @torch.no_grad()
    def test_cot(self):
        self.net.eval()
        self.net2.eval()
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _, _ = self.net(inputs)
            outputs2, _, _ = self.net2(inputs)
            outputs = outputs1 + outputs2
            self.test_acc.update(self.calc_acc(outputs, targets.int()).item() * 100.0)

    def metrics_update(self, inputs, clean, targets, prior, prior_cov, ce, pri, reg_kl):
        self.m_cov.update(
            torch.logical_and(prior_cov, F.one_hot(clean, self.num_classes)).sum().item(),
            inputs.shape[0],
        )
        clean_index = targets == clean
        noisy_index = targets != clean
        self.m_unc_clean.update(((prior[clean_index] > 0).sum(1).float().mean().item()))
        self.m_unc_noisy.update(((prior[noisy_index] > 0).sum(1).float().mean().item()))

        # self.m_unc.update((prior > 0).sum(1).float().mean().item())
        self.l_ce.update(ce.item())
        self.l_pri.update(pri.item())
        self.l_kl.update(reg_kl.item())

    def wandb_update(self, epoch):
        stats = {
            "L_ce": self.l_ce.avg,
            "L_pri": self.l_pri.avg,
            "L_kl": self.l_kl.avg,
            "Coverage": self.m_cov.avg,
            # "Uncertainty": self.m_unc.avg,
            "Clean Uncertainty": self.m_unc_clean.avg,
            "Noisy Uncertainty": self.m_unc_noisy.avg,
            "Memory Queue Depth": self.queue_depth.avg,
            "epoch": epoch,
            "train acc": self.train_acc.avg,
            "test acc": self.test_acc.avg,
        }
        wandb.log(stats)
        print(f"Train acc: {self.train_acc.avg} Test acc: {self.test_acc.avg}\n")
        [
            i.reset()
            for i in [
                self.l_ce,
                self.l_pri,
                self.l_kl,
                self.m_cov,
                self.m_unc_noisy,
                self.m_unc_clean,
                self.train_acc,
                self.test_acc,
                self.queue_depth,
            ]
        ]
