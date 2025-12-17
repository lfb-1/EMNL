import torch.nn as nn
import torch
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np


class DynamicPartial(nn.Module):
    def __init__(self, num_samples, beta=0.9, num_classes=10, T=0.5, memory_size=1000, momentum=0.999):
        super(DynamicPartial, self).__init__()
        self.latent = (torch.ones(num_samples, num_classes) / num_classes).cuda()
        self.beta = beta
        self.T = T

    def update_hist(self, probs, index):
        probs = torch.clamp(probs, 1e-8, 1.0 - 1e-8).detach()

        # Additional safety check for NaN/Inf values
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("[WARNING] DynamicPartial: NaN/Inf detected in probs, skipping update")
            return

        probs_sum = probs.sum(1, keepdim=True)
        probs_sum = torch.clamp(probs_sum, min=1e-8)  # Prevent division by zero
        probs = probs / probs_sum

        self.latent[index] = self.beta * self.latent[index] + (1 - self.beta) * probs

    def sample_latent(self, index=None):
        latent_distribution = self.latent[index] if index is not None else self.latent

        # Apply temperature scaling with numerical stability
        latent_scaled = latent_distribution ** (1 / self.T)

        # Add safety checks for numerical stability
        if torch.isnan(latent_scaled).any() or torch.isinf(latent_scaled).any():
            print("[WARNING] DynamicPartial: NaN/Inf in latent_scaled, using uniform distribution")
            if index is not None:
                uniform_dist = torch.ones_like(latent_distribution) / latent_distribution.shape[-1]
                return dist.Categorical(uniform_dist)
            else:
                uniform_dist = torch.ones_like(self.latent) / self.latent.shape[-1]
                return dist.Categorical(uniform_dist)

        # Normalize with numerical stability
        norm_ld_sum = latent_scaled.sum(1, keepdim=True)
        norm_ld_sum = torch.clamp(norm_ld_sum, min=1e-8)
        norm_ld = latent_scaled / norm_ld_sum

        return dist.Categorical(norm_ld)


def sample_neg(prior_cov, num_classes, num=None):
    device = prior_cov.device
    dtype = prior_cov.dtype

    prior_np = prior_cov.detach().cpu().numpy()
    num_np = None if num is None else num.detach().cpu().numpy()

    merged = []
    for i in range(prior_np.shape[0]):
        positives = prior_np[i] > 0
        negative_indices = np.where(~positives)[0]

        if negative_indices.size == 0:
            merged.append(torch.from_numpy(positives.astype(np.float32)))
            continue

        if negative_indices.size == 1:
            sample_count = 0
        else:
            if num_np is not None:
                desired = int(round(float(num_np[i]) * negative_indices.size))
            else:
                desired = np.random.randint(1, negative_indices.size)

            desired = max(0, desired)
            desired = min(desired, negative_indices.size - 1)
            sample_count = desired

        chosen = () if sample_count <= 0 else tuple(np.random.choice(negative_indices, sample_count, replace=False))
        combined = positives.astype(np.float32)
        if len(chosen) > 0:
            combined[np.array(chosen)] = 1.0
        merged.append(torch.from_numpy(combined))

    return torch.stack(merged, dim=0).to(device=device, dtype=dtype)


#! Two approaches for Eq. 12
#! Option 1: log_outputs.softmax(0)
#! Option 2: log_outputs / log_outputs.sum(0,keepdim=True), logsumexp is used for computing in log space


def prior_loss(log_outputs, log_prior):
    log_outputs_normalized = log_outputs[0] - torch.logsumexp(log_outputs[1], dim=0, keepdim=True)
    return F.kl_div(
        log_outputs[0],
        (log_prior + log_outputs_normalized).log_softmax(1),
        reduction="batchmean",
        log_target=True,
    )

    return F.kl_div(
        log_outputs,
        (log_prior + log_outputs.log_softmax(0)).log_softmax(1),
        reduction="batchmean",
        log_target=True,
    )
    # return F.kl_div(
    #     log_outputs,
    #     (log_prior + (log_outputs - torch.logsumexp(log_outputs,
    #      dim=0, keepdim=True))).log_softmax(1),
    #     reduction="batchmean",
    #     log_target=True,
    # )


def pxy_kl(log_outputs, tildey, log_prior):
    return F.kl_div(
        (tildey.log_softmax(1) + log_prior).log_softmax(1),
        log_outputs[0].detach(),
        reduction="batchmean",
        log_target=True,
    )


def pyx_kl(log_outputs, tildey, log_prior):
    log_outputs_normalized = log_outputs[0] - torch.logsumexp(log_outputs[1], dim=0, keepdim=True)
    return F.kl_div(
        (
            tildey.log_softmax(1)
            + torch.logsumexp(log_outputs_normalized + log_prior, dim=1, keepdim=True)
            # + torch.logsumexp(
            #     (log_outputs - torch.logsumexp(log_outputs,
            #      dim=0, keepdim=True)) + log_prior,
            #     dim=1,
            #     keepdim=True,
            # )
        ).log_softmax(1),
        log_outputs[0].detach(),
        reduction="batchmean",
        log_target=True,
    )
