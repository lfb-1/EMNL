# EMNL
Bridging Generative and Discriminative Noisy-Label Learning via Direction-Agnostic EM Formulation

## Hydra Configuration

Training is now driven by [Hydra](https://hydra.cc/) configs under `conf/experiment`. Make sure `hydra-core` is installed (e.g., `pip install hydra-core`) and launch with the default CIFAR-10 setup:

```
python main.py
```

Select a different preset by overriding the `experiment` group and any hyperparameters you need:

```
# CIFAR-100 with a custom noise rate and data root
python main.py experiment=cifar100 experiment.r=0.6 root=/data/cifar-100

# CIFAR-10N using the "clean_label" target annotations
python main.py experiment=cifar10n target=clean_label root=/data/cifar-10
```

Each YAML mirrors the legacy `configs.py` entries, so you can add new experiments by creating a file in `conf/experiment` and overriding values at the CLI.
