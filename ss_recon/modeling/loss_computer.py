from abc import ABC, abstractmethod

import torch
from fvcore.common.registry import Registry

from ss_recon.data.transforms.transform import build_normalizer
from ss_recon.utils import complex_utils as cplx
from ss_recon.utils import transforms as T

LOSS_COMPUTER_REGISTRY = Registry("LOSS_COMPUTER")  # noqa F401 isort:skip
LOSS_COMPUTER_REGISTRY.__doc__ = """
Registry for loss computers.

The registered object will be called with `obj(cfg)`
and expected to return a LossComputer object.
"""

EPS = 1e-11
IMAGE_LOSSES = ["l1", "l2", "psnr", "nrmse", "mag_l1", "perp_loss"]
KSPACE_LOSSES = ["k_l1", "k_l1_normalized"]


def build_loss_computer(cfg, name):
    return LOSS_COMPUTER_REGISTRY.get(name)(cfg)


class LossComputer(ABC):
    def __init__(self, cfg):
        self._normalizer = build_normalizer(cfg)

    @abstractmethod
    def __call__(self, input, output):
        pass

    def _get_metrics(self, target: torch.Tensor, output: torch.Tensor, loss_name):
        # Compute metrics
        abs_error = cplx.abs(output - target)
        abs_mag_error = torch.abs(cplx.abs(output) - cplx.abs(target))
        l1 = torch.mean(abs_error)
        mag_l1 = torch.mean(abs_mag_error)
        N = target.shape[0]

        abs_error = abs_error.view(N, -1)
        tgt_mag = cplx.abs(target).view(N, -1)
        l2 = torch.sqrt(torch.mean(abs_error ** 2, dim=1))
        psnr = 20 * torch.log10(tgt_mag.max(dim=1)[0] / (l2 + EPS))
        nrmse = l2 / torch.sqrt(torch.mean(tgt_mag ** 2, dim=1))

        metrics_dict = {
            "l1": l1,
            "l2": l2.mean(),
            "psnr": psnr.mean(),
            "nrmse": nrmse.mean(),
            "mag_l1": mag_l1,
        }

        if loss_name == "perp_loss":
            metrics_dict.update(perp_loss(output, target))

        if loss_name in KSPACE_LOSSES:
            target, output = T.fft2(target), T.fft2(output)
            abs_error = cplx.abs(target - output)
            if loss_name == "k_l1":
                metrics_dict["loss"] = torch.mean(abs_error)
            elif loss_name == "k_l1_normalized":
                metrics_dict["loss"] = torch.mean(abs_error / (cplx.abs(target) + EPS))
            else:
                assert False  # should not reach here
        else:
            loss = metrics_dict[loss_name]
            metrics_dict["loss"] = loss

        return metrics_dict


@LOSS_COMPUTER_REGISTRY.register()
class BasicLossComputer(LossComputer):
    def __init__(self, cfg):
        super().__init__(cfg)
        loss_name = cfg.MODEL.RECON_LOSS.NAME
        assert loss_name in IMAGE_LOSSES or loss_name in KSPACE_LOSSES
        self.loss = loss_name
        self.renormalize_data = cfg.MODEL.RECON_LOSS.RENORMALIZE_DATA

    def __call__(self, input, output):
        pred: torch.Tensor = output["pred"]
        target = output["target"].to(pred.device)

        if self.renormalize_data:
            normalization_args = {k: input.get(k, output[k]) for k in ["mean", "std"]}
            normalized = self._normalizer.undo(
                image=pred,
                target=target,
                mean=normalization_args["mean"],
                std=normalization_args["std"],
            )
            output = normalized["image"]
            target = normalized["target"]
        else:
            output = pred

        metrics_dict = self._get_metrics(target, output, self.loss)
        return metrics_dict


@LOSS_COMPUTER_REGISTRY.register()
class N2RLossComputer(LossComputer):
    def __init__(self, cfg):
        super().__init__(cfg)
        recon_loss = cfg.MODEL.RECON_LOSS.NAME
        consistency_loss = cfg.MODEL.CONSISTENCY.LOSS_NAME

        assert recon_loss in IMAGE_LOSSES or recon_loss in KSPACE_LOSSES
        assert consistency_loss in IMAGE_LOSSES or consistency_loss in KSPACE_LOSSES

        self.recon_loss = recon_loss
        self.consistency_loss = consistency_loss
        self.renormalize_data = cfg.MODEL.RECON_LOSS.RENORMALIZE_DATA
        self.consistency_weight = cfg.MODEL.CONSISTENCY.LOSS_WEIGHT
        # self.use_robust = cfg.MODEL.LOSS.USE_ROBUST
        # self.beta = cfg.MODEL.LOSS.BETA
        # self.robust_step_size = cfg.MODEL.LOSS.ROBUST_STEP_SIZE

    def _compute_metrics(self, input, output, loss):
        """Computes image metrics on prediction and target data."""
        if output is None or len(output) == 0:
            return {k: torch.Tensor([0.0]).detach() for k in ["l1", "l2", "psnr", "loss"]}

        pred: torch.Tensor = output["pred"]
        target = output["target"].to(pred.device)
        if self.renormalize_data:
            normalized = self._normalizer.undo(
                image=pred, target=target, mean=input["mean"], std=input["std"]
            )
            output = normalized["image"]
            target = normalized["target"]
        else:
            output = pred

        # Compute metrics
        metrics_dict = self._get_metrics(target, output, loss)
        return metrics_dict

    # def compute_robust_loss(self, group_loss):
    #     if torch.is_grad_enabled():  # update adv_probs if in training mode
    #         adjusted_loss = group_loss
    #         if self.do_adj:
    #             adjusted_loss += self.loss_adjustment
    #         logit_step = self.robust_step_size * adjusted_loss.data
    #         if self.stable:
    #             self.adv_probs_logits = self.adv_probs_logits + logit_step
    #         else:
    #             self.adv_probs = self.adv_probs * torch.exp(logit_step)
    #             self.adv_probs = self.adv_probs / self.adv_probs.sum()
    #
    #     if self.stable:
    #         adv_probs = torch.softmax(self.adv_probs_logits, dim=-1)
    #     else:
    #         adv_probs = self.adv_probs
    #     robust_loss = group_loss @ adv_probs
    #     return robust_loss, adv_probs

    def __call__(self, input, output):
        output_recon = output.get("recon", None)
        output_consistency = output.get("consistency", None)

        loss = 0
        metrics_recon = {
            "recon_{}".format(k): v
            for k, v in self._compute_metrics(
                input.get("supervised", None), output_recon, self.recon_loss
            ).items()
        }
        if output_recon is not None:
            loss += metrics_recon["recon_loss"]

        metrics_consistency = {
            "cons_{}".format(k): v
            for k, v in self._compute_metrics(
                input.get("unsupervised", None), output_consistency, self.consistency_loss
            ).items()  # noqa
        }
        if output_consistency is not None:
            loss += self.consistency_weight * metrics_consistency["cons_loss"]

        metrics = {}
        if output_consistency is not None:
            metrics.update(metrics_consistency)
        if output_recon is not None:
            metrics.update(metrics_recon)

        metrics["loss"] = loss
        return metrics


def perp_loss(yhat, y):
    """Implementation of the perpendicular loss.

    Args:
        yhat: Predicted reconstruction. Must be complex.
        y: Target reconstruction. Must be complex.

    Returns:
        Dict[str, scalar]:

    References:
        Terpstra, et al. "Rethinking complex image reconstruction:
        ⟂-loss for improved complex image reconstruction with deep learning."
        International Society of Magnetic Resonance in Medicine Annual Meeting
        2021.
    """
    if cplx.is_complex(yhat):
        yhat = torch.view_as_real(yhat)
    if cplx.is_complex(y):
        y = torch.view_as_real(y)

    P = torch.abs(yhat[..., 0] * y[..., 1] - yhat[..., 1] * y[..., 0]) / cplx.abs(y)
    l1 = torch.abs(cplx.abs(y) - cplx.abs(yhat))

    return {"p_perp_loss": torch.mean(P), "perp_loss": torch.mean(P + l1)}
