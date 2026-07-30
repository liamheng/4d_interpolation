import abc
from functools import partial

import torch
import torchmetrics

from others.metrics.hausdorf_distance import HausdorffDistance2D
from others.metrics.registration_dice import RegistrationDiceOnline, RegistrationDiceOffline
from others.metrics.registration_unsupervised import PercentageNegativeJacobian, StdLogJacobianDeterminant, \
    MagnitudeGradJacobianDeterminant
from others.metrics.samplewise_metrics import MSE, MAE, RSE, NMSE, NMI, PSNR, SSIM, NCC, LPIPS
from utils import get_class_from_subclasses


class MyMetrics(torchmetrics.Metric, abc.ABC):
    """ MyMetrics is a wrapper for torchmetrics.Metric.
        It can calculate multiple metrics according to the options,
        and can calculate the standard variance of the metrics.
    """

    @staticmethod
    def get_metrics(task='segmentation'):
        return get_class_from_subclasses(MyMetrics, 'My' + task + 'Metrics', allow_case=True)

    def __init__(self, opt, device):
        super().__init__()
        self.metrics_list = [x.lower() for x in opt.metrics_list]
        self.result_dict = {}
        self.require_std = opt.metrics_calculate_std_var
        if self.require_std:
            for metrics_name in self.metrics_list:
                self.add_state(metrics_name + '_list', default=torch.tensor([]), dist_reduce_fx=None)
        self.to(device)

    def update(self, preds, target, *args, **kwargs):
        """
        基类默认：仍然把 preds/target 传入所有度量（与原逻辑相同）。
        具体任务子类（如 MyRegistrationMetrics）可基于 kwargs 做路由。
        """
        target = target.int()
        if self.require_std:
            for metrics_name in self.metrics_list:
                metrics = self.result_dict[metrics_name]
                metrics.update(preds, target)
                metrics_list = getattr(self, metrics_name + '_list')
                metrics_list = torch.cat([metrics_list, metrics.compute().unsqueeze(0)])
                setattr(self, metrics_name + '_list', metrics_list)
                metrics.reset()
        else:
            for metrics_name, metrics in self.result_dict.items():
                metrics.update(preds, target)

    def compute(self):
        if self.require_std:
            result = {}
            for metrics_name in self.metrics_list:
                metrics_list = getattr(self, metrics_name + '_list')
                result[metrics_name + '_mean'] = metrics_list.mean().item()
                result[metrics_name + '_std'] = metrics_list.std().item()
            self.reset()
            return result

        result = {}
        for metrics_name, metrics in self.result_dict.items():
            result[metrics_name] = metrics.compute().item()

        self.reset()
        return result


# ------------------------------
# Segmentation metrics registry
# ------------------------------
segmentation_metrics_name_cls_map = {
    'iou': torchmetrics.JaccardIndex,
    'f1': torchmetrics.F1Score,
    'accuracy': torchmetrics.Accuracy,
    'auroc': torchmetrics.AUROC,
    'aur': torchmetrics.AUROC,
    'dice': torchmetrics.Dice,
    'mcc': torchmetrics.MatthewsCorrCoef,
    'hausdorff': HausdorffDistance2D
}

# metrics that have the attribute "task" should be treated differently
segmentation_list_have_attr_task = ['iou', 'f1', 'accuracy', 'auroc', 'aur', 'mcc', 'hausdorff']


class MySegmentationMetrics(MyMetrics):
    def __init__(self, opt, device):
        super().__init__(opt, device)
        for metrics_name in self.metrics_list:
            if metrics_name not in segmentation_metrics_name_cls_map:
                raise NotImplementedError('Metric ' + metrics_name + ' is not implemented.')

            metrics_cls = segmentation_metrics_name_cls_map[metrics_name]
            if metrics_name in segmentation_list_have_attr_task:
                # in this selection branch, all metrics have the attribute "task"
                metrics_part = partial(metrics_cls, task='binary') \
                    if opt.output_nc == 1 else partial(metrics_cls, task='multiclass', num_classes=opt.output_nc)
            else:
                metrics_part = partial(metrics_cls, ignore_index=0 if opt.output_nc == 1 else None)
            if metrics_name in ['aur', 'auroc', 'auc']:
                self.result_dict[metrics_name] = metrics_part(average='samples').to(device=device)
            else:
                self.result_dict[metrics_name] = metrics_part(threshold=opt.metrics_threshold, average='samples').to(
                    device=device)


# ------------------------------
# Registration metrics registry
# ------------------------------
registration_metrics_name_cls_map = {
    # >>> 改动：用“样本级”实现替换原先 TM 聚合版本
    'mse': MSE,  # 原: torchmetrics.MeanSquaredError
    'mae': MAE,  # 原: torchmetrics.MeanAbsoluteError
    'rse': RSE,  # 原: torchmetrics.RelativeSquaredError
    'nmse': NMSE,  # 原: torchmetrics.RelativeSquaredError（你代码里注释 nmse≈rse）
    'ssim': SSIM,
    'psnr': PSNR,
    'nmi': NMI,
    'ncc': NCC,
    'lpips': LPIPS,
    'dice_online': RegistrationDiceOnline,
    'dice_offline': RegistrationDiceOffline,
    'njd': PercentageNegativeJacobian,
    'slj': StdLogJacobianDeterminant,  # Std Log Jacobian Determinant
    'mgj': MagnitudeGradJacobianDeterminant  # Magnitude Grad Jacobian Determinant
}

# 路由用集合（便于扩展）
registration_naive_set = {'mse', 'mae', 'rse', 'nmse', 'ssim', 'psnr', 'nmi', 'dice_online'}
registration_require_label_set = {'dice_offline'}  # pred_masks + target_masks
registration_unsupervised_set = {'njd', 'slj', 'mgj'}  # displacement


class MyRegistrationMetrics(MyMetrics):
    def __init__(self, opt, device):
        super().__init__(opt, device)
        for metrics_name in self.metrics_list:
            if metrics_name not in registration_metrics_name_cls_map:
                raise NotImplementedError('Metric ' + metrics_name + ' is not implemented.')

            metrics_cls = registration_metrics_name_cls_map[metrics_name]
            metric_instance = metrics_cls(opt=opt).to(device)

            # >>> 以下三支分支：保持你原路由不变，仅在“else”里按需传参
            # if metrics_name == 'dice_online':
            #     metric_instance = metrics_cls(opt=opt).to(device)
            # elif metrics_name == 'dice_offline':
            #     metric_instance = metrics_cls(opt=opt).to(device)
            # elif metrics_name == 'njd':
            #     metric_instance = metrics_cls(is_3d=bool(getattr(opt, 'is_3d', False))).to(device)
            # else:
            #     # >>> 最小改动：
            #     # - 对 SSIM/PSNR：用 reduction='none'，返回 (N,) 以支持样本级 std；
            #     # - 对其它（换成了我们“样本级”实现）：无需传 average/reduction；
            #     if metrics_name == 'ssim':
            #         metric_instance = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            #     elif metrics_name == 'psnr':
            #         metric_instance = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)
            #     else:
            #         metric_instance = metrics_cls().to(device)

            self.result_dict[metrics_name] = metric_instance

    def update(self, preds, target, *args, **kwargs):
        """
        统一路由规则（便于扩展）：
          - registration_require_label_set  ->  metrics.update(pred_masks, target_masks)
          - registration_unsupervised_set  ->  metrics.update(displacement)
          - 其他                           ->  metrics.update(preds, target)
        """
        displacement = kwargs.get('displacement', None)
        pred_masks = kwargs.get('pred_masks', None)
        target_masks = kwargs.get('target_masks', None)

        # print(preds.shape, target.shape)

        if self.require_std:
            for metrics_name in self.metrics_list:
                metrics = self.result_dict[metrics_name]

                if metrics_name in registration_require_label_set:
                    if pred_masks is None or target_masks is None:
                        raise ValueError(
                            f"[MyRegistrationMetrics.update] metric '{metrics_name}' 需要 kwargs['pred_masks'] 和 kwargs['target_masks']")
                    metrics.update(pred_masks=pred_masks, target_masks=target_masks)

                elif metrics_name in registration_unsupervised_set:
                    if displacement is None:
                        raise ValueError(
                            f"[MyRegistrationMetrics.update] metric '{metrics_name}' 需要 kwargs['displacement']")
                    metrics.update(displacement=displacement)

                else:
                    metrics.update(preds, target)

                # 统计方差：逐 metric 采样一次
                metrics_list = getattr(self, metrics_name + '_list')
                metrics_list = torch.cat([metrics_list, metrics.compute().unsqueeze(0)])
                setattr(self, metrics_name + '_list', metrics_list)
                metrics.reset()
        else:
            for metrics_name, metrics in self.result_dict.items():
                if metrics_name in registration_require_label_set:
                    if pred_masks is None or target_masks is None:
                        raise ValueError(
                            f"[MyRegistrationMetrics.update] metric '{metrics_name}' 需要 kwargs['pred_masks'] 和 kwargs['target_masks']")
                    metrics.update(pred_masks=pred_masks, target_masks=target_masks)

                elif metrics_name in registration_unsupervised_set:
                    if displacement is None:
                        raise ValueError(
                            f"[MyRegistrationMetrics.update] metric '{metrics_name}' 需要 kwargs['displacement']")
                    metrics.update(displacement=displacement)

                else:
                    metrics.update(preds, target)
