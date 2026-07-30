# models/lrvf_model.py
import torch
from torch import nn
from functools import partial

from models.base_model import BaseModel
from others.other_components.lrvf.lrvf_effective_rank import effective_rank_velocity_field
from utils.utils_3d import SliceProcessor

from others.backbones.spatial_transformer import SpatialTransformer
from others.losses.registration_loss import NCCLoss, MSELoss
from others.losses.vector_distance import CharbonnierLoss

from others.other_components.lrvf.lrvf_encoder import MultiScaleEncoder3D
from others.other_components.lrvf.lrvf_lowrank_svf import LowRankSVFScaleBlock3D
from others.other_components.lrvf.lrvf_utils import (
    downsample_like,
    upsample_displacement_like,
    compose_displacements,
)


class LRVFModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(task='registration')

        parser.add_argument('--ms_scales', type=int, nargs='+', default=[4, 2, 1])
        parser.add_argument('--ms_loss_weights', type=float, nargs='+', default=[0.2, 0.3, 0.5])

        # Training-time toggles (default: disabled)
        parser.add_argument('--train_swap_first_last', action='store_true', default=False,
                            help='Randomly swap first/last frames during training (single-direction) to approximate bidirectional training.')
        parser.add_argument('--train_swap_prob', type=float, default=0.5,
                            help='Probability of swapping first/last when train_swap_first_last is enabled.')

        # Performance toggles (default: disabled)
        parser.add_argument('--cache_base_grid', action='store_true', default=False,
                            help='Cache base grids in SpatialTransformer and SVFIntegrator (speeds up training, uses extra VRAM).')
        parser.add_argument('--use_amp', action='store_true', default=False,
                            help='Enable AMP mixed-precision training (uses autocast + GradScaler).')

        parser.add_argument('--enc_base_ch', type=int, default=48)
        parser.add_argument('--enc_num_blocks', type=int, default=4)
        parser.add_argument('--enc_norm', type=str, default='instance',
                            choices=['none', 'batch', 'instance', 'group'])

        parser.add_argument('--enc_scale_ch_mult', type=int, nargs='+', default=None)

        # rank: allow 1-int or 3-int
        parser.add_argument('--rank_coarse', type=int, nargs='+', default=[64, 64, 64])
        parser.add_argument('--rank_mid', type=int, nargs='+', default=[32, 32, 32])
        parser.add_argument('--rank_fine', type=int, nargs='+', default=[16, 16, 16])

        parser.add_argument('--bases_k_coarse', type=int, default=128)
        parser.add_argument('--bases_k_mid', type=int, default=128)
        parser.add_argument('--bases_k_fine', type=int, default=128)

        parser.add_argument('--svf_integrator', type=str, default='ss',
                            choices=['ss', 'euler', 'rk2', 'rk4', 'rk45'])
        parser.add_argument('--svf_ss_init_max_disp', type=float, default=0.5)

        parser.add_argument('--image_loss', default='ncc', choices=['mse', 'ncc'],
                            help='Deprecated: training now supports parallel MSE+NCC; control weights via --lambda_mse/--lambda_ncc.')
        parser.add_argument('--ncc_win', type=int, nargs='+', default=[9])
        parser.add_argument('--lambda_mse', type=float, default=0.0)
        parser.add_argument('--lambda_ncc', type=float, default=1.0)
        parser.add_argument('--lambda_charb', type=float, default=1.0)
        parser.add_argument('--charb_eps', type=float, default=1e-6)

        parser.add_argument('--lambda_reg', type=float, default=0.0)
        parser.add_argument('--reg_loss', type=str, default='h1')

        parser.add_argument('--zero_weight_decay_names', type=str, nargs='*', default=['R'])

        parser.add_argument('--no_use_bidirectional', dest='use_bidirectional', action='store_false')
        parser.set_defaults(use_bidirectional=True)

        parser.add_argument('--core_hidden_ch', type=int, default=128)
        parser.add_argument('--core_num_res_blocks', type=int, default=4)
        parser.add_argument('--core_norm', type=str, default='instance',
                            choices=['none', 'batch', 'instance', 'group'])

        parser.add_argument('--cal_effective_rank', action='store_true', default=False,
                            help='Calculate and print effective rank of predicted velocity fields (for debugging/analysis; may slow down training).')

        return parser

    @staticmethod
    def _parse_rank_to_3(rank_value, name_for_error):
        if isinstance(rank_value, int):
            r = int(rank_value)
            return r, r, r
        if isinstance(rank_value, (list, tuple)):
            if len(rank_value) == 1:
                r = int(rank_value[0])
                return r, r, r
            if len(rank_value) == 3:
                return int(rank_value[0]), int(rank_value[1]), int(rank_value[2])
        raise ValueError(
            "%s must be an int or a list/tuple of length 1 or 3, but got: %s"
            % (name_for_error, str(rank_value))
        )

    def __init__(self, opt):
        super().__init__(opt)
        if not opt.is_3d:
            raise ValueError("LRVFModel requires --is_3d.")

        self.loss_names = ['loss_main', 'loss_mse', 'loss_ncc', 'loss_charb', 'loss_reg']

        self.ms_scales = list(opt.ms_scales)
        if len(self.ms_scales) < 1 or self.ms_scales[-1] != 1:
            raise ValueError("--ms_scales must end with 1.")
        self.ms_loss_weights = list(opt.ms_loss_weights)
        if len(self.ms_loss_weights) != len(self.ms_scales):
            raise ValueError("--ms_loss_weights length must match --ms_scales.")
        wsum = float(sum(self.ms_loss_weights))
        self.ms_loss_weights = [float(w) / wsum for w in self.ms_loss_weights]

        for i in range(len(self.ms_scales)):
            self.loss_names.append('loss_mse_s%d' % i)
            self.loss_names.append('loss_ncc_s%d' % i)
            self.loss_names.append('loss_charb_s%d' % i)

        self.model_names = ['net_main']

        if opt.is_train:
            self.visual_names = [
                'first_frame', 'last_frame',
                'warp_0_1', 'difference_0_1',
                'velocity_0_1', 'disp_0_1',
            ]
            if opt.use_bidirectional:
                self.visual_names += ['warp_1_0', 'difference_1_0', 'velocity_1_0', 'disp_1_0']
        else:
            self.visual_names = ['first_frame', 'last_frame', 'video', 'video_pred', 'difference', 'velocity_0_1', 'disp_0_1']
            if opt.use_bidirectional:
                self.visual_names += ['velocity_1_0', 'disp_1_0']

        # network
        self.net_main = nn.Module()

        # build per-scale channel multipliers mapping
        scale_ch_multipliers = None
        if getattr(opt, 'enc_scale_ch_mult', None) is not None:
            mult_list = list(opt.enc_scale_ch_mult)
            if len(mult_list) != len(self.ms_scales):
                raise ValueError("--enc_scale_ch_mult length must match --ms_scales.")
            scale_ch_multipliers = {}
            for i, s in enumerate(self.ms_scales):
                scale_ch_multipliers[int(s)] = int(mult_list[i])

        self.net_main.encoder = MultiScaleEncoder3D(
            opt.input_nc, opt.enc_base_ch, opt.enc_num_blocks, opt.enc_norm,
            self.ms_scales, scale_ch_multipliers=scale_ch_multipliers
        )

        rank_coarse_3 = self._parse_rank_to_3(opt.rank_coarse, 'rank_coarse')
        rank_mid_3 = self._parse_rank_to_3(opt.rank_mid, 'rank_mid')
        rank_fine_3 = self._parse_rank_to_3(opt.rank_fine, 'rank_fine')

        rank_by_index = []
        for i in range(len(self.ms_scales)):
            if i == 0:
                rank_by_index.append(rank_coarse_3)
            elif i == len(self.ms_scales) - 1:
                rank_by_index.append(rank_fine_3)
            else:
                rank_by_index.append(rank_mid_3)

        k_by_index = []
        for i in range(len(self.ms_scales)):
            if i == 0:
                k_by_index.append(int(opt.bases_k_coarse))
            elif i == len(self.ms_scales) - 1:
                k_by_index.append(int(opt.bases_k_fine))
            else:
                k_by_index.append(int(opt.bases_k_mid))

        self.net_main.scale_blocks = nn.ModuleDict()
        for i, s in enumerate(self.ms_scales):
            s = int(s)
            out_ch = int(self.net_main.encoder.out_channels_by_scale[s])
            in_ch = out_ch * 4  # f0,f1,diff,absdiff
            self.net_main.scale_blocks[str(s)] = LowRankSVFScaleBlock3D(
                in_ch,
                rank_by_index[i],
                k_by_index[i],
                opt.svf_integrator,
                float(opt.svf_ss_init_max_disp),
                hidden_ch=int(opt.core_hidden_ch),
                core_num_res_blocks=int(opt.core_num_res_blocks),
                core_norm=str(opt.core_norm),
                            cache_base_grid=bool(opt.cache_base_grid),
            )

        self.transformers = nn.ModuleDict()
        for s in self.ms_scales:
            s = int(s)
            self.transformers[str(s)] = SpatialTransformer(
                mode='displacement',
                is_3d=True,
                padding_mode='border',
                align_corners=False,
                image_interp='bilinear',
                cache_base_grid=bool(opt.cache_base_grid),
            )

        self.net_main.to(self.device)

        # AMP (mixed precision) — optional
        self.use_amp = bool(getattr(opt, 'use_amp', False)) and torch.cuda.is_available()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            self._amp_autocast = partial(torch.amp.autocast, device_type="cuda")
        else:
            self._amp_autocast = torch.cuda.amp.autocast

        # GradScaler: check separately (your torch.amp may not have GradScaler)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self._amp_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self._amp_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)


        # losses
        # Parallel similarity losses (MSE + NCC). Control with --lambda_mse and --lambda_ncc.
        ncc_win = opt.ncc_win[0] if isinstance(opt.ncc_win, (list, tuple)) and len(opt.ncc_win) == 1 else opt.ncc_win
        self.criterion_ncc = NCCLoss(is_3d=True, window_size=ncc_win, reduction='mean', swap_xy=False)
        self.criterion_mse = MSELoss(reduction='mean', swap_xy=False)

        self.criterion_charb = CharbonnierLoss(epsilon=float(opt.charb_eps), reduction='mean')

        self.lambda_reg = float(opt.lambda_reg)
        self.reg_loss = None
        if self.lambda_reg > 0:
            from others.losses.registration_regularization import SHORT_TO_LOSS_CLASS as REG_SHORT_TO_LOSS
            self.reg_loss = REG_SHORT_TO_LOSS[opt.reg_loss]()

        self._interval_len = None
        # self.set_optimizers()

    def set_input(self, data_dict):
        if 'video_path' in data_dict:
            self.image_paths = data_dict['video_path']

        self.first_frame = data_dict['first_frame'].to(self.device)
        self.last_frame = data_dict['last_frame'].to(self.device)

        # Optional online augmentation: randomly swap first/last frames during training
        if bool(getattr(self.opt, 'train_swap_first_last', False)) and bool(getattr(self.opt, 'is_train', False)):
            p = float(getattr(self.opt, 'train_swap_prob', 0.5))
            if torch.rand((), device=self.device).item() < p:
                # Swap tensors used for training
                self.first_frame, self.last_frame = self.last_frame, self.first_frame
                # Swap indices in the data dict (if present) so later logic stays consistent
                if 'n_first_frame' in data_dict and 'n_last_frame' in data_dict:
                    data_dict['n_first_frame'], data_dict['n_last_frame'] = data_dict['n_last_frame'], data_dict['n_first_frame']

        if 'video' in data_dict and 'n_first_frame' in data_dict and 'n_last_frame' in data_dict:
            self.video_full = data_dict['video'].to(self.device)        # (B,T,1,D,H,W)
            self.video_full_ch = self.video_full[:, :, 0, ...]          # (B,T,D,H,W)

            self.n_first_frame = data_dict['n_first_frame'].item()
            self.n_last_frame = data_dict['n_last_frame'].item()

            self.video_interval = self.video_full_ch[:, self.n_first_frame:self.n_last_frame + 1, ...]
            self.video = self.video_interval
            self._interval_len = self.n_last_frame - self.n_first_frame + 1

    def forward(self):
        first_pyr = {}
        last_pyr = {}
        for s in self.ms_scales:
            s = int(s)
            first_pyr[s] = downsample_like(self.first_frame, s)
            last_pyr[s] = downsample_like(self.last_frame, s)

        feats0 = self.net_main.encoder(first_pyr)
        feats1 = self.net_main.encoder(last_pyr)

        self._first_pyr = first_pyr
        self._last_pyr = last_pyr

        # 0->1
        self._vel_0_1_by_scale = {}
        self._disp_0_1_by_scale = {}
        self._warp_0_1_by_scale = {}

        disp_prev = None
        for s in self.ms_scales:
            s = int(s)
            f0 = feats0[s]
            f1 = feats1[s]
            x = torch.cat([f0, f1, f0 - f1, (f0 - f1).abs()], dim=1)

            block = self.net_main.scale_blocks[str(s)]
            vel_s = block.predict_velocity(x)
            disp_s = block.integrate(vel_s)

            if disp_prev is None:
                disp_total = disp_s
            else:
                disp_prev_up = upsample_displacement_like(disp_prev, disp_s)
                disp_total = compose_displacements(disp_prev_up, disp_s, self.transformers[str(s)])

            warp_s = self.transformers[str(s)](
                first_pyr[s], mode='displacement', disp=disp_total, return_disp=False
            )

            self._vel_0_1_by_scale[s] = vel_s
            self._disp_0_1_by_scale[s] = disp_total
            self._warp_0_1_by_scale[s] = warp_s
            disp_prev = disp_total

        s_fin = int(self.ms_scales[-1])
        self.velocity_0_1 = self._vel_0_1_by_scale[s_fin]
        self.disp_0_1 = self._disp_0_1_by_scale[s_fin]
        self.warp_0_1 = self._warp_0_1_by_scale[s_fin]

        # 1->0
        if self.opt.use_bidirectional:
            self._vel_1_0_by_scale = {}
            self._disp_1_0_by_scale = {}
            self._warp_1_0_by_scale = {}

            disp_prev = None
            for s in self.ms_scales:
                s = int(s)
                f0 = feats0[s]
                f1 = feats1[s]
                x = torch.cat([f1, f0, f1 - f0, (f1 - f0).abs()], dim=1)

                block = self.net_main.scale_blocks[str(s)]
                vel_s = block.predict_velocity(x)
                disp_s = block.integrate(vel_s)

                if disp_prev is None:
                    disp_total = disp_s
                else:
                    disp_prev_up = upsample_displacement_like(disp_prev, disp_s)
                    disp_total = compose_displacements(disp_prev_up, disp_s, self.transformers[str(s)])

                warp_s = self.transformers[str(s)](
                    last_pyr[s], mode='displacement', disp=disp_total, return_disp=False
                )

                self._vel_1_0_by_scale[s] = vel_s
                self._disp_1_0_by_scale[s] = disp_total
                self._warp_1_0_by_scale[s] = warp_s
                disp_prev = disp_total

            self.velocity_1_0 = self._vel_1_0_by_scale[s_fin]
            self.disp_1_0 = self._disp_1_0_by_scale[s_fin]
            self.warp_1_0 = self._warp_1_0_by_scale[s_fin]

        # inference-only interpolation
        if self.opt.is_train:
            return
        if not hasattr(self, 'video_interval'):
            return

        T = self.video_interval.shape[1]
        frames = []
        for i in range(T):
            alpha = 0.0 if T == 1 else (float(i) / float(T - 1))
            pred_frame = self._predict_frame_at_alpha(alpha)  # (B,1,D,H,W)
            frames.append(pred_frame)

        self.video_pred = torch.cat(frames, dim=1)  # (B,T,D,H,W)
        self.video = self.video_interval

    def optimize_parameters(self):
        # Forward & loss under autocast (optional)
        with self._amp_autocast(enabled=self.use_amp):
            self.forward()

            loss_mse = torch.tensor(0.0, device=self.device)
            loss_ncc = torch.tensor(0.0, device=self.device)
            loss_charb = torch.tensor(0.0, device=self.device)

            for i, s in enumerate(self.ms_scales):
                s = int(s)
                w = float(self.ms_loss_weights[i])
                warp_s = self._warp_0_1_by_scale[s]
                tgt_s = self._last_pyr[s]

                ln_mse = self.criterion_mse(warp_s, tgt_s)

                ln_ncc = self.criterion_ncc(warp_s, tgt_s)
                lc = self.criterion_charb(warp_s, tgt_s)

                setattr(self, 'loss_mse_s%d' % i, ln_mse)
                setattr(self, 'loss_ncc_s%d' % i, ln_ncc)
                setattr(self, 'loss_charb_s%d' % i, lc)

                loss_mse = loss_mse + w * ln_mse
                loss_ncc = loss_ncc + w * ln_ncc
                loss_charb = loss_charb + w * lc

            if self.opt.use_bidirectional:
                for i, s in enumerate(self.ms_scales):
                    s = int(s)
                    w = float(self.ms_loss_weights[i])
                    warp_s = self._warp_1_0_by_scale[s]
                    tgt_s = self._first_pyr[s]

                    ln_mse = self.criterion_mse(warp_s, tgt_s)
                    ln_ncc = self.criterion_ncc(warp_s, tgt_s)
                    lc = self.criterion_charb(warp_s, tgt_s)

                    setattr(self, 'loss_mse_s%d' % i, getattr(self, 'loss_mse_s%d' % i) + ln_mse)
                    setattr(self, 'loss_ncc_s%d' % i, getattr(self, 'loss_ncc_s%d' % i) + ln_ncc)
                    setattr(self, 'loss_charb_s%d' % i, getattr(self, 'loss_charb_s%d' % i) + lc)

                    loss_mse = loss_mse + w * ln_mse
                    loss_ncc = loss_ncc + w * ln_ncc
                    loss_charb = loss_charb + w * lc

            loss_reg = torch.tensor(0.0, device=self.device)
            if self.lambda_reg > 0 and self.reg_loss is not None:
                loss_reg = self.reg_loss(self.velocity_0_1)
                if self.opt.use_bidirectional:
                    loss_reg = loss_reg + self.reg_loss(self.velocity_1_0)

            self.loss_mse = loss_mse
            self.loss_ncc = loss_ncc
            self.loss_charb = loss_charb
            self.loss_reg = loss_reg
            self.loss_main = float(getattr(self.opt, 'lambda_mse', 1.0)) * self.loss_mse + float(getattr(self.opt, 'lambda_ncc', 0.05)) * self.loss_ncc + float(getattr(self.opt, 'lambda_charb', 0.0)) * self.loss_charb + self.lambda_reg * self.loss_reg

        # Backward & step
        if self.use_amp:
            # zero grad
            for opt in getattr(self, 'optimizers', []):
                if hasattr(opt, 'zero_grad'):
                    opt.zero_grad()
                elif hasattr(opt, 'optimizer') and hasattr(opt.optimizer, 'zero_grad'):
                    opt.optimizer.zero_grad(set_to_none=True)
            # backward
            self._amp_scaler.scale(self.loss_main).backward()
            # step
            for opt in getattr(self, 'optimizers', []):
                base_opt = opt.optimizer if hasattr(opt, 'optimizer') else opt
                self._amp_scaler.step(base_opt)
            self._amp_scaler.update()
        else:
            self.optimizers_zero_grad()
            self.backward_loss(self.loss_main)
            self.optimizers_step_integration()

    def compute_visuals(self):
        slice_processor = SliceProcessor()

        if self.opt.is_train:
            # Compute differences BEFORE slicing to keep shapes consistent
            self.difference_0_1 = self.warp_0_1 - self.last_frame
            if self.opt.use_bidirectional:
                self.difference_1_0 = self.warp_1_0 - self.first_frame

            self.first_frame = slice_processor(self.first_frame)
            self.last_frame = slice_processor(self.last_frame)
            self.warp_0_1 = slice_processor(self.warp_0_1)
            self.difference_0_1 = slice_processor(self.difference_0_1)
            self.velocity_0_1 = slice_processor(self.velocity_0_1)
            self.disp_0_1 = slice_processor(self.disp_0_1)

            if self.opt.use_bidirectional:
                self.warp_1_0 = slice_processor(self.warp_1_0)
                self.difference_1_0 = slice_processor(self.difference_1_0)
                self.velocity_1_0 = slice_processor(self.velocity_1_0)
                self.disp_1_0 = slice_processor(self.disp_1_0)
            return

        if not hasattr(self, 'video') or not hasattr(self, 'video_pred'):
            return

        if self.video.shape[1] > 0:
            random_t_idx = torch.randint(0, self.video.shape[1], (1,)).item()
            self.video = self.video[:, random_t_idx: random_t_idx + 1, ...]
            self.video_pred = self.video_pred[:, random_t_idx: random_t_idx + 1, ...]
        else:
            self.video = torch.zeros_like(self.video_pred[:, 0:1, ...])
            self.video_pred = self.video_pred[:, 0:1, ...]

        self.difference = self.video_pred - self.video

        self.first_frame = slice_processor(self.first_frame)
        self.last_frame = slice_processor(self.last_frame)
        self.video = slice_processor(self.video)
        self.video_pred = slice_processor(self.video_pred)
        self.difference = slice_processor(self.difference)
        self.velocity_0_1 = slice_processor(self.velocity_0_1)
        self.disp_0_1 = slice_processor(self.disp_0_1)
        if self.opt.use_bidirectional:
            self.velocity_1_0 = slice_processor(self.velocity_1_0)
            self.disp_1_0 = slice_processor(self.disp_1_0)

    def update_metrics(self):
        if self.opt.cal_effective_rank:
            import math
            if not hasattr(self, 'rank_accum'):
                self.rank_accum = []
            self.rank_accum.append(effective_rank_velocity_field(self.velocity_0_1).mean().item())
            print('rank:', math.log10(sum(self.rank_accum) / len(self.rank_accum)))
            return

        if self.opt.is_train:
            return
        if not hasattr(self, 'metrics') or self.metrics is None:
            return
        if not hasattr(self, 'video') or not hasattr(self, 'video_pred'):
            return

        if self.video_pred.shape[1] != self.video.shape[1]:
            raise ValueError("video_pred and video length mismatch: pred=%d gt=%d" %
                             (self.video_pred.shape[1], self.video.shape[1]))

        for t in range(self.video.shape[1]):
            self.metrics.update(self.video_pred[:, t:t + 1, ...], self.video[:, t:t + 1, ...])

    # ----------------- interpolation helpers -----------------

    def _predict_frame_at_alpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0.0:
            return self.first_frame
        if alpha >= 1.0:
            return self.last_frame

        disp_f = self._disp_from_vel_by_scale(self._vel_0_1_by_scale, alpha)
        pred_f = self.transformers[str(int(self.ms_scales[-1]))](
            self.first_frame, mode='displacement', disp=disp_f, return_disp=False
        )

        if not self.opt.use_bidirectional:
            return pred_f

        beta = 1.0 - alpha
        disp_b = self._disp_from_vel_by_scale(self._vel_1_0_by_scale, beta)
        pred_b = self.transformers[str(int(self.ms_scales[-1]))](
            self.last_frame, mode='displacement', disp=disp_b, return_disp=False
        )

        eps = 1e-6
        w0 = 1.0 / (alpha + eps)
        w1 = 1.0 / (beta + eps)
        wsum = w0 + w1
        w0 = w0 / wsum
        w1 = w1 / wsum

        return pred_f * w0 + pred_b * w1

    def _disp_from_vel_by_scale(self, vel_by_scale, scale_factor):
        scale_factor = float(scale_factor)
        disp_prev = None
        for s in self.ms_scales:
            s = int(s)
            vel_s = vel_by_scale[s]
            block = self.net_main.scale_blocks[str(s)]
            disp_s = block.integrate(vel_s * scale_factor)

            if disp_prev is None:
                disp_total = disp_s
            else:
                disp_prev_up = upsample_displacement_like(disp_prev, disp_s)
                disp_total = compose_displacements(disp_prev_up, disp_s, self.transformers[str(s)])
            disp_prev = disp_total
        return disp_prev

    # ----------------- optimizer grouping -----------------

    def set_optimizers(self):
        from utils import optimizers as optim_utils
        from utils import schedulers
        from utils import lightning_fabric_utils

        model = self.net_main
        zero_keywords = list(getattr(self.opt, 'zero_weight_decay_names', ['R']))

        norm_param_names = set()
        if getattr(self.opt, 'set_bias_norm_zero_weight_decay', False):
            norm_param_names = optim_utils._get_norm_param_names(model)

        decay_params = []
        nodecay_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            force_no_decay = False
            for k in zero_keywords:
                if k in name:
                    force_no_decay = True
                    break
            is_bias = name.endswith('.bias')
            is_norm = (name in norm_param_names)
            if force_no_decay or is_bias or is_norm:
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = []
        if len(decay_params) > 0:
            param_groups.append({'params': decay_params, 'weight_decay': float(self.opt.optimizer_weight_decay)})
        if len(nodecay_params) > 0:
            param_groups.append({'params': nodecay_params, 'weight_decay': 0.0})

        self.optimizers = [optim_utils.CommonOptimizer(self.opt, param_groups)]
        self.schedulers = [schedulers.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if self.opt.use_lightning_fabric:
            lightning_fabric_utils.get_fabric().setup(model, self.optimizers[0])
