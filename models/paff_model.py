import random
import math
import os

import torch
import torch.nn.functional as F

from others.backbones.spatial_transformer import SpatialTransformer
from others.backbones.unet import Unet3d
from others.losses.registration_loss import NCCLoss, GradientLoss
from others.losses.vector_distance import CharbonnierLoss
from utils.utils_3d import SliceProcessor
from models.base_model import BaseModel


class PAFFModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(task='registration')

        parser.add_argument('--max_harmonic_order', type=int, default=4,
                            help='maximum harmonic order for velocity field encoding')
        parser.add_argument('--fourier_use_constant_term', dest='fourier_use_constant_term', action='store_true',
                            help='If True, include the constant term in the Fourier parameterization.')
        parser.add_argument('--fourier_disable_constant_term', dest='fourier_use_constant_term', action='store_false',
                            help='If set, disable the constant term in the Fourier parameterization.')
        parser.add_argument('--fourier_constant_term_only', action='store_true',
                            help='If True and fourier_use_constant_term is enabled, keep only the constant Fourier term '
                                 'and disable all higher-order cosine/sine terms.')
        parser.add_argument('--first_frame_t', type=float, default=0.0, help='time point of the first frame')
        parser.add_argument('--last_frame_t', type=float, default=0.5, help='time point of the last frame')

        parser.add_argument('--net_main_ngf', type=int, default=64,
                            help='number of generator filters in the last conv layer')
        parser.add_argument('--net_main_depth', type=int, default=5,
                            help='depth of the Unet3d backbone for main network')
        parser.add_argument('--net_main_use_checkpoint', action='store_true',
                            help='whether to use checkpointing in the main network to save memory')
        parser.add_argument('--net_main_skip_store_mode', type=str, default='dense', choices=['dense', 'per_level'],
                            help='skip connection storage mode for checkpointing in the main network')
        parser.add_argument('--net_refinement_ngf', type=int, default=16,
                            help='number of generator filters in the last conv layer of refinement network')
        parser.add_argument('--net_refinement_depth', type=int, default=3,
                            help='depth of the Unet3d backbone for refinement network')
        parser.add_argument('--net_refinement_use_checkpoint', action='store_true',
                            help='whether to use checkpointing in the refinement network to save memory')
        parser.add_argument('--net_refinement_skip_store_mode', type=str, default='dense',
                            choices=['dense', 'per_level'],
                            help='skip connection storage mode for checkpointing in the refinement network')

        parser.add_argument('--no_refinement', action='store_true',
                            help='If True, bypass refinement network and output the weighted sum of warped frames.')
        parser.add_argument('--refinement_do_residual', action='store_true',
                            help='whether the refinement network predicts residuals')
        parser.add_argument('--refinement_residual_random_weight', action='store_true',
                            help='whether to randomly weight first and last frame contributions when using residual refinement')
        parser.add_argument('--refinement_residual_activation', type=str, default='Identity',
                            choices=['tanh', 'Identity'],
                            help='activation function for residual refinement output')
        parser.add_argument('--refinement_residual_rescale_factor', type=float, default=1.0,
                            help='scaling factor for residual refinement output')

        parser.add_argument('--extrapolation_refine_from_last_only', action='store_true',
                            help='If True, during extrapolation, input to refinement net is cat(warped_last, warped_last) '
                                 'instead of cat(warped_first, warped_last).')
        parser.add_argument('--refinement_detach_warp_grad', action='store_true',
                            help='If True, detach warped inputs to refinement so loss_refine only updates net_refinement.')

        parser.add_argument('--b1_tau_grid_size', type=int, default=32,
                            help='Number of knots for tau(t) lookup table on [0,1] (>=8 recommended)')
        parser.add_argument('--b1_metric', type=str, default='dv_du_norm', choices=['v_norm', 'dv_du_norm'],
                            help='Metric s(u) used to build tau: v_norm=E||v||, dv_du_norm=E||∂v/∂u||')
        parser.add_argument('--b1_metric_downsample', type=int, default=2,
                            help='Downsample factor when computing B1 metric (>=1). Uses avg_pool3d on coeffs.')
        parser.add_argument('--b1_gamma', type=float, default=1.0,
                            help='Nonlinearity strength: rho(u) = (normalize(s(u))+eps)^gamma')
        parser.add_argument('--b1_rho_min', type=float, default=0.5,
                            help='Lower bound for rho(u)=tau\'(t) to avoid collapse')
        parser.add_argument('--b1_rho_max', type=float, default=2.0,
                            help='Upper bound for rho(u)=tau\'(t)')
        parser.add_argument('--b1_allow_tau_grad', dest='b1_detach_tau', action='store_false',
                            help='If set, allow gradients through tau construction (default: detach tau as tau=G(v))')
        parser.set_defaults(b1_detach_tau=True)

        parser.add_argument('--b1_disable_remap', action='store_true',
                            help='Disable B1 remap: use identity tau(t)=t (exact), skip building from metric.')

        parser.add_argument('--loss_ncc_window_size', type=int, nargs='+', default=[7, 7, 7],
                            help='window size for NCC loss computation')
        parser.add_argument('--loss_reg_alpha', type=float, default=1.5,
                            help='exponent alpha for harmonic coefficient regularization loss')
        parser.add_argument('--loss_refine_weight', type=float, default=1.0, help='weight for refinement loss')
        parser.add_argument('--loss_morph_weight', type=float, default=1.0, help='weight for morphing loss')
        parser.add_argument('--loss_morph_charb_weight', type=float, default=0.0,
                            help='extra weight for Charbonnier term in morphing loss (both sides); default 0 disables')
        parser.add_argument('--loss_cycle_weight', type=float, default=1.0, help='weight for cycle consistency loss')
        parser.add_argument('--loss_reg_weight', type=float, default=0.005, help='weight for regularization loss')
        parser.add_argument('--loss_gradient_weight', type=float, default=1.0,
                            help='weight for gradient loss on refined frames')
        parser.add_argument('--loss_refinement_only_high_bands', action='store_true',
                            help='if set, compute refinement loss only on high-frequency bands of the frames')

        parser.add_argument('--frame_random_num', type=int, default=None,
                            help='(train only) randomly sample K frames from a video for training to reduce memory.')
        parser.add_argument('--train_random_time_flip', action='store_true',
                            help='(train only) randomly flip video along time axis as online augmentation; '
                                 'when flipped, first and last are swapped accordingly.')

        parser.add_argument('--interval_extrapolation_start_frame', type=int, default=None,
                            help='Start frame offset relative to n_last_frame for extrapolation (e.g., 1).')
        parser.add_argument('--interval_extrapolation_end_frame', type=int, default=None,
                            help='End frame offset relative to n_last_frame. If >=1, enables extrapolation mode.')

        parser.add_argument('--test_export_time_reparam_fig1', action='store_true',
                            help='If set, export Figure 1 style SVG for time reparameterization analysis during test only.')
        parser.add_argument('--test_export_time_reparam_fig3', action='store_true',
                            help='If set, export Figure 3 style SVG for time reparameterization analysis during test only.')
        parser.add_argument('--test_export_time_reparam_dense_points', type=int, default=401,
                            help='Dense sampling points used to draw time reparameterization curves in test export.')
        parser.add_argument('--test_export_time_reparam_proxy_mode', type=str, default='highpass_cdiff',
                            choices=['raw_cdiff', 'highpass_cdiff'],
                            help='Proxy used in Figure 3 export during test only.')
        parser.add_argument('--test_export_time_reparam_output_dir', type=str,
                            default='/tmp/main6_time_reparam_analysis',
                            help='Absolute output directory for test-time time reparameterization SVG export.')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['loss_main', 'loss_refine', 'loss_morph', 'loss_cycle', 'loss_reg']
        self.model_names = ['net_main', 'net_refinement'] if not self.opt.no_refinement else ['net_main']
        self.visual_names = ['video', 'video_pred_refinement', 'video_pred_from_first', 'video_pred_from_last',
                             'difference', 'first_frame', 'last_frame', 'frame_cycle_from_first',
                             'frame_cycle_from_last']

        if self.opt.fourier_use_constant_term:
            self.visual_names.append('harmonic_coeff_constant')

        for order in range(1, self.opt.max_harmonic_order + 1):
            self.visual_names.extend([
                f'harmonic_coeff_order_{order}_cos',
                f'harmonic_coeff_order_{order}_sin',
            ])

        num_fourier_coeffs = 2 * self.opt.max_harmonic_order + (1 if self.opt.fourier_use_constant_term else 0)

        self.net_main = Unet3d(
            input_nc=2 * self.opt.input_nc,
            output_nc=3 * num_fourier_coeffs,
            ngf=self.opt.net_main_ngf,
            depth=self.opt.net_main_depth,
            backbone="modern",
            modern_cfg=dict(
                nc_multiplier=(1, 2, 4, 8),
                num_blocks=2,
                norm_type="group",
                activation="silu",
                attn_levels=(3,),
                num_heads=4,
                use_checkpoint=self.opt.net_main_use_checkpoint,
                skip_store_mode=self.opt.net_main_skip_store_mode,
                downsample_mode="conv",
                upsample_mode="nearest",
                align_mode="pad",
                out_activation="Identity",
            )
        ).to(self.device)

        if not self.opt.no_refinement:
            self.net_refinement = Unet3d(
                input_nc=2 * self.opt.input_nc,
                output_nc=self.opt.input_nc,
                ngf=self.opt.net_refinement_ngf,
                depth=self.opt.net_refinement_depth,
                backbone="modern",
                modern_cfg=dict(
                    nc_multiplier=(1, 2, 4),
                    num_blocks=1,
                    norm_type="group",
                    activation="silu",
                    attn_levels=(),
                    num_heads=4,
                    use_checkpoint=self.opt.net_refinement_use_checkpoint,
                    skip_store_mode=self.opt.net_refinement_skip_store_mode,
                    downsample_mode="conv",
                    upsample_mode="nearest",
                    align_mode="pad",
                    out_activation=self.opt.refinement_residual_activation if self.opt.refinement_do_residual else "Sigmoid"
                )
            ).to(self.device)

        self.spatial_transformer = SpatialTransformer(is_3d=opt.is_3d, mode='tvf').to(self.device)

        if self.opt.is_train:
            self.criterion_l1 = CharbonnierLoss().to(self.device)
            loss_ncc_window_size = self.opt.loss_ncc_window_size if len(self.opt.loss_ncc_window_size) != 1 else \
                self.opt.loss_ncc_window_size[0]
            self.criterion_ncc = NCCLoss(is_3d=True, window_size=loss_ncc_window_size, eps=1e-5)
            self.criterion_gradient = GradientLoss(is_3d=True).to(self.device)

        self._train_frame_indices = None
        self._is_extrapolating = False
        self._has_valid_gt = True

        self._time_reparam_export_case_dir = None
        self._time_reparam_export_dataset_name = None
        self._time_reparam_export_sample_id = None
        self._time_reparam_analysis_data = None

    def set_input(self, data_dict):
        self.image_paths = data_dict['video_path']
        self._time_reparam_analysis_data = None
        self._prepare_time_reparam_export_case_dir()
        raw_video = data_dict['video'].to(self.device)  

        n_first = data_dict['n_first_frame'].item()
        n_last = data_dict['n_last_frame'].item()

        if self.opt.is_train and getattr(self.opt, 'train_random_time_flip', False):
            if random.random() < 0.5:
                raw_video = torch.flip(raw_video, dims=[1])
                raw_len = raw_video.shape[1]
                old_n_first = n_first
                old_n_last = n_last
                n_first = raw_len - 1 - old_n_last
                n_last = raw_len - 1 - old_n_first

        self.first_frame = raw_video[:, n_first, 0, ...].unsqueeze(1)  
        self.last_frame = raw_video[:, n_last, 0, ...].unsqueeze(1)

        self.n_first_frame = n_first
        self.n_last_frame = n_last

        self._is_extrapolating = False
        self._has_valid_gt = True

        if (not self.opt.is_train) and (self.opt.interval_extrapolation_end_frame is not None) \
                and (self.opt.interval_extrapolation_end_frame >= 1):
            self._is_extrapolating = True

            extrap_start = self.opt.interval_extrapolation_start_frame
            if extrap_start is None or extrap_start <= 0:
                extrap_start = 1
            extrap_end = self.opt.interval_extrapolation_end_frame

            target_start_idx = n_last + extrap_start
            target_end_idx = n_last + extrap_end

            raw_len = raw_video.shape[1]

            if target_end_idx >= raw_len:
                self._has_valid_gt = False
                self.video = raw_video[:, n_first:n_last + 1, 0, ...]
            else:
                self.video = raw_video[:, target_start_idx: target_end_idx + 1, 0, ...]

        else:
            self.video = raw_video[:, n_first:n_last + 1, 0, ...]

    def forward(self):
        self.harmonic_coeffs = self.net_main(torch.cat([self.first_frame, self.last_frame], dim=1))
        num_fourier_coeffs = 2 * self.opt.max_harmonic_order + (1 if self.opt.fourier_use_constant_term else 0)
        self.harmonic_coeffs = self.harmonic_coeffs.reshape(
            self.harmonic_coeffs.shape[0],
            num_fourier_coeffs,
            3,
            self.harmonic_coeffs.shape[2],
            self.harmonic_coeffs.shape[3],
            self.harmonic_coeffs.shape[4],
        )

        if self.opt.fourier_use_constant_term and getattr(self.opt, 'fourier_constant_term_only', False):
            self.harmonic_coeffs = self.harmonic_coeffs.clone()
            self.harmonic_coeffs[:, 1:, ...] = 0

        if getattr(self.opt, 'b1_disable_remap', False) or float(getattr(self.opt, 'b1_gamma', 1.0)) == 0.0:
            self.tau_lut = build_identity_tau_lut(
                B=self.harmonic_coeffs.shape[0],
                K=int(self.opt.b1_tau_grid_size),
                device=self.harmonic_coeffs.device,
                dtype=self.harmonic_coeffs.dtype
            )
        else:
            if getattr(self.opt, 'b1_detach_tau', True):
                with torch.no_grad():
                    self.tau_lut = build_tau_lut_b1(self.harmonic_coeffs, self.opt)
            else:
                self.tau_lut = build_tau_lut_b1(self.harmonic_coeffs, self.opt)

        if self._should_export_time_reparam_analysis() and self.harmonic_coeffs.shape[0] == 1:
            self._time_reparam_analysis_data = collect_time_reparam_analysis_data(
                harmonic_coeffs=self.harmonic_coeffs.detach(),
                tau_lut=self.tau_lut.detach(),
                opt=self.opt,
                dense_points=int(getattr(self.opt, 'test_export_time_reparam_dense_points', 401)),
                t_start=float(self.opt.first_frame_t),
                t_end=float(self.opt.last_frame_t),
            )

        from_first_frame_list = []
        from_last_frame_list = []
        refinement_frame_list = []

        if self._is_extrapolating:
            extrap_start = self.opt.interval_extrapolation_start_frame
            if extrap_start is None or extrap_start <= 0:
                extrap_start = 1
            extrap_end = self.opt.interval_extrapolation_end_frame

            base_offset = self.n_last_frame - self.n_first_frame
            target_offsets = range(base_offset + extrap_start, base_offset + extrap_end + 1)

            frame_loop_indices = list(target_offsets)

        else:
            T_full = self.video.shape[1]

            self._train_frame_indices = None
            if self.opt.is_train and (self.opt.frame_random_num is not None) and (T_full > 0):
                K = int(self.opt.frame_random_num)
                K = max(1, min(K, T_full))
                if K < T_full:
                    idx = torch.randperm(T_full, device=self.video.device)[:K]
                    idx, _ = torch.sort(idx)
                    self._train_frame_indices = idx
                    self.video = self.video[:, idx, ...]
                    frame_loop_indices = [i.item() for i in idx]
                else:
                    frame_loop_indices = range(T_full)
            else:
                frame_loop_indices = range(T_full)

        frame_diff = self.n_last_frame - self.n_first_frame
        if frame_diff == 0:
            dt = 0.0
        else:
            dt = (self.opt.last_frame_t - self.opt.first_frame_t) / frame_diff

        for k_relative in frame_loop_indices:

            current_t = self.opt.first_frame_t + k_relative * dt

            warped_from_first = self.spatial_transformer(
                img=self.first_frame,
                t=current_t - self.opt.first_frame_t,
                integrator='euler',
                v_of_t=lambda t: cal_velocity_field_at_u(
                    u=apply_tau_lut(t + self.opt.first_frame_t, self.tau_lut, self.opt),
                    harmonic_coeffs=self.harmonic_coeffs,
                    opt=self.opt,
                ))

            warped_from_last = self.spatial_transformer(
                img=self.last_frame,
                t=current_t - self.opt.last_frame_t,
                integrator='euler',
                v_of_t=lambda t: cal_velocity_field_at_u(
                    u=apply_tau_lut(t + self.opt.last_frame_t, self.tau_lut, self.opt),
                    harmonic_coeffs=self.harmonic_coeffs,
                    opt=self.opt,
                ))

            if self.opt.no_refinement:
                first_distance = abs(current_t - self.opt.first_frame_t)
                last_distance = abs(self.opt.last_frame_t - current_t)
                denom = first_distance + last_distance + 1e-6
                refinement_frame = (last_distance / denom) * warped_from_first + \
                                   (first_distance / denom) * warped_from_last
            else:
                if self.opt.refinement_detach_warp_grad:
                    warped_from_first_for_refine = warped_from_first.detach()
                    warped_from_last_for_refine = warped_from_last.detach()
                else:
                    warped_from_first_for_refine = warped_from_first
                    warped_from_last_for_refine = warped_from_last

                if self.opt.refinement_do_residual:
                    if self.opt.refinement_residual_random_weight:
                        first_distance = random.uniform(0, 1)
                        last_distance = 1.0 - first_distance
                    else:
                        first_distance = abs(current_t - self.opt.first_frame_t)
                        last_distance = abs(self.opt.last_frame_t - current_t)

                    denom = first_distance + last_distance + 1e-6
                    combined_frame = (last_distance / denom) * warped_from_first_for_refine + \
                                     (first_distance / denom) * warped_from_last_for_refine

                    if self._is_extrapolating and self.opt.extrapolation_refine_from_last_only:
                        refinement_input = torch.cat([warped_from_last_for_refine, warped_from_last_for_refine], dim=1)
                    else:
                        refinement_input = torch.cat([warped_from_first_for_refine, warped_from_last_for_refine], dim=1)

                    refinement_frame = combined_frame + self.net_refinement(
                        refinement_input) * self.opt.refinement_residual_rescale_factor
                else:
                    if self._is_extrapolating and self.opt.extrapolation_refine_from_last_only:
                        refinement_input = torch.cat([warped_from_last_for_refine, warped_from_last_for_refine], dim=1)
                    else:
                        refinement_input = torch.cat([warped_from_first_for_refine, warped_from_last_for_refine], dim=1)

                    refinement_frame = self.net_refinement(refinement_input)

            from_first_frame_list.append(warped_from_first)
            from_last_frame_list.append(warped_from_last)
            refinement_frame_list.append(refinement_frame)

        self.video_pred_from_first = torch.cat(from_first_frame_list, dim=1)
        self.video_pred_from_last = torch.cat(from_last_frame_list, dim=1)
        self.video_pred_refinement = torch.cat(refinement_frame_list, dim=1)

        self.frame_cycle_from_first = self.spatial_transformer(
            img=self.first_frame,
            t=1,
            integrator='euler',
            v_of_t=lambda t: cal_velocity_field_at_u(
                u=apply_tau_lut(t + self.opt.first_frame_t, self.tau_lut, self.opt),
                harmonic_coeffs=self.harmonic_coeffs,
                opt=self.opt,
            ))
        self.frame_cycle_from_last = self.spatial_transformer(
            img=self.last_frame,
            t=1,
            integrator='euler',
            v_of_t=lambda t: -cal_velocity_field_at_u(
                u=apply_tau_lut(self.opt.last_frame_t - t, self.tau_lut, self.opt),
                harmonic_coeffs=self.harmonic_coeffs,
                opt=self.opt,
            ))

        self._export_time_reparam_analysis_if_needed(frame_loop_indices=frame_loop_indices, dt=dt)

    def _backward_loss_integration(self, loss, retain_graph=False):
        if self.opt.grad_accum is not None:
            loss = loss / self.opt.grad_accum

        self.backward_loss(loss, retain_graph=retain_graph)

    def _step_optimizers_integration(self):
        should_step = (self.opt.grad_accum is None) or (self.opt.current_iteration % self.opt.grad_accum == 0)
        if not should_step:
            return
        self.optimizers_step_integration()

    def compute_visuals(self):
        T_pred = self.video_pred_refinement.shape[1]
        if T_pred > 0:
            random_t_idx = torch.randint(0, T_pred, (1,)).item()

            self.video_pred_refinement_vis = self.video_pred_refinement[:, random_t_idx: random_t_idx + 1, ...]
            self.video_pred_from_first_vis = self.video_pred_from_first[:, random_t_idx: random_t_idx + 1, ...]
            self.video_pred_from_last_vis = self.video_pred_from_last[:, random_t_idx: random_t_idx + 1, ...]

            if self.video.shape[1] > random_t_idx:
                self.video_vis = self.video[:, random_t_idx: random_t_idx + 1, ...]
                self.difference_vis = self.video_pred_refinement_vis - self.video_vis
            else:
                self.video_vis = self.video_pred_refinement_vis
                self.difference_vis = self.video_pred_refinement_vis * 0
        else:
            return

        slice_processor = SliceProcessor()

        self.video = slice_processor(self.video_vis)
        self.video_pred_refinement = slice_processor(self.video_pred_refinement_vis)
        self.frame_cycle_from_first = slice_processor(self.frame_cycle_from_first)
        self.frame_cycle_from_last = slice_processor(self.frame_cycle_from_last)
        self.video_pred_from_first = slice_processor(self.video_pred_from_first_vis)
        self.video_pred_from_last = slice_processor(self.video_pred_from_last_vis)
        self.difference = slice_processor(self.difference_vis)
        self.first_frame = slice_processor(self.first_frame)
        self.last_frame = slice_processor(self.last_frame)

        coeff_offset = 0
        if self.opt.fourier_use_constant_term:
            self.harmonic_coeff_constant = slice_processor(self.harmonic_coeffs[:, 0, ...])
            coeff_offset = 1

        for order in range(1, self.opt.max_harmonic_order + 1):
            cos_idx = coeff_offset + (order - 1)
            sin_idx = coeff_offset + self.opt.max_harmonic_order + (order - 1)

            setattr(
                self,
                f'harmonic_coeff_order_{order}_cos',
                slice_processor(self.harmonic_coeffs[:, cos_idx, ...])
            )
            setattr(
                self,
                f'harmonic_coeff_order_{order}_sin',
                slice_processor(self.harmonic_coeffs[:, sin_idx, ...])
            )

    def optimize_parameters(self):
        self.forward()

        video_pred_for_loss, video_gt_for_loss = self.video_pred_refinement, self.video
        if self.opt.loss_refinement_only_high_bands:
            video_pred_for_loss = highpass_3d(self.video_pred_refinement, k=5, iters=2)
            video_gt_for_loss = highpass_3d(self.video, k=5, iters=2)
        self.loss_refine = (
                self.criterion_l1(video_pred_for_loss, video_gt_for_loss) + self.opt.loss_gradient_weight *
                self.criterion_gradient(video_pred_for_loss, video_gt_for_loss)
        )

        self.loss_morph = (
                self.criterion_ncc(self.video_pred_from_first, self.video) +
                self.criterion_ncc(self.video_pred_from_last, self.video)
        )
        if self.opt.loss_morph_charb_weight != 0.0:
            self.loss_morph = self.loss_morph + self.opt.loss_morph_charb_weight * (
                    self.criterion_l1(self.video_pred_from_first, self.video) +
                    self.criterion_l1(self.video_pred_from_last, self.video)
            )

        self.loss_cycle = (
                self.criterion_ncc(self.frame_cycle_from_first, self.first_frame) +
                self.criterion_ncc(self.frame_cycle_from_last, self.last_frame)
        )

        harmonic_orders = torch.arange(1, self.opt.max_harmonic_order + 1, device=self.device).float()
        reg_weights = harmonic_orders ** self.opt.loss_reg_alpha
        if self.harmonic_coeffs.shape[1] == 2 * self.opt.max_harmonic_order + 1:
            reg_weights = torch.cat([torch.zeros(1, device=self.device), reg_weights, reg_weights], dim=0)
        else:
            reg_weights = reg_weights.repeat(2)
        self.loss_reg = (reg_weights.unsqueeze(0) *
                         torch.mean(self.harmonic_coeffs ** 2, dim=[2, 3, 4, 5])).mean()

        self.loss_refine = self.loss_refine * self.opt.loss_refine_weight
        self.loss_morph = self.loss_morph * self.opt.loss_morph_weight
        self.loss_cycle = self.loss_cycle * self.opt.loss_cycle_weight
        self.loss_reg = self.loss_reg * self.opt.loss_reg_weight

        self.loss_main = self.loss_refine + self.loss_morph + self.loss_cycle + self.loss_reg

        self._backward_loss_integration(self.loss_main)
        self._step_optimizers_integration()

    def update_metrics(self):
        if self._is_extrapolating and (not self._has_valid_gt):
            return

        min_len = min(self.video.shape[1], self.video_pred_refinement.shape[1])
        if min_len == 0:
            return

        for i in range(min_len):
            self.metrics.update(self.video_pred_refinement[:, i:i + 1, ...], self.video[:, i:i + 1, ...])

    def _should_export_time_reparam_analysis(self):
        if self.opt.is_train:
            return False
        return bool(
            getattr(self.opt, 'test_export_time_reparam_fig1', False) or
            getattr(self.opt, 'test_export_time_reparam_fig3', False)
        )

    def _prepare_time_reparam_export_case_dir(self):
        self._time_reparam_export_case_dir = None
        self._time_reparam_export_dataset_name = None
        self._time_reparam_export_sample_id = None

        if not self._should_export_time_reparam_analysis():
            return

        video_path = _extract_first_path(self.image_paths)
        if video_path is None or video_path == '':
            return

        dataset_name_for_save = getattr(self.opt, 'save_dataset_name', None)
        if dataset_name_for_save is None or dataset_name_for_save == '':
            dataset_name_for_save = getattr(self.opt, 'dataset_name', 'dataset')

        sample_id = os.path.basename(os.path.dirname(video_path))
        root_dir = os.path.abspath(getattr(
            self.opt,
            'test_export_time_reparam_output_dir',
            '/tmp/main6_time_reparam_analysis'
        ))
        case_dir = os.path.join(root_dir, str(dataset_name_for_save), str(sample_id))
        os.makedirs(case_dir, exist_ok=True)

        self._time_reparam_export_case_dir = case_dir
        self._time_reparam_export_dataset_name = str(dataset_name_for_save)
        self._time_reparam_export_sample_id = str(sample_id)

    def _export_time_reparam_analysis_if_needed(self, frame_loop_indices, dt):
        if not self._should_export_time_reparam_analysis():
            return
        if self._time_reparam_export_case_dir is None:
            return
        if self._time_reparam_analysis_data is None:
            return
        if self.harmonic_coeffs.shape[0] != 1:
            return

        if getattr(self.opt, 'test_export_time_reparam_fig1', False):
            save_time_reparam_figure1_svg(
                analysis_data=self._time_reparam_analysis_data,
                save_path=os.path.join(self._time_reparam_export_case_dir, 'figure1_time_reparam_summary.svg'),
                dataset_name=self._time_reparam_export_dataset_name,
                sample_id=self._time_reparam_export_sample_id,
            )

        if getattr(self.opt, 'test_export_time_reparam_fig3', False):
            if self._is_extrapolating:
                return
            if self.video.shape[1] != len(frame_loop_indices):
                return
            if self.video.shape[1] < 3:
                return

            t_query = torch.tensor(
                [self.opt.first_frame_t + int(k) * float(dt) for k in frame_loop_indices],
                device=self.tau_lut.device,
                dtype=self.tau_lut.dtype
            )
            tau_query = apply_tau_lut_to_grid(t_query, self.tau_lut, self.opt)
            rho_eff_query = finite_difference_batch(tau_query, t_query)[0]
            proxy_query = compute_temporal_change_proxy(
                self.video.detach(),
                mode=getattr(self.opt, 'test_export_time_reparam_proxy_mode', 'highpass_cdiff')
            )[0]

            save_time_reparam_figure3_svg(
                t_query=t_query.detach().cpu(),
                rho_eff_query=rho_eff_query.detach().cpu(),
                proxy_query=proxy_query.detach().cpu(),
                save_path=os.path.join(self._time_reparam_export_case_dir, 'figure3_rho_proxy_scatter.svg'),
                dataset_name=self._time_reparam_export_dataset_name,
                sample_id=self._time_reparam_export_sample_id,
                proxy_mode=getattr(self.opt, 'test_export_time_reparam_proxy_mode', 'highpass_cdiff'),
            )


def _as_tensor_like(x, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def cal_velocity_field_at_u(u=0.0, harmonic_coeffs=None, opt=None):
    """Finite Fourier parameterization of v(x,u) on the periodic phase axis u.

    harmonic_coeffs: (B, 2N, 3, D, H, W) or (B, 2N+1, 3, D, H, W) where N=max_harmonic_order
    u: scalar tensor/float or (B,) tensor. Values can be outside [0,1]; periodicity holds.
    returns: (B, 3, D, H, W)
    """
    B = harmonic_coeffs.shape[0]
    N = opt.max_harmonic_order
    num_coeffs = harmonic_coeffs.shape[1]
    if num_coeffs == 2 * N + 1:
        has_constant_term = True
    elif num_coeffs == 2 * N:
        has_constant_term = False
    else:
        raise ValueError(f"Unexpected harmonic coeff shape: {tuple(harmonic_coeffs.shape)}")

    n = torch.arange(1, N + 1, dtype=harmonic_coeffs.dtype, device=harmonic_coeffs.device)
    u = _as_tensor_like(u, harmonic_coeffs)
    if u.dim() == 0:
        u = u.expand(B)
    elif u.dim() == 1 and u.shape[0] != B:
        u = u.expand(B)

    cos_vals = torch.cos(2 * math.pi * u[:, None] * n[None, :])
    sin_vals = torch.sin(2 * math.pi * u[:, None] * n[None, :])

    cos_terms = cos_vals[:, :, None, None, None, None]
    sin_terms = sin_vals[:, :, None, None, None, None]

    offset = 1 if has_constant_term else 0
    a = harmonic_coeffs[:, offset:offset + N, :, ...]
    b = harmonic_coeffs[:, offset + N:offset + 2 * N, :, ...]

    v = torch.sum(
        a * cos_terms + b * sin_terms,
        dim=1,
        keepdim=False
    )
    if has_constant_term:
        v = v + harmonic_coeffs[:, 0, :, ...]
    return v


def cal_dv_du_at_u(u=0.0, harmonic_coeffs=None, opt=None):
    """Analytic derivative ∂v/∂u under the finite Fourier parameterization."""
    B = harmonic_coeffs.shape[0]
    N = opt.max_harmonic_order
    num_coeffs = harmonic_coeffs.shape[1]
    if num_coeffs == 2 * N + 1:
        offset = 1
    elif num_coeffs == 2 * N:
        offset = 0
    else:
        raise ValueError(f"Unexpected harmonic coeff shape: {tuple(harmonic_coeffs.shape)}")

    n = torch.arange(1, N + 1, dtype=harmonic_coeffs.dtype, device=harmonic_coeffs.device)
    u = _as_tensor_like(u, harmonic_coeffs)
    if u.dim() == 0:
        u = u.expand(B)
    elif u.dim() == 1 and u.shape[0] != B:
        u = u.expand(B)

    cos_vals = torch.cos(2 * math.pi * u[:, None] * n[None, :])
    sin_vals = torch.sin(2 * math.pi * u[:, None] * n[None, :])

    w = (2 * math.pi * n)[None, :]
    sin_terms = (sin_vals * w)[:, :, None, None, None, None]
    cos_terms = (cos_vals * w)[:, :, None, None, None, None]

    a = harmonic_coeffs[:, offset:offset + N, :, ...]
    b = harmonic_coeffs[:, offset + N:offset + 2 * N, :, ...]

    dv_du = torch.sum(
        (-a) * sin_terms + b * cos_terms,
        dim=1,
        keepdim=False
    )
    return dv_du


def _downsample_coeffs_for_metric(harmonic_coeffs: torch.Tensor, ds: int) -> torch.Tensor:
    """Downsample coeffs with avg pooling for cheaper metric computation."""
    if ds is None or ds <= 1:
        return harmonic_coeffs
    B, num_coeffs, C, D, H, W = harmonic_coeffs.shape
    x = harmonic_coeffs.reshape(B, num_coeffs * C, D, H, W)
    x = F.avg_pool3d(x, kernel_size=ds, stride=ds, padding=0)
    D2, H2, W2 = x.shape[-3:]
    return x.reshape(B, num_coeffs, C, D2, H2, W2)


def build_tau_lut_b1(harmonic_coeffs: torch.Tensor, opt) -> torch.Tensor:
    """Build a per-sample tau(t) lookup table on [0,1] using B1 (deterministic) scheme."""
    K = int(opt.b1_tau_grid_size)
    assert K >= 8, "b1_tau_grid_size should be >= 8"
    device = harmonic_coeffs.device
    dtype = harmonic_coeffs.dtype

    t_grid = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype)
    dt = 1.0 / K

    coeffs_metric = _downsample_coeffs_for_metric(harmonic_coeffs, int(opt.b1_metric_downsample))

    B = harmonic_coeffs.shape[0]
    s_vals = torch.zeros((B, K + 1), device=device, dtype=dtype)

    for i in range(K + 1):
        ti = t_grid[i]
        if opt.b1_metric == 'dv_du_norm':
            v_i = cal_dv_du_at_u(u=ti, harmonic_coeffs=coeffs_metric, opt=opt)
        else:
            v_i = cal_velocity_field_at_u(u=ti, harmonic_coeffs=coeffs_metric, opt=opt)

        s_i = torch.sqrt(torch.sum(v_i ** 2, dim=1) + 1e-12)  
        s_vals[:, i] = s_i.mean(dim=[1, 2, 3])

    eps = 1e-6
    s_vals = s_vals / (s_vals.mean(dim=1, keepdim=True) + eps)

    gamma = float(opt.b1_gamma)
    rho = torch.clamp((s_vals + eps) ** gamma, min=float(opt.b1_rho_min), max=float(opt.b1_rho_max))

    tau_raw = torch.zeros((B, K + 1), device=device, dtype=dtype)
    for i in range(1, K + 1):
        tau_raw[:, i] = tau_raw[:, i - 1] + 0.5 * (rho[:, i - 1] + rho[:, i]) * dt

    total = tau_raw[:, -1:].clamp_min(eps)
    tau_raw = tau_raw / total  

    a = float(opt.last_frame_t)
    a = max(0.0, min(1.0, a))

    pos = a * K
    i0 = int(math.floor(pos))
    i1 = min(i0 + 1, K)
    w = pos - i0
    tau_a = ((1.0 - w) * tau_raw[:, i0] + w * tau_raw[:, i1]).clamp_min(eps).clamp_max(1.0 - eps)

    tau = tau_raw.clone()
    for i in range(K + 1):
        ti = float(i) / K
        if ti <= a + 1e-12:
            tau[:, i] = (a / tau_a) * tau_raw[:, i]
        else:
            tau[:, i] = a + ((1.0 - a) / (1.0 - tau_a)) * (tau_raw[:, i] - tau_a)

    tau[:, 0] = 0.0
    tau[:, -1] = 1.0
    return tau


def build_identity_tau_lut(B: int, K: int, device, dtype) -> torch.Tensor:
    """
    Exact identity LUT: tau_grid[i] = i/K for i=0..K, replicated for each sample.
    This makes apply_tau_lut(t) return t (with the same periodic handling) exactly.
    """
    grid = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype)[None, :]  
    return grid.repeat(B, 1)


def apply_tau_lut(t_abs, tau_lut: torch.Tensor, opt) -> torch.Tensor:
    """Map observation time t_abs to phase coordinate u = tau(t_abs) via per-sample LUT."""
    K = int(opt.b1_tau_grid_size)
    B = tau_lut.shape[0]
    t = _as_tensor_like(t_abs, tau_lut)

    if t.dim() == 0:
        t = t.expand(B)
    elif t.dim() == 1 and t.shape[0] != B:
        t = t.expand(B)

    t_floor = torch.floor(t)
    t_frac = torch.remainder(t - t_floor, 1.0)

    pos = t_frac * K
    i0 = torch.floor(pos).long().clamp(0, K)
    i1 = (i0 + 1).clamp(0, K)
    w = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)

    tau0 = torch.gather(tau_lut, 1, i0[:, None]).squeeze(1)
    tau1 = torch.gather(tau_lut, 1, i1[:, None]).squeeze(1)
    tau_frac = (1.0 - w) * tau0 + w * tau1

    return t_floor + tau_frac


def _extract_first_path(image_paths):
    if isinstance(image_paths, str):
        return image_paths
    if isinstance(image_paths, (list, tuple)) and len(image_paths) > 0:
        return image_paths[0]
    return None


def interp_uniform_grid_1d(y_knots: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    if x_query.dim() != 1:
        raise ValueError(f'x_query should be 1D, got shape {tuple(x_query.shape)}')
    B, Kp1 = y_knots.shape
    K = Kp1 - 1

    xq = x_query.to(device=y_knots.device, dtype=y_knots.dtype).clamp(0.0, 1.0)
    pos = xq * K
    i0 = torch.floor(pos).long().clamp(0, K)
    i1 = (i0 + 1).clamp(0, K)
    w = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)

    gather_i0 = i0.unsqueeze(0).expand(B, -1)
    gather_i1 = i1.unsqueeze(0).expand(B, -1)
    y0 = torch.gather(y_knots, 1, gather_i0)
    y1 = torch.gather(y_knots, 1, gather_i1)
    return (1.0 - w.unsqueeze(0)) * y0 + w.unsqueeze(0) * y1


def apply_tau_lut_to_grid(t_grid: torch.Tensor, tau_lut: torch.Tensor, opt) -> torch.Tensor:
    if t_grid.dim() != 1:
        raise ValueError(f't_grid should be 1D, got shape {tuple(t_grid.shape)}')

    K = int(opt.b1_tau_grid_size)
    B = tau_lut.shape[0]
    t = t_grid.to(device=tau_lut.device, dtype=tau_lut.dtype).unsqueeze(0).expand(B, -1)

    t_floor = torch.floor(t)
    t_frac = torch.remainder(t - t_floor, 1.0)

    pos = t_frac * K
    i0 = torch.floor(pos).long().clamp(0, K)
    i1 = (i0 + 1).clamp(0, K)
    w = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)

    tau0 = torch.gather(tau_lut, 1, i0)
    tau1 = torch.gather(tau_lut, 1, i1)
    tau_frac = (1.0 - w) * tau0 + w * tau1
    return t_floor + tau_frac


def cumulative_trapezoid_uniform(y: torch.Tensor, x_start: float, x_end: float) -> torch.Tensor:
    if y.dim() != 2:
        raise ValueError(f'y should be 2D, got shape {tuple(y.shape)}')
    B, P = y.shape
    out = torch.zeros_like(y)
    if P <= 1:
        return out
    step = (float(x_end) - float(x_start)) / float(P - 1)
    increments = 0.5 * (y[:, 1:] + y[:, :-1]) * step
    out[:, 1:] = torch.cumsum(increments, dim=1)
    return out


def finite_difference_batch(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if y.dim() != 2:
        raise ValueError(f'y should be 2D, got shape {tuple(y.shape)}')
    if x.dim() != 1:
        raise ValueError(f'x should be 1D, got shape {tuple(x.shape)}')

    x = x.to(device=y.device, dtype=y.dtype)
    B, P = y.shape
    out = torch.zeros_like(y)
    if P <= 1:
        return out

    dx0 = (x[1] - x[0]).abs().clamp_min(1e-12)
    dx1 = (x[-1] - x[-2]).abs().clamp_min(1e-12)
    out[:, 0] = (y[:, 1] - y[:, 0]) / dx0
    out[:, -1] = (y[:, -1] - y[:, -2]) / dx1

    if P > 2:
        dx_mid = (x[2:] - x[:-2]).abs().clamp_min(1e-12)
        out[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / dx_mid.unsqueeze(0)

    return out


def collect_time_reparam_analysis_data(harmonic_coeffs: torch.Tensor, tau_lut: torch.Tensor, opt, dense_points: int,
                                       t_start: float, t_end: float):
    dense_points = max(33, int(dense_points))
    K = int(opt.b1_tau_grid_size)
    device = harmonic_coeffs.device
    dtype = harmonic_coeffs.dtype

    u_knots = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype)
    coeffs_metric = _downsample_coeffs_for_metric(harmonic_coeffs, int(opt.b1_metric_downsample))

    B = harmonic_coeffs.shape[0]
    s_raw_knots = torch.zeros((B, K + 1), device=device, dtype=dtype)
    for i in range(K + 1):
        ui = u_knots[i]
        if opt.b1_metric == 'dv_du_norm':
            metric_field = cal_dv_du_at_u(u=ui, harmonic_coeffs=coeffs_metric, opt=opt)
        else:
            metric_field = cal_velocity_field_at_u(u=ui, harmonic_coeffs=coeffs_metric, opt=opt)
        metric_mag = torch.sqrt(torch.sum(metric_field ** 2, dim=1) + 1e-12)
        s_raw_knots[:, i] = metric_mag.mean(dim=[1, 2, 3])

    eps = 1e-6
    s_norm_knots = s_raw_knots / (s_raw_knots.mean(dim=1, keepdim=True) + eps)
    rho_knots = torch.clamp(
        (s_norm_knots + eps) ** float(opt.b1_gamma),
        min=float(opt.b1_rho_min),
        max=float(opt.b1_rho_max)
    )

    u_dense = torch.linspace(0.0, 1.0, dense_points, device=device, dtype=dtype)
    s_raw_dense = interp_uniform_grid_1d(s_raw_knots, u_dense)
    s_norm_dense = interp_uniform_grid_1d(s_norm_knots, u_dense)
    rho_dense = interp_uniform_grid_1d(rho_knots, u_dense)

    psi_dense = cumulative_trapezoid_uniform(rho_dense, 0.0, 1.0)
    psi_dense = psi_dense / psi_dense[:, -1:].clamp_min(eps)

    t_dense = torch.linspace(float(t_start), float(t_end), dense_points, device=device, dtype=dtype)
    tau_dense = apply_tau_lut_to_grid(t_dense, tau_lut, opt)
    dtau_dt_dense = finite_difference_batch(tau_dense, t_dense)

    return {
        'u_knots': u_knots.detach(),
        's_raw_knots': s_raw_knots.detach(),
        's_norm_knots': s_norm_knots.detach(),
        'rho_knots': rho_knots.detach(),
        'u_dense': u_dense.detach(),
        's_raw_dense': s_raw_dense.detach(),
        's_norm_dense': s_norm_dense.detach(),
        'rho_dense': rho_dense.detach(),
        'psi_dense': psi_dense.detach(),
        't_dense': t_dense.detach(),
        'tau_dense': tau_dense.detach(),
        'dtau_dt_dense': dtau_dt_dense.detach(),
    }


def compute_temporal_change_proxy(video: torch.Tensor, mode: str = 'highpass_cdiff') -> torch.Tensor:
    if video.dim() != 5:
        raise ValueError(f'Expected video shape (B,T,D,H,W), got {tuple(video.shape)}')

    x = video
    if mode == 'highpass_cdiff':
        x = highpass_3d(video.unsqueeze(2), k=5, iters=2).squeeze(2)
    elif mode == 'raw_cdiff':
        x = video
    else:
        raise ValueError(f'Unknown proxy mode: {mode}')

    B, T, D, H, W = x.shape
    proxy = torch.zeros((B, T), device=x.device, dtype=x.dtype)
    if T <= 1:
        return proxy

    if T >= 3:
        diff_mid = torch.abs(x[:, 2:, ...] - x[:, :-2, ...])
        proxy[:, 1:-1] = diff_mid.mean(dim=[2, 3, 4]) * 0.5

    diff_left = torch.abs(x[:, 1, ...] - x[:, 0, ...])
    diff_right = torch.abs(x[:, -1, ...] - x[:, -2, ...])
    proxy[:, 0] = diff_left.mean(dim=[1, 2, 3])
    proxy[:, -1] = diff_right.mean(dim=[1, 2, 3])
    return proxy


def compute_linear_fit_and_pearson(x: torch.Tensor, y: torch.Tensor):
    x = x.flatten().to(torch.float32)
    y = y.flatten().to(torch.float32)

    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean

    denom_x = torch.sum(x_centered ** 2).clamp_min(1e-12)
    denom_y = torch.sum(y_centered ** 2).clamp_min(1e-12)

    slope = torch.sum(x_centered * y_centered) / denom_x
    intercept = y_mean - slope * x_mean
    pearson = torch.sum(x_centered * y_centered) / torch.sqrt(denom_x * denom_y)
    return slope.item(), intercept.item(), pearson.item()


def save_time_reparam_figure1_svg(analysis_data, save_path: str, dataset_name: str, sample_id: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    u_dense = analysis_data['u_dense'].detach().cpu().numpy()
    s_raw_dense = analysis_data['s_raw_dense'][0].detach().cpu().numpy()
    rho_dense = analysis_data['rho_dense'][0].detach().cpu().numpy()
    psi_dense = analysis_data['psi_dense'][0].detach().cpu().numpy()
    t_dense = analysis_data['t_dense'].detach().cpu().numpy()
    tau_dense = analysis_data['tau_dense'][0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.plot(u_dense, s_raw_dense, linewidth=2.0)
    ax.set_xlabel('u')
    ax.set_ylabel('s(u)')
    ax.set_title('Metric curve')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(u_dense, rho_dense, linewidth=2.0)
    ax.set_xlabel('u')
    ax.set_ylabel('rho(u)')
    ax.set_title('Allocation weight')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(u_dense, psi_dense, linewidth=2.0)
    ax.plot(u_dense, u_dense, linestyle='--', linewidth=1.2)
    ax.set_xlabel('u')
    ax.set_ylabel('Psi(u)')
    ax.set_title('Cumulative mapping')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_dense, tau_dense, linewidth=2.0, label='tau(t)')
    ax.plot(t_dense, t_dense, linestyle='--', linewidth=1.2, label='identity')
    ax.set_xlabel('t')
    ax.set_ylabel('tau(t)')
    ax.set_title('Observed time to phase time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f'{dataset_name} / {sample_id}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def save_time_reparam_figure3_svg(t_query: torch.Tensor, rho_eff_query: torch.Tensor, proxy_query: torch.Tensor,
                                  save_path: str, dataset_name: str, sample_id: str, proxy_mode: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if t_query.numel() < 3:
        return

    x = rho_eff_query[1:-1].detach().cpu()
    y = proxy_query[1:-1].detach().cpu()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() < 2:
        return

    slope, intercept, pearson = compute_linear_fit_and_pearson(x, y)

    x_np = x.numpy()
    y_np = y.numpy()

    x_line = torch.linspace(float(x.min()), float(x.max()), 200, dtype=x.dtype, device=x.device)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_np, y_np)
    ax.plot(x_line.numpy(), y_line.numpy(), linewidth=2.0)
    ax.set_xlabel('rho_eff(t) = d tau / d t')
    ax.set_ylabel('temporal change proxy')
    ax.set_title(f'{dataset_name} / {sample_id}')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'Pearson r = {pearson:.4f}\nproxy = {proxy_mode}',
            transform=ax.transAxes, va='top', ha='left')
    fig.tight_layout()
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def lowpass_avgpool_3d(x: torch.Tensor, k: int = 5, iters: int = 2) -> torch.Tensor:
    """
    x: (B, T, 1, D, H, W) or (B, 1, D, H, W)
    Applies separable-ish low-pass via repeated avg_pool3d (stride=1).
    """
    if x.dim() == 6:  
        B, T, C, D, H, W = x.shape
        y = x.reshape(B * T, C, D, H, W)
        for _ in range(iters):
            y = F.avg_pool3d(y, kernel_size=k, stride=1, padding=k // 2)
        return y.reshape(B, T, C, D, H, W)
    elif x.dim() == 5:  
        y = x
        for _ in range(iters):
            y = F.avg_pool3d(y, kernel_size=k, stride=1, padding=k // 2)
        return y
    else:
        raise ValueError(f"Unexpected shape: {tuple(x.shape)}")


def highpass_3d(x: torch.Tensor, k: int = 5, iters: int = 2) -> torch.Tensor:
    return x - lowpass_avgpool_3d(x, k=k, iters=iters)