import abc
import inspect
import math
from functools import partial

import torch

from utils import parse_str


def get_scheduler_cls(lr_scheduler):
    """ Return the scheduler class

    :param lr_scheduler: the name of the learning rate scheduler
    """
    if lr_scheduler == 'constant':
        return ConstantScheduler
    elif lr_scheduler == 'linear':
        return LinearScheduler
    else:
        return CommonScheduler


def get_scheduler(optimizer, opt):
    """ Return a learning rate scheduler, which is defined by <opt.lr_scheduler>.
        For 'constant', we keep the same learning rate for the entire training process.
        For 'linear', we keep the same learning rate for the first <opt.decay_epochs_num> epochs,
        and linearly decay the rate to zero over the next <opt.epochs_num - opt.decay_epochs_num> epochs.
        And other schedulers defined in torch.optim.lr_scheduler are also supported.

    :param optimizer: the optimizer of the network
    :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions．　
    """
    scheduler_cls = get_scheduler_cls(opt.lr_scheduler)
    scheduler = scheduler_cls(opt, optimizer)
    # Optional iteration-based warmup (linear) controlled by opt.warmup_percentage in [0, 1]
    if float(getattr(opt, 'warmup_percentage', 0.0) or 0.0) > 0.0:
        scheduler = WarmupWrapperScheduler(opt, optimizer, scheduler)
    return scheduler


def get_option_setter(scheduler_name):
    """ Return the static method <modify_commandline_options> of the scheduler class."""
    dataset_class = get_scheduler_cls(scheduler_name)
    return partial(dataset_class.modify_commandline_options, lr_scheduler=scheduler_name)


class BaseScheduler(abc.ABC):
    """ This class is an abstract base class (ABC) for schedulers.
        To create a subclass, you need to implement the following five functions:
            -- <__init__>:                      initialize the class; first call BaseScheduler.__init__(self, opt).
            -- <step>:                          update learning rate and update network weights.
            -- <modify_commandline_options>:    (optionally) add model-specific options and set default optimizer options.
    """

    def __init__(self, opt, optimizer):
        """ Initialize the class; save options in the class"""
        self.optimizer = optimizer
        self.opt = opt

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        """ update learning rate and network weights"""
        pass

    @staticmethod
    def modify_commandline_options(parser, lr_scheduler):
        """Add new scheduler-specific options, and rewrite default values for existing options.

        :param parser:          original option parser
        :param lr_scheduler:    the name of learning rate scheduler
        :return parser:         the modified parser
        """
        return parser


class ConstantScheduler(BaseScheduler):
    """This scheduler keeps the learning rate constant"""

    @staticmethod
    def modify_commandline_options(parser, *args, **kwargs):
        return parser

    def __init__(self, opt, optimizer):
        super().__init__(opt, optimizer)

    def step(self):
        # keep lr unchanged
        pass


class LinearScheduler(BaseScheduler):
    """ In the first (epochs_num - decay_epochs_num) epochs, the lr is constant.
        In the last decay_epochs_num epochs, the lr is linearly decayed to 0.
    """

    @staticmethod
    def modify_commandline_options(parser, *args, **kwargs):
        parser.add_argument('--decay_epochs_num', type=int, required=True,
                            help='in the last decay_epochs_num epochs, the lr is linearly decayed to 0')
        return parser

    def __init__(self, opt, optimizer):
        super().__init__(opt, optimizer)
        # the call_times is assume as the current epoch number (just for convenience)
        self.called_times = 0

    def step(self):
        self.called_times += 1
        # if the current epoch number is larger than the decay_epochs_num, then decay the learning rate
        if self.called_times > self.opt.decay_epochs_num:
            # linearly decay the learning rate to 0
            lr = parse_str(self.opt.optimizer_lr) * (
                    self.opt.epochs_num - self.called_times + 1) / self.opt.decay_epochs_num
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # keep the learning rate
            pass


class WarmupWrapperScheduler(BaseScheduler):
    """
    A thin wrapper to add *iteration-based* warmup to an existing (typically epoch-based) scheduler.

    Usage pattern in this project:
      - train.py will call model.update_learning_rate() at *iteration-end* ONLY when warmup is enabled,
        with opt.lr_update_mode = "iter".
      - train.py will keep calling model.update_learning_rate() at *epoch-end* as before,
        with opt.lr_update_mode = "epoch".

    During warmup:
      - On "iter" updates: we manually set optimizer lrs following a linear warmup curve.
      - On "epoch" updates: we do NOT step the main scheduler (to keep it "frozen" until warmup ends).

    After warmup:
      - On "iter" updates: no-op.
      - On "epoch" updates: delegate to the main scheduler.
    """

    @staticmethod
    def modify_commandline_options(parser, lr_scheduler):
        # warmup_percentage is expected to be added by TrainOptions; we keep scheduler options unchanged here.
        return parser

    def __init__(self, opt, optimizer, main_scheduler: BaseScheduler):
        super().__init__(opt, optimizer)
        self.main_scheduler = main_scheduler

        warmup_pct = float(getattr(opt, 'warmup_percentage', 0.0) or 0.0)
        if warmup_pct <= 0.0:
            self.warmup_iters = 0
        else:
            total_iters = getattr(opt, 'total_iterations', None)
            if total_iters is None:
                raise ValueError(
                    "warmup_percentage>0 requires opt.total_iterations to be set (e.g., in train.py)."
                )
            total_iters = int(total_iters)
            self.warmup_iters = int(math.floor(total_iters * warmup_pct))
            # Guard against degenerate values like total_iters>0 but pct very small.
            if self.warmup_iters < 1 and total_iters > 0:
                self.warmup_iters = 1

        # Snapshot the target/base lrs (after optimizer creation).
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def _apply_warmup_lrs(self, cur_iter: int):
        """
        Linear warmup from 0 -> base_lr over warmup_iters steps.
        cur_iter is assumed to be 1-based (i.e., first iter is 1).
        """
        if self.warmup_iters <= 0:
            return
        # Clamp
        t = max(0, min(cur_iter, self.warmup_iters))
        scale = float(t) / float(self.warmup_iters)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def _warmup_done(self) -> bool:
        if self.warmup_iters <= 0:
            return True
        cur_iter = int(getattr(self.opt, 'current_iteration', 0) or 0)
        return cur_iter >= self.warmup_iters

    def step(self, *args, **kwargs):
        # We intentionally use opt as global "context" without an explicit hook argument.
        mode = getattr(self.opt, 'lr_update_mode', 'epoch')
        cur_iter = int(getattr(self.opt, 'current_iteration', 0) or 0)

        if not self._warmup_done():
            # Warmup phase
            if mode == 'iter':
                # Treat current_iteration as 1-based step count for warmup scaling.
                self._apply_warmup_lrs(cur_iter)
            # If epoch-end during warmup: do nothing (main scheduler stays frozen)
            return

        # After warmup: delegate only on epoch-end updates.
        if mode == 'epoch':
            self.main_scheduler.step(*args, **kwargs)
        # If iter-end after warmup: no-op.


def get_scheduler_cls_by_name(name):
    for scheduler_name, scheduler_cls in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass):
        if scheduler_name.lower() == name.lower():
            return scheduler_cls
    raise NotImplementedError('Scheduler [%s] not recognized.' % name)


class CommonScheduler(BaseScheduler):
    """ This scheduler is a wrapper of common schedulers in torch.optim.lr_scheduler"""

    @staticmethod
    def modify_commandline_options(parser, lr_scheduler):
        scheduler_cls = get_scheduler_cls_by_name(lr_scheduler)
        for arg_name, arg_parameter in inspect.signature(scheduler_cls.__init__).parameters.items():
            if arg_name == 'self' or arg_name == 'optimizer':
                continue
            if arg_parameter.default == arg_parameter.empty:
                parser.add_argument('--lr_scheduler_' + arg_name, required=True)
            else:
                parser.add_argument('--lr_scheduler_' + arg_name, default=arg_parameter.default)
        return parser

    def __init__(self, opt, optimizer):
        super(CommonScheduler, self).__init__(opt, optimizer)
        self.scheduler = get_scheduler_cls_by_name(opt.lr_scheduler)(
            optimizer,
            **{k.replace('lr_scheduler_', ''): parse_str(v)
               for k, v in vars(opt).items()
               if k.startswith('lr_scheduler_')}
        )

    def step(self, *args, **kwargs):
        self.scheduler.step(*args, **kwargs)
