import inspect
import torch

from utils import get_class_from_subclasses, parse_str


# -----------------------------
# Param-group helpers
# -----------------------------
def _get_norm_param_names(model: torch.nn.Module):
    """
    Return a set of full parameter names that belong to normalization layers.
    Includes parameters directly owned by the norm module (recurse=False) only.
    """
    norm_types = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
    )

    norm_param_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            for p_name, _p in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{p_name}" if module_name else p_name
                norm_param_names.add(full_name)
    return norm_param_names


def build_param_groups_with_optional_zero_decay_for_bias_norm(
    model: torch.nn.Module,
    weight_decay: float,
    set_bias_norm_zero_weight_decay: bool,
):
    """
    If set_bias_norm_zero_weight_decay is True and weight_decay > 0:
      - Split parameters into decay / no_decay groups.
      - no_decay includes biases + all norm layer params (BN/GN/IN/LN).
      - decay uses the provided weight_decay
    Else:
      - Return a flat list(model.parameters()) (single weight_decay in optimizer kwargs).
    """
    if weight_decay is None:
        weight_decay = 0.0
    weight_decay = float(weight_decay)

    if (not bool(set_bias_norm_zero_weight_decay)) or weight_decay <= 0.0:
        return list(model.parameters())

    norm_param_names = _get_norm_param_names(model)

    decay_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or (name in norm_param_names):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = []
    if len(decay_params) > 0:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if len(no_decay_params) > 0:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return param_groups


# -----------------------------
# Optimizer wrapper (Fix B)
# -----------------------------
class CommonOptimizer(torch.optim.Optimizer):
    """
    A wrapper that is a torch.optim.Optimizer (passes isinstance checks),
    delegating behavior to an inner torch optimizer.

    IMPORTANT: During torch.optim.Optimizer.__init__, it will call self.add_param_group().
    We MUST avoid delegating add_param_group to inner before self.optimizer is stable,
    otherwise recursion occurs.
    """

    @staticmethod
    def modify_commandline_options(parser, optimizer_name):
        optimizer_cls = get_class_from_subclasses(
            torch.optim.Optimizer, optimizer_name, allow_case=True, allow_underline=False
        )

        parser.add_argument(
            "--set_bias_norm_zero_weight_decay",
            action="store_true",
            help="If set, bias and all normalization parameters (BN/GN/IN/LN) will use weight_decay=0; "
                 "other params use optimizer_weight_decay. Default: False."
        )

        for arg_name, arg_parameter in inspect.signature(optimizer_cls.__init__).parameters.items():
            if arg_name in ["self", "params"]:
                continue

            arg_flag = "--optimizer_" + arg_name

            if arg_parameter.default != inspect.Parameter.empty and isinstance(arg_parameter.default, (list, tuple)):
                parser.add_argument(arg_flag, default=arg_parameter.default, nargs="+")
                continue

            if arg_parameter.default == inspect.Parameter.empty:
                parser.add_argument(arg_flag, required=True)
            else:
                parser.add_argument(arg_flag, default=arg_parameter.default)

        return parser

    @staticmethod
    def _coerce_optimizer_arg(raw_value, default_value):
        if isinstance(raw_value, list):
            parsed = [parse_str(x) for x in raw_value]
            if isinstance(default_value, tuple):
                return tuple(parsed)
            return parsed
        parsed = parse_str(raw_value)
        if isinstance(default_value, tuple) and isinstance(parsed, list):
            return tuple(parsed)
        if isinstance(default_value, list) and isinstance(parsed, tuple):
            return list(parsed)
        return parsed

    def __init__(self, opt, params):
        optimizer_cls = get_class_from_subclasses(
            torch.optim.Optimizer, opt.optimizer, allow_case=True, allow_underline=False
        )

        sig = inspect.signature(optimizer_cls.__init__).parameters
        defaults = {
            name: p.default
            for name, p in sig.items()
            if name not in ["self", "params"]
        }

        kwargs = {}
        for k, v in vars(opt).items():
            if not k.startswith("optimizer_"):
                continue
            arg_name = k.replace("optimizer_", "")
            default_value = defaults.get(arg_name, inspect.Parameter.empty)
            if default_value == inspect.Parameter.empty:
                kwargs[arg_name] = parse_str(v)
            else:
                kwargs[arg_name] = self._coerce_optimizer_arg(v, default_value)

        # Build inner torch optimizer first
        inner = optimizer_cls(params, **kwargs)

        # Mark we are in base init, and set optimizer early to avoid __getattr__ recursion
        self.optimizer = inner
        self._in_base_init = True

        # Call base Optimizer init, but ensure our add_param_group won't delegate during this phase
        super().__init__(inner.param_groups, inner.defaults)

        # Base init done
        self._in_base_init = False

        # Bind live storages so schedulers and others see correct values
        self.param_groups = inner.param_groups
        self.state = inner.state
        self.defaults = inner.defaults

    # Delegate core APIs
    def step(self, closure=None):
        return self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        """
        During torch.optim.Optimizer.__init__, this method is called.
        We must NOT delegate to inner then; use base implementation.
        After init, delegate to inner and rebind storages.
        """
        if getattr(self, "_in_base_init", False) or not hasattr(self, "optimizer"):
            # Use base implementation to avoid touching self.optimizer during base init
            return torch.optim.Optimizer.add_param_group(self, param_group)

        out = self.optimizer.add_param_group(param_group)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        return out

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        out = self.optimizer.load_state_dict(state_dict)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        return out

    def __getattr__(self, name):
        # Avoid recursion if optimizer isn't ready
        if name in ("optimizer", "_in_base_init"):
            raise AttributeError(name)
        return getattr(self.optimizer, name)
