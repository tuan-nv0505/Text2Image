import collections
from collections.abc import Iterable
from itertools import repeat

from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def _ntuple(n):
    """Return a function that converts input to an n-tuple.

    Scalar values are repeated n times, while iterables are converted to tuples.
    Strings are treated as scalars to avoid character-level splitting.

    Args:
        n: Target tuple length.

    Returns:
        Function that converts input to n-tuple.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple