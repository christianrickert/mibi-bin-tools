from typing import Union, Iterable


def any_true(a: Union[bool, Iterable[bool]]) -> bool:
    """ `any` that allows singleton values

    Args:
        a (bool | Iterable[bool]):
            value or iterable of booleans

    Returns:
        bool:
            whether any true values where found
    """
    return any(
        a if hasattr(a, '__iter__') else [a]
    )
