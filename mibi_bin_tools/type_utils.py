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
    return any(make_iterable(a))


def make_iterable(a: Union[type, Iterable[type]], ignore_str=True) -> Iterable[type]:
    """ Convert noniterable type to singelton in list

    Args:
        a (T | Iterable[T]):
            value or iterable of type T
        ignore_str (bool):
            whether to ignore the iterability of the str type

    Returns:
        List[T]:
            a as singleton in list, or a if a was already iterable.
    """
    return a if hasattr(a, '__iter__') and not (isinstance(a, str) and ignore_str) else [a]
