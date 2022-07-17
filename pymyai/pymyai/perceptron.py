import numpy as np


def per_and(x1: int, x2: int) -> int:
    """
    >>> per_and(1, 1)
    1
    >>> per_and(1, 0)
    0
    >>> per_and(0, 1)
    0
    >>> per_and(0, 0)
    0
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7
    tmp = np.sum(w * x) - b
    if tmp > 0:
        return 1
    else:
        return 0


def per_nand(x1: int, x2: int) -> int:
    """
    >>> per_nand(1, 1)
    0
    >>> per_nand(1, 0)
    1
    >>> per_nand(0, 1)
    1
    >>> per_nand(0, 0)
    1
    """
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp > 0:
        return 1
    else:
        return 0


def per_or(x1: int, x2: int) -> int:
    """
    >>> per_or(1, 1)
    1
    >>> per_or(1, 0)
    1
    >>> per_or(0, 1)
    1
    >>> per_of(0, 0)
    0
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.2
    tmp = np.sum(w * x) - b
    if tmp > 0:
        return 1
    else:
        return 0


def per_xor(x1: int, x2: int) -> int:
    """
    >>> per_xor(1, 1)
    0
    >>> per_xor(1, 0)
    1
    >>> per_xor(0, 1)
    1
    >>> per_xor(0, 0)
    0
    """
    s1 = per_nand(x1, x2)
    s2 = per_or(x1, x2)
    y = per_and(s1, s2)
    return y
