"""In place permutations and matrix mutiplication."""

import typing

import sympy


def prepare_permutation(perm: typing.List[int]):
    """Convert a permutation into the format used by apply_permutation.

    Args:
        perm: The permutation. This will be changed by this function
    """
    for i, _ in enumerate(perm):
        while perm[i] < i:
            perm[i] = perm[perm[i]]


def apply_permutation(perm: typing.List[int], data: typing.List[typing.Any]):
    """Apply a permutation to some data.

    Args:
        perm: The prepared permutation
        data: The data to apply the permutation to
    """
    for i, j in enumerate(perm):
        data[i], data[j] = data[j], data[i]


def prepare_matrix(mat: sympy.matrices.dense.MutableDenseMatrix) -> typing.List[int]:
    """Convert a matrix into the format used by apply_matrix.

    Args:
        mat: The matrix

    Returns:
        The permutation to pass into apply_matrix
    """
    assert mat.shape[0] == mat.shape[1]
    dim = mat.shape[0]
    lower, upper, swaps = mat.transpose().LUdecomposition()
    for i in range(dim):
        for j in range(dim):
            if j > i:
                mat[i, j] = lower[j, i]
            else:
                mat[i, j] = upper[j, i]
    perm = list(range(dim))
    for i, j in swaps:
        perm[i], perm[j] = perm[j], perm[i]
    prepare_permutation(perm)
    return perm


def apply_matrix(
    perm: typing.List[int], mat: sympy.matrices.dense.MutableDenseMatrix,
    data: typing.List[typing.Any]
):
    """Apply a matrix to some data.

    Args:
        perm: The permutation returned by prepare_matrix
        mat: The prepared matrix
        data: The data to apply the matrix to
    """
    assert mat.shape[0] == mat.shape[1]
    dim = mat.shape[0]

    apply_permutation(perm, data)

    for i in range(dim):
        for j in range(i+1, dim):
            data[i] += mat[i, j] * data[j]
    for i in range(1, dim + 1):
        data[dim - i] *= mat[dim - i, dim - i]
        for j in range(dim - i):
            data[dim - i] += mat[dim - i, j] * data[j]
