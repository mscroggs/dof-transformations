"""In place permutations and matrix mutiplication."""

from typing import List, Any, Tuple

import sympy


def prepare_permutation(perm_in: List[int]) -> List[int]:
    """Convert a permutation into the format used by apply_permutation.

    Args:
        perm_in: The permutation

    Returns:
        A prepared permutation
    """
    perm = [i for i in perm_in]
    for i, _ in enumerate(perm):
        while perm[i] < i:
            perm[i] = perm[perm[i]]
    return perm


def apply_permutation(perm: List[int], data: List[Any]):
    """Apply a permutation to some data.

    Args:
        perm: The prepared permutation
        data: The data to apply the permutation to
    """
    for i, j in enumerate(perm):
        data[i], data[j] = data[j], data[i]


def prepare_matrix(
    mat_in: sympy.matrices.dense.MutableDenseMatrix,
) -> Tuple[sympy.matrices.dense.MutableDenseMatrix, List[int]]:
    """Convert a matrix into the format used by apply_matrix.

    Args:
        mat_in: The matrix

    Returns:
        The permutation and matrix to pass into apply_matrix
    """
    assert mat_in.shape[0] == mat_in.shape[1]
    dim = mat_in.shape[0]
    lower, upper, swaps = mat_in.transpose().LUdecomposition()
    mat = sympy.Matrix(
        [
            [lower[j, i] if j > i else upper[j, i] for j in range(dim)]
            for i in range(dim)
        ]
    )
    perm = list(range(dim))
    for i, j in swaps:
        perm[i], perm[j] = perm[j], perm[i]
    return mat, prepare_permutation(perm)


def apply_matrix(
    mat: sympy.matrices.dense.MutableDenseMatrix,
    perm: List[int],
    data: List[Any],
):
    """Apply a matrix to some data.

    Args:
        mat: The prepared matrix
        perm: The permutation returned by prepare_matrix
        data: The data to apply the matrix to
    """
    assert mat.shape[0] == mat.shape[1]
    dim = mat.shape[0]

    apply_permutation(perm, data)

    for i in range(dim):
        for j in range(i + 1, dim):
            data[i] += mat[i, j] * data[j]
    for i in range(dim - 1, -1, -1):
        data[i] *= mat[i, i]
        for j in range(i):
            data[i] += mat[i, j] * data[j]
