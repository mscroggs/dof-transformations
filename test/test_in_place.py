import random

import pytest
import sympy

import in_place


@pytest.mark.parametrize("size", [1, 5, 10, 20])
def test_permutation(size):
    for _ in range(10):
        perm = list(range(size))
        random.shuffle(perm)

        p = in_place.prepare_permutation(perm)

        data = list(range(size))
        in_place.apply_permutation(p, data)

        assert perm == data


@pytest.mark.parametrize("size", [1, 5, 10, 20])
def test_matrix(size):
    for _ in range(10):
        matrix = sympy.Matrix(
            [[random.randrange(4) for j in range(size)] for i in range(size)]
        )
        while matrix.det() == 0:
            matrix = sympy.Matrix(
                [[random.randrange(4) for j in range(size)] for i in range(size)]
            )

        vec = [random.randrange(4) for i in range(size)]

        m, perm = in_place.prepare_matrix(matrix)

        v = [i for i in vec]
        in_place.apply_matrix(m, perm, v)

        assert list(matrix @ sympy.Matrix(vec)) == v
