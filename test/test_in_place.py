import sympy
import pytest
import random
import in_place


@pytest.mark.parametrize("size", [1, 5, 10, 20])
def test_permutation(size):
    for _ in range(10):
        perm = list(range(size))
        random.shuffle(perm)

        p = [i for i in perm]
        in_place.prepare_permutation(p)

        data = list(range(size))
        in_place.apply_permutation(p, data)

        assert perm == data


@pytest.mark.parametrize("size", [1, 5, 10, 20])
def test_matrix(size):
    for _ in range(10):
        matrix = sympy.Matrix([[random.randrange(4) for j in range(size)] for i in range(size)])
        while matrix.det() == 0:
            matrix = sympy.Matrix([[random.randrange(4) for j in range(size)] for i in range(size)])

        vec = [random.randrange(4) for i in range(size)]

        m = sympy.Matrix([[matrix[i, j] for j in range(size)] for i in range(size)])
        perm = in_place.prepare_matrix(m)

        v = [i for i in vec]
        in_place.apply_matrix(perm, m, v)

        assert list(matrix @ sympy.Matrix(vec)) == v
