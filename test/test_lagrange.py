import pytest
import symfem
import sympy

from dof_transformations import compute_base_transformations


@pytest.mark.parametrize("cell_type", ["triangle", "quadrilateral"])
def test_lagrange_2d(cell_type):
    """Test that the transformations are correct for degree 4 Lagrange on 2D cells."""
    element = symfem.create_element(cell_type, "Lagrange", 4)
    t = compute_base_transformations(element)
    assert len(t) == 1
    assert t["interval reflection"] == sympy.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])


def test_lagrange_tetrahedron():
    """Test that the transformations are correct for degree 4 Lagrange on a tetrahedron."""
    element = symfem.create_element("tetrahedron", "Lagrange", 4)
    t = compute_base_transformations(element)
    assert len(t) == 3
    assert t["interval reflection"] == sympy.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert t["triangle rotation"] == sympy.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert t["triangle reflection"] == sympy.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_lagrange_hexahedron():
    """Test that the transformations are correct for degree 4 Lagrange on a hexahedron."""
    element = symfem.create_element("hexahedron", "Lagrange", 3)
    t = compute_base_transformations(element)
    assert len(t) == 3
    assert t["interval reflection"] == sympy.Matrix([[0, 1], [1, 0]])
    assert t["quadrilateral rotation"] == sympy.Matrix(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
    )
    assert t["quadrilateral reflection"] == sympy.Matrix(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
