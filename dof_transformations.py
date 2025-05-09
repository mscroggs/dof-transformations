"""DOF transformations."""

from typing import Callable, Dict, List, Tuple

import symfem
import sympy
from symfem.geometry import PointType


def get_sub_entity_transformations(
    reference: symfem.references.Reference,
) -> List[Tuple[str, Tuple[int, int], Callable[[PointType], PointType]]]:
    """Get maps that transform each sub-entity type.

    Args:
        reference: The reference cell

    Returns:
        A list of triplets containing the transformation name, (tdim, entity number), and function
        that transforms the entity
    """
    if reference != reference.default_reference():
        raise ValueError(
            "Computing transformations is not supported for non-default references."
        )

    if reference.name == "interval":
        return []
    if reference.name == "triangle":
        return [("interval reflection", (1, 0), lambda x: (x[1], x[0]))]
    if reference.name == "tetrahedron":
        return [
            ("interval reflection", (1, 0), lambda x: (x[0], x[2], x[1])),
            ("triangle rotation", (2, 3), lambda x: (x[1], 1 - x[0] - x[1], x[2])),
            ("triangle reflection", (2, 3), lambda x: (x[1], x[0], x[2])),
        ]
    if reference.name == "quadrilateral":
        return [("interval reflection", (1, 0), lambda x: (1 - x[0], x[1]))]
    if reference.name == "hexahedron":
        return [
            ("interval reflection", (1, 0), lambda x: (1 - x[0], x[1], x[2])),
            ("quadrilateral rotation", (2, 0), lambda x: (x[1], 1 - x[0], x[2])),
            ("quadrilateral reflection", (2, 0), lambda x: (x[1], x[0], x[2])),
        ]
    if reference.name == "prism":
        return [
            ("interval reflection", (1, 0), lambda x: (1 - x[0] - x[1], x[1], x[2])),
            ("quadrilateral rotation", (2, 1), lambda x: (x[2], x[1], 1 - x[0])),
            ("quadrilateral reflection", (2, 1), lambda x: (x[2], x[1], x[0])),
            ("triangle rotation", (2, 0), lambda x: (x[1], 1 - x[0] - x[1], x[2])),
            ("triangle reflection", (2, 0), lambda x: (x[1], x[0], x[2])),
        ]
    if reference.name == "pyramid":
        return [
            ("interval reflection", (1, 0), lambda x: (1 - x[0], x[1], x[2])),
            ("quadrilateral rotation", (2, 0), lambda x: (x[1], 1 - x[0], x[2])),
            ("quadrilateral reflection", (2, 0), lambda x: (x[1], x[0], x[2])),
            ("triangle rotation", (2, 1), lambda x: (x[2], x[1], 1 - x[0] - x[2])),
            ("triangle reflection", (2, 1), lambda x: (x[2], x[1], x[0])),
        ]

    raise ValueError(f"Unsupported cell: {reference.name}")


def get_maps(
    function: Callable[[PointType], PointType],
) -> Tuple[Tuple[sympy.Expr, ...], Tuple[sympy.Expr, ...]]:
    """Get the forward and backward maps from a function.

    Args:
        function: The function

    Returns:
        The forward and backward maps
    """
    x = symfem.symbols.x
    fwd_map = function(x)
    tdim = len(fwd_map)

    j_inv = sympy.Matrix([[r.diff(v) for v in x[:tdim]] for r in fwd_map]).inv()
    constant = tuple(m.subs(x[0], 0).subs(x[1], 0).subs(x[2], 0) for m in fwd_map)
    bwd_map = tuple(j_inv @ sympy.Matrix([v - c for v, c in zip(x, constant)])[:, 0])

    return fwd_map, bwd_map


def compute_base_transformations(
    element: symfem.finite_element.CiarletElement,
) -> Dict[str, sympy.Matrix]:
    """Compute the base transformations for an element.

    Args:
        element: The element

    Returns:
        A dictionary of base transformation matrices
    """
    reference = element.reference
    maps = get_sub_entity_transformations(reference)

    transformations = {}

    mapping = element.dofs[0].mapping
    for d in element.dofs:
        if d.mapping != mapping:
            raise ValueError(
                "DOF transformations not implemeneted for elements "
                "with mixed mapping types."
            )
    assert mapping is not None
    push_forward = symfem.mappings.get_mapping(mapping)

    basis = element.get_basis_functions()

    for name, entity, function in maps:
        fwd_map, bwd_map = get_maps(function)

        matrix = []
        dofs = element.entity_dofs(*entity)
        for i in dofs:
            pushed_function = push_forward(basis[i], fwd_map, bwd_map)  # type: ignore
            matrix.append([element.dofs[j].eval(pushed_function) for j in dofs])
        transformations[name] = sympy.Matrix(matrix)

    return transformations
