"""DOF transformations."""

import typing

import symfem
import sympy
from symfem.geometry import PointType


def get_sub_entity_permutations(
    reference: symfem.references.Reference
) -> typing.List[typing.Tuple[
    str, typing.Tuple[int, int], typing.Callable[[PointType], PointType]
]]:
    """Get maps that permute each sub-entity type.

    Args:
        reference: The reference cell

    Returns:
        A list of triplets containing the transformation name, (tdim, entity number), and function
        that permutes the entity
    """
    if reference != reference.default_reference():
        raise ValueError("Computing transformations is not supported for non-default references.")

    if reference.name == "interval":
        return []
    if reference.name == "triangle":
        return [("interval reflection", (1, 2), lambda x: (1 - x[0], x[1]))]
    if reference.name == "tetrahedron":
        return [
            ("interval reflection", (1, 5), lambda x: (1 - x[0], x[1], x[2])),
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
    function: typing.Callable[[PointType], PointType]
) -> typing.Tuple[typing.Tuple[sympy.core.expr.Expr, ...], typing.Tuple[sympy.core.expr.Expr, ...]]:
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
    constant = tuple(m.subs(x[0], 0).subs(x[1], 0).subs(x[2], 0) for v, m in zip(x, fwd_map))
    bwd_map = tuple(j_inv @ sympy.Matrix([v - c for v, c in zip(x, constant)])[:, 0])

    return fwd_map, bwd_map


def compute_base_transformations(
    element: symfem.finite_element.CiarletElement
) -> typing.Dict[str, sympy.matrices.dense.MutableDenseMatrix]:
    """Compute the base transformations for an element.

    Args:
        element: The element

    Returns:
        A dictionary of base transformation matrices
    """
    reference = element.reference
    perm = get_sub_entity_permutations(reference)

    transformations = {}

    mapping = element.dofs[0].mapping
    for d in element.dofs:
        if d.mapping != mapping:
            raise ValueError("DOF transformations not implemeneted for elements "
                             "with mixed mapping types.")

    basis = element.get_basis_functions()

    for name, entity, function in perm:
        fwd_map, bwd_map = get_maps(function)

        dofs = element.entity_dofs(*entity)
        entity_basis = [basis[d] for d in dofs]
        pulled_basis = [getattr(symfem.mappings, f"{mapping}")(f, fwd_map, bwd_map, reference.tdim)
                        for f in entity_basis]

        transformations[name] = sympy.Matrix([
            [element.dofs[d].eval(f) for d in dofs] for f in pulled_basis])

    return transformations
