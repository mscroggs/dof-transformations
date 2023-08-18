"""DOF transformations."""

import typing

import symfem
import sympy
from symfem.geometry import PointType


def get_sub_entity_permutations(
    reference: symfem.references.Reference
) -> typing.List[typing.Tuple[str, typing.Tuple[int, int], typing.Callable[[PointType], PointType]]]:
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
    print(perm)

    return {"interval reflection": sympy.Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])}
