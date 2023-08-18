"""DOF transformations."""

import typing
import sympy
import symfem


def compute_base_transformations(
    element: symfem.finite_element.CiarletElement
) -> typing.Dict[str, sympy.matrices.dense.MutableDenseMatrix]:
    """Compute the base transformations for an element.

    Args:
        element: The element

    Returns:
        A dictionary of base transformation matrices
    """
    raise NotImplementedError()
