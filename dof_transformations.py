import typing
import sympy
import symfem


def compute_base_transformations(
    element: symfem.finite_element.CiarletElement
) -> typing.Dict[str, sympy.matrices.dense.MutableDenseMatrix]:
    raise NotImplementedError()
