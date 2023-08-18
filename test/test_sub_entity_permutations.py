import pytest
import symfem

from dof_transformations import get_sub_entity_permutations


@pytest.mark.parametrize("cellname", [
    "interval", "triangle", "quadrilateral", "tetrahedron",
    "hexahedron", "prism", "pyramid",
])
def test_sub_entity_permutations(cellname):
    reference = symfem.create_reference(cellname)
    perms = get_sub_entity_permutations(reference)

    for perm in perms:
        print(perm[0])

        entity_vertices = reference.sub_entity(*perm[1]).vertices
        mapped_points = [perm[2](p) for p in entity_vertices]
        for p in mapped_points:
            assert mapped_points.count(p) == 1
            assert p in entity_vertices
