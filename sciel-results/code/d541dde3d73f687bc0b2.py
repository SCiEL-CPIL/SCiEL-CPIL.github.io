<<<<<<< SEARCH
def _min_side_all(centers, angles):
    """(N,) minimal outer side length for each hexagon (origin-centered outer)."""
    verts = centers + (VERTS_UNIT @ np.stack(
        [np.array([math.cos(math.radians(a)), math.sin(math.radians(a))]) for a in angles]).T).T
    return np.max(verts @ NORMALS, axis=1)
=======
def _min_side_all(centers, angles):
    """(N,) minimal outer side length for each hexagon (origin-centered outer)."""
    sides = np.zeros(N)
    for i in range(N):
        sides[i] = _min_side(centers[i], angles[i])
    return sides
>>>>>>> REPLACE

