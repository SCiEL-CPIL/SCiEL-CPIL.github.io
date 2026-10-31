_COMBINATION_INDICES = None

def _get_combination_indices():
    global _COMBINATION_INDICES
    if _COMBINATION_INDICES is None:
        _COMBINATION_INDICES = np.array(list(combinations(range(N), 3)))
    return _COMBINATION_INDICES

def _compute_areas(points):
    """Compute areas of all triangles formed by the points."""
    indices = _get_combination_indices()
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    return areas
