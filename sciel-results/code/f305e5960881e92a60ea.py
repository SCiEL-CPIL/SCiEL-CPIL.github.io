import itertools

def _make_points(nrows, ncols):
    """Generate a grid of points."""
    result = []
    for row in range(nrows):
        for col in range(ncols):
            result.append([col, row])
    return result

def construct():
    return _make_points(3, 4)
