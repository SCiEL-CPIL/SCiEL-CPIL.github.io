"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _score_from_points(points):
    """Compute (max pairwise dist / min pairwise dist)^2. Lower is better."""
    n = len(points)
    if n < 2:
        return np.inf
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Exclude self-distances
    iu = np.triu_indices(n, k=1)
    pd = dists[iu]
    if len(pd) == 0:
        return np.inf
    dmin = pd.min()
    dmax = pd.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2


def _deformed_grid(params):
    """
    Build 16 points from a 4x4 base grid with sinusoidal deformation.
    params:
      p0: amplitude of x-warp (horizontal sinusoidal shift)
      p1: amplitude of y-warp (vertical sinusoidal shift)
      p2: frequency of x-warp
      p3: frequency of y-warp
      p4: scale factor (uniform scaling)
      p5: rotation angle
    """
    p0, p1, p2, p3, p4, p5 = params
    # Base 4x4 grid centered at origin
    xs = np.array([0, 1, 2, 3], dtype=float)
    ys = np.array([0, 1, 2, 3], dtype=float)
    GX, GY = np.meshgrid(xs, ys, indexing='ij')  # 4x4 arrays
    # Flatten to 16 points
    x = GX.ravel()
    y = GY.ravel()
    # Center the grid
    x = x - 1.5
    y = y - 1.5
    # Apply sinusoidal deformation
    # x-warp: shift x based on y position
    x = x + p0 * np.sin(p2 * y)
    # y-warp: shift y based on x position
    y = y + p1 * np.sin(p3 * x)
    # Scale
    x = x * p4
    y = y * p4
    # Rotate
    c, s = np.cos(p5), np.sin(p5)
    xr = c * x - s * y
    yr = s * x + c * y
    return np.column_stack([xr, yr])


def _objective(params):
    """Objective function for DE: minimize the ratio squared."""
    points = _deformed_grid(params)
    # Check distinctness
    d = _score_from_points(points)
    if not np.isfinite(d):
        return 1e10
    return d


def construct():
    # Define bounds for DE
    bounds = [
        (-1.0, 1.0),   # p0: x-warp amplitude
        (-1.0, 1.0),   # p1: y-warp amplitude
        (0.5, 4.0),    # p2: x-warp frequency
        (0.5, 4.0),    # p3: y-warp frequency
        (0.5, 2.0),    # p4: scale
        (-np.pi, np.pi),  # p5: rotation
    ]
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=200,
        popsize=20,
        tol=1e-8,
        seed=42,
        polish=True,
    )
    
    best_params = result.x
    points = _deformed_grid(best_params)
    
    # Verify distinctness
    score = _score_from_points(points)
    if not np.isfinite(score):
        # Fallback: regular grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    
    return points
# EVOLVE-BLOCK-END
