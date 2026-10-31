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
    Build 16 points from a 4x4 base grid with multi-mode deformation.
    
    Parameters (12 total):
      p0: radial deformation amplitude (quadratic in r)
      p1: radial deformation amplitude (quartic in r)
      p2: x-sinusoidal warp amplitude (mode 1)
      p3: y-sinusoidal warp amplitude (mode 1)
      p4: x-sinusoidal warp amplitude (mode 2)
      p5: y-sinusoidal warp amplitude (mode 2)
      p6: x-warp frequency (mode 1)
      p7: y-warp frequency (mode 1)
      p8: x-warp frequency (mode 2)
      p9: y-warp frequency (mode 2)
      p10: shear (xy coupling)
      p11: rotation angle
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = params
    
    # Base 4x4 grid centered at origin
    xs = np.array([0, 1, 2, 3], dtype=float)
    ys = np.array([0, 1, 2, 3], dtype=float)
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    x = GX.ravel()
    y = GY.ravel()
    
    # Center the grid
    x = x - 1.5
    y = y - 1.5
    
    # Radial distance from center
    r = np.sqrt(x**2 + y**2)
    r_norm = r / (1.5 * np.sqrt(2))  # normalize to [0, 1]
    
    # Radial deformation: scale points based on distance from center
    # This allows "breathing" the grid - push corners in or out
    radial_scale = 1.0 + p0 * r_norm**2 + p1 * r_norm**4
    x = x * radial_scale
    y = y * radial_scale
    
    # Apply multi-mode sinusoidal deformation
    # Mode 1: low frequency
    x = x + p2 * np.sin(p6 * y)
    y = y + p3 * np.sin(p7 * x)
    
    # Mode 2: higher frequency
    x = x + p4 * np.sin(p8 * y)
    y = y + p5 * np.sin(p9 * x)
    
    # Shear: mix x and y
    x_new = x + p10 * y
    y_new = y - p10 * x
    x, y = x_new, y_new
    
    # Rotate
    c, s = np.cos(p11), np.sin(p11)
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
    # Define bounds for DE (12 parameters)
    bounds = [
        (-0.5, 0.5),    # p0: radial quadratic amplitude
        (-0.5, 0.5),    # p1: radial quartic amplitude
        (-0.3, 0.3),    # p2: x-warp amplitude mode 1
        (-0.3, 0.3),    # p3: y-warp amplitude mode 1
        (-0.2, 0.2),    # p4: x-warp amplitude mode 2
        (-0.2, 0.2),    # p5: y-warp amplitude mode 2
        (0.5, 3.0),     # p6: x-warp frequency mode 1
        (0.5, 3.0),     # p7: y-warp frequency mode 1
        (1.0, 5.0),     # p8: x-warp frequency mode 2
        (1.0, 5.0),     # p9: y-warp frequency mode 2
        (-0.3, 0.3),    # p10: shear
        (-np.pi, np.pi),# p11: rotation
    ]
    
    # Run differential evolution with more iterations for better convergence
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=300,
        popsize=25,
        tol=1e-10,
        seed=42,
        polish=True,
        mutation=(0.3, 1.0),
        recombination=0.7,
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
