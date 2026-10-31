# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


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


def _sinusoidal_grid(params):
    """
    Build 16 points from a 4x4 base grid with sinusoidal deformation.
    params:
      p0: global scale
      p1: spatial frequency k
      p2: amplitude A for x-displacement (mode 1)
      p3: amplitude B for y-displacement (mode 2)
      p4: phase shift phi1
      p5: phase shift phi2
      p6: global rotation angle
    """
    p0, k, A, B, phi1, phi2, rot = params
    
    # Base 4x4 grid centered at origin
    # Coordinates: -1.5, -0.5, 0.5, 1.5
    xs = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)
    ys = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    x = GX.ravel()
    y = GY.ravel()
    
    # Sinusoidal deformation
    # Mode 1: sin(kx + phi1) * cos(ky + phi2)
    # Mode 2: cos(kx + phi1) * sin(ky + phi2)
    
    m1 = np.sin(k * x + phi1) * np.cos(k * y + phi2)
    m2 = np.cos(k * x + phi1) * np.sin(k * y + phi2)
    
    # Displacements
    dx = A * m1
    dy = B * m2
    
    x_new = x + dx
    y_new = y + dy
    
    # Apply global scale
    x_new = x_new * p0
    y_new = y_new * p0
    
    # Apply rotation
    c, s = np.cos(rot), np.sin(rot)
    xr = c * x_new - s * y_new
    yr = s * x_new + c * y_new
    
    return np.column_stack([xr, yr])


def _objective(params):
    """Objective function for DE: minimize the ratio squared."""
    points = _sinusoidal_grid(params)
    # Check distinctness
    d = _score_from_points(points)
    if not np.isfinite(d):
        return 1e10
    return d


def construct():
    # Define bounds for DE
    # p0: scale, p1: freq, p2: ampA, p3: ampB, p4: phi1, p5: phi2, p6: rot
    bounds = [
        (0.5, 2.0),       # p0: scale
        (0.5, 3.0),       # p1: frequency k (low freq to preserve structure)
        (-0.5, 0.5),      # p2: amplitude A
        (-0.5, 0.5),      # p3: amplitude B
        (0.0, 2*np.pi),   # p4: phase phi1
        (0.0, 2*np.pi),   # p5: phase phi2
        (-np.pi/4, np.pi/4), # p6: rotation
    ]
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=300,
        popsize=30,
        tol=1e-9,
        seed=42,
        polish=False,
        init='sobol',
        mutation=(0.3, 1.0),
        recombination=0.7,
    )
    
    best_params = result.x
    
    # Local refinement using L-BFGS-B
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    
    try:
        local_res = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if local_res.success and local_res.fun < _objective(best_params):
            best_params = local_res.x
    except Exception:
        pass
    
    points = _sinusoidal_grid(best_params)
    
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
