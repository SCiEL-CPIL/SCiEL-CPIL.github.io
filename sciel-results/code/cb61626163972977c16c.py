"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
Strategy: Combines spectral (Fourier) parameterization for convexity enforcement
with a smooth log-sum-exp surrogate and multi-start L-BFGS-B optimization.
The spectral representation ensures the points remain in convex position
(distorted regular polygon), reducing the search space and avoiding degenerate
non-convex configurations. A tau-annealing schedule allows the optimizer to
navigate the landscape from coarse to fine resolution.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute triple indices for efficiency
_triple_indices = np.array(list(combinations(range(N), 3)))
_TRI_I = _triple_indices[:, 0]
_TRI_J = _triple_indices[:, 1]
_TRI_K = _triple_indices[:, 2]

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min(values, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_v = np.min(values)
    shifted = -tau * (values - min_v)
    # Clip to avoid overflow (shifted should be <= 0)
    shifted = np.clip(shifted, -100, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_v - (1.0 / tau) * log_sum

def _decode_spectral(params):
    """
    Decode spectral parameters into 2D points.
    params: array of shape (2*M, 2) where M is the number of harmonics.
    We use M=3 harmonics for 13 points.
    The points are generated as:
    x_k = sum_{m=0}^{M-1} [a_m cos(2*pi*m*k/N + phi_m) + b_m sin(2*pi*m*k/N + psi_m)]
    This ensures smoothness and convexity for small coefficients.
    To ensure convexity, we can also add a base regular polygon component.
    
    Actually, a simpler and more robust spectral approach for convex polygons:
    Represent the radius as a function of angle: r(theta) = R0 + sum c_m cos(m*theta + phase_m).
    If the coefficients are small enough, the curve remains convex and star-shaped.
    Then sample N points at equal angles.
    
    Let's use: r_k = 1.0 + sum_{m=1}^{M-1} params[2*m-1] * cos(2*pi*m*k/N + params[2*m])
    We fix the mean radius to 1.0 to handle scaling (though scaling is irrelevant for the ratio).
    We need to ensure r_k > 0. With small params, this is guaranteed.
    
    Number of params: 2*(M-1). Let's use M=4 harmonics (m=1,2,3) -> 6 params.
    Wait, N=13. Frequencies 1, 2, 3 are safe.
    """
    # params shape: (2*M_harm, )
    M_harm = len(params) // 2
    k = np.arange(N)
    angles = 2 * np.pi * k / N
    
    # Base regular polygon
    radii = np.ones(N)
    
    for m in range(1, M_harm + 1):
        amp = params[2*(m-1)]
        phase = params[2*(m-1) + 1]
        radii += amp * np.cos(m * angles + phase)
        
    # Ensure positivity
    # If radii become negative, the point crosses the origin, breaking convexity/ordering.
    # We clip radii to a small positive value, but this might introduce discontinuities.
    # Better: constrain the amplitude during optimization or use a softplus-like transform.
    # For now, let's just compute points. The optimizer will likely keep amplitudes small.
    # However, to be safe against non-convexity, we can check the hull.
    
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Center the points to remove translation invariance?
    # The ratio is invariant to translation and scaling.
    # Centering helps numerical stability.
    pts -= np.mean(pts, axis=0)
    
    return pts

def _objective(params, tau):
    """
    Objective function: minimize -smooth_min(triangle_areas / hull_area).
    params: spectral coefficients.
    """
    pts = _decode_spectral(params)
    
    # Check convexity / validity
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    # Check if points are in convex position (hull contains all points)
    try:
        hull = ConvexHull(pts)
        # If any point is not on the hull or inside, it's not strictly convex polygon vertices
        # But the problem says "Place 13 points in a planar convex region".
        # The score is min triangle area / hull area.
        # It does NOT require the 13 points to BE the convex hull vertices.
        # However, to maximize the min triangle area, points should be well-spread.
        # Usually, optimal configurations have points on the hull.
        # If a point is inside, it forms smaller triangles with nearby hull points.
        # So we assume the spectral representation generates a convex polygon (points on hull).
    except Exception:
        return 1e9

    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _generate_initial_params(rng, n_harmonics=3):
    """
    Generate random initial spectral parameters.
    """
    # Small amplitudes to start near regular polygon
    amps = rng.uniform(-0.1, 0.1, n_harmonics)
    phases = rng.uniform(0, 2 * np.pi, n_harmonics)
    
    params = np.zeros(2 * n_harmonics)
    params[0::2] = amps
    params[1::2] = phases
    
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s if available)
    
    best_score = -np.inf
    best_params = None
    best_pts = None
    
    # Number of harmonics to use
    n_harmonics = 3
    
    # Multi-start with L-BFGS-B
    num_starts = 20
    tau_schedule = [10, 20, 50, 100, 200]
    
    # Pre-generate starts
    rng = np.random.default_rng(42)
    starts = []
    for i in range(num_starts):
        p = _generate_initial_params(rng, n_harmonics)
        starts.append(p)
    
    # Annealing loop
    for tau in tau_schedule:
        if time.time() > deadline - 5.0:
            break
            
        for i in range(len(starts)):
            if time.time() > deadline - 3.0:
                break
                
            try:
                # Bounds: amplitudes should be small to maintain convexity
                # r = 1 + sum a_m cos(...). If |a_m| < 0.1, r > 0.7.
                # Let's bound amplitudes to [-0.3, 0.3] to be safe.
                bounds = [(-0.3, 0.3), (0, 2*np.pi)] * n_harmonics
                
                res = minimize(
                    _objective, 
                    starts[i], 
                    args=(tau,),
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                # Evaluate true score of the result
                new_params = res.x
                pts = _decode_spectral(new_params)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = new_params.copy()
                    best_pts = pts.copy()
                    
            except Exception:
                pass
    
    # If no valid solution found, fallback
    if best_pts is None:
        # Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _compute_min_ratio(best_pts)

    return best_pts

# EVOLVE-BLOCK-END
