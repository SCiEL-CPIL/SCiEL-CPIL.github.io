"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial.distance import cdist


def _compute_radius_for_ring(ring_radii, ring_counts, ring_phases):
    """
    Compute the maximum common radius given ring radii, counts, and phases.
    
    For each ring i with count n_i at radius R_i:
    - Circumference constraint: 2*pi*R_i >= n_i * 2*r  =>  r <= pi*R_i/n_i
    - Radial gap constraint between ring i and i+1: R_{i+1} - R_i >= 2*r
    - Boundary constraint: 1 - R_{max} >= r
    
    The actual achievable radius is the minimum of:
    1. min_i (pi * R_i / n_i)  [circumference]
    2. min_i (R_{i+1} - R_i) / 2  [radial gaps, for i < k-1]
    3. (1 - R_{k-1})  [boundary]
    4. (R_1 - R_0) / 2  [gap from center to first ring, if R_0 > 0]
    
    But we also need to check actual pairwise distances for non-uniform phases.
    """
    k = len(ring_radii)
    if k == 0:
        return 0.0
    
    # Build points to compute actual minimum distance
    points = []
    for i in range(k):
        r = ring_radii[i]
        n = ring_counts[i]
        if n == 0:
            continue
        phase = ring_phases[i]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points.append(np.column_stack([xs, ys]))
    
    if not points:
        return 0.0
    
    points = np.vstack(points)
    
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)


def _optimize_ring_structure(ring_counts):
    """
    Optimize the ring radii and phases to maximize the common circle radius.
    
    Strategy:
    1. Determine good initial ring radii based on hexagonal packing intuition
    2. Optimize radii and phases using gradient-free methods
    3. The objective is to maximize the minimum distance (circle radius)
    """
    k = len(ring_counts)
    
    # Initial radii: based on area considerations
    # Total area = pi, each circle area = pi*r^2, N circles
    # Average spacing ~ sqrt(pi/N)
    # For ring i with n_i points, radius ~ sqrt((i+1)/k) * (1 - r)
    
    # Better initial guess: assume uniform radial spacing
    # The radial extent should be roughly 1 - r
    # With k rings, spacing ~ (1-r)/k
    
    # Start with evenly spaced radii
    initial_radii = np.linspace(0.05, 0.95, k)
    initial_phases = np.zeros(k)
    
    # Try multiple initializations
    best_radius = 0.0
    best_params = None
    
    for init in range(3):
        if init == 0:
            radii0 = np.linspace(0.05, 0.95, k)
            phases0 = np.zeros(k)
        elif init == 1:
            # Slightly different spacing
            radii0 = np.linspace(0.03, 0.97, k)
            phases0 = np.linspace(0, 0.5, k)
        else:
            # Concentric with hexagonal offset
            radii0 = np.linspace(0.08, 0.92, k)
            phases0 = np.array([0 if i % 2 == 0 else np.pi / ring_counts[i] for i in range(k)])
        
        params0 = np.concatenate([radii0, phases0])
        
        def objective(params):
            radii = np.maximum(params[:k], 0.0)
            phases = params[k:]
            r = _compute_radius_for_ring(radii, ring_counts, phases)
            return -r
        
        bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
        
        result = minimize(
            objective, params0,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        
        r_val = -result.fun
        if r_val > best_radius:
            best_radius = r_val
            best_params = result.x
    
    return best_params, best_radius


def construct():
    """Construct the packing configuration using radial-ring decomposition."""
    
    # Define ring counts: sum to 300
    # Use a pattern that approximates hexagonal packing in polar coordinates
    # Ring i has approximately 6*i points for i >= 1, plus 1 at center
    # 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 = 271, need 29 more
    # Try: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29 = 300
    
    ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    
    # Optimize ring structure
    best_params, best_radius = _optimize_ring_structure(ring_counts)
    
    radii = np.maximum(best_params[:len(ring_counts)], 0.0)
    phases = best_params[len(ring_counts):]
    
    # Build final points
    points = []
    for i, (r, phase, count) in enumerate(zip(radii, phases, ring_counts)):
        if count == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + phase
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points.append(np.column_stack([xs, ys]))
    
    if not points:
        # Fallback: simple grid
        pts = np.zeros((N, 2))
        return pts
    
    points = np.vstack(points)
    
    # Verify we have exactly N points
    if len(points) != N:
        # Truncate or pad as needed
        if len(points) > N:
            points = points[:N]
        else:
            # Pad with center point
            pad = np.zeros((N - len(points), 2))
            points = np.vstack([points, pad])
    
    return points

# EVOLVE-BLOCK-END
