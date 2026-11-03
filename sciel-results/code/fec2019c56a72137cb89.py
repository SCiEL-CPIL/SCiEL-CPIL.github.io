"""50 equal circles in a unit circle - ring-structured optimization with DE."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 50
_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _build_centres(params):
    """Build 50 circle centers from ring parameters.
    
    params layout:
      - 1 center at origin (fixed)
      - Ring 1: 6 circles, radius r1, angle offset a1
      - Ring 2: 12 circles, radius r2, angle offset a2
      - Ring 3: 18 circles, radius r3, angle offset a3
      - Ring 4: 13 circles, radius r4, angle offset a4
    
    params = [r1, a1, r2, a2, r3, a3, r4, a4] (8 parameters)
    """
    r1, a1, r2, a2, r3, a3, r4, a4 = params
    
    centres = np.zeros((N, 2))
    idx = 0
    
    # Center circle
    centres[idx, 0] = 0.0
    centres[idx, 1] = 0.0
    idx += 1
    
    # Ring 1: 6 circles
    for k in range(6):
        angle = a1 + 2.0 * np.pi * k / 6
        centres[idx, 0] = r1 * np.cos(angle)
        centres[idx, 1] = r1 * np.sin(angle)
        idx += 1
    
    # Ring 2: 12 circles
    for k in range(12):
        angle = a2 + 2.0 * np.pi * k / 12
        centres[idx, 0] = r2 * np.cos(angle)
        centres[idx, 1] = r2 * np.sin(angle)
        idx += 1
    
    # Ring 3: 18 circles
    for k in range(18):
        angle = a3 + 2.0 * np.pi * k / 18
        centres[idx, 0] = r3 * np.cos(angle)
        centres[idx, 1] = r3 * np.sin(angle)
        idx += 1
    
    # Ring 4: 13 circles
    for k in range(13):
        angle = a4 + 2.0 * np.pi * k / 13
        centres[idx, 0] = r4 * np.cos(angle)
        centres[idx, 1] = r4 * np.sin(angle)
        idx += 1
    
    return centres


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    dist_origin = np.hypot(centres[:, 0], centres[:, 1])
    r_contain = 1.0 - np.max(dist_origin)
    
    # Min pairwise distance / 2
    min_pair = np.inf
    for i in range(N):
        d = np.hypot(centres[i+1:, 0] - centres[i, 0], centres[i+1:, 1] - centres[i, 1])
        if d.size > 0:
            m = np.min(d)
            if m < min_pair:
                min_pair = m
    r_pair = min_pair / 2.0
    return min(r_contain, r_pair)


def _soft_violation(centres, t, beta=100.0):
    """Soft violation measure: how much the packing violates radius t."""
    dist_origin = np.hypot(centres[:, 0], centres[:, 1])
    # Containment violation: max(0, dist_origin - (1 - t))
    contain_viol = np.maximum(0, dist_origin - (1.0 - t))
    max_contain = np.max(contain_viol) if contain_viol.size > 0 else 0.0
    
    # Pairwise violation: max(0, 2*t - dist_ij)
    max_pair = 0.0
    for i in range(N):
        d = np.hypot(centres[i+1:, 0] - centres[i, 0], centres[i+1:, 1] - centres[i, 1])
        if d.size > 0:
            viol = np.maximum(0, 2.0 * t - d)
            mv = np.max(viol)
            if mv > max_pair:
                max_pair = mv
    
    # Soft minimum of the two violation measures
    # We want the minimum of max_contain and max_pair to be <= 0
    # Use log-sum-exp for soft min
    if max_contain < 1e-15 and max_pair < 1e-15:
        return 0.0
    a = max_contain * beta
    b = max_pair * beta
    if a < 1e-15:
        a = 1e-15
    if b < 1e-15:
        b = 1e-15
    m = min(a, b)
    s = np.log(np.exp(a - m) + np.exp(b - m)) + m
    return s / beta


def _binary_search_radius(centres, t_low=0.1, t_high=0.13, iterations=30, beta=100.0):
    """Binary search for the maximum radius t such that packing is valid."""
    for _ in range(iterations):
        t_mid = (t_low + t_high) / 2
        if _soft_violation(centres, t_mid, beta) < 1e-10:
            t_low = t_mid
        else:
            t_high = t_mid
    return (t_low + t_high) / 2


def _de_objective(params):
    """Objective for differential evolution: minimize negative radius."""
    if time.time() > _DEADLINE - 15.0:
        return 1.0  # Large penalty
    
    centres = _build_centres(params)
    
    # Check for invalid configurations
    if not np.all(np.isfinite(centres)):
        return 1.0
    
    # Ensure circles are inside the container
    dist_origin = np.hypot(centres[:, 0], centres[:, 1])
    if np.max(dist_origin) > 1.0:
        return 1.0
    
    r = _compute_radius(centres)
    return -r


def _local_optimize(centres):
    """Local optimization using Nelder-Mead on individual coordinates."""
    def objective(params):
        pts = params.reshape(-1, 2)
        r = _compute_radius(pts)
        return -r
    
    x0 = centres.flatten()
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11})
        if result.fun < -_compute_radius(centres):
            return result.x.reshape(-1, 2)
    except:
        pass
    return centres


def _hexagonal_init(n=50):
    """Initialize with hexagonal lattice pattern."""
    points = []
    spacing = 0.22
    
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * spacing + 0.5 * spacing * (j % 2)
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 0.9:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) >= n:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:n]
        return points[idx]
    
    return np.random.randn(n, 2) * 0.5


def construct():
    global _DEADLINE
    t_start = time.time()
    
    best_centres = None
    best_radius = 0.0
    
    # Phase 1: Differential evolution on ring parameters
    # params = [r1, a1, r2, a2, r3, a3, r4, a4]
    # Bounds: radii in [0.1, 0.9], angles in [0, 2*pi]
    bounds = [
        (0.1, 0.5),   # r1
        (0.0, 2*np.pi), # a1
        (0.3, 0.7),   # r2
        (0.0, 2*np.pi), # a2
        (0.5, 0.85),  # r3
        (0.0, 2*np.pi), # a3
        (0.7, 0.98),  # r4
        (0.0, 2*np.pi), # a4
    ]
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            _de_objective, bounds,
            maxiter=200, popsize=15,
            tol=1e-8, mutation=(0.5, 1.0), recombination=0.7,
            seed=42, polish=True,
            constraints=[]
        )
        
        if de_result.success or de_result.fun < -0.1:
            best_params = de_result.x
            best_centres = _build_centres(best_params)
            best_radius = _compute_radius(best_centres)
    except:
        pass
    
    # Phase 2: Local optimization on the full 100-dimensional space
    if best_centres is not None and time.time() < _DEADLINE - 20.0:
        polished = _local_optimize(best_centres)
        polished_radius = _compute_radius(polished)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centres = polished
    
    # Phase 3: Try multiple random restarts with ring structure
    if time.time() < _DEADLINE - 30.0:
        for restart in range(5):
            if time.time() > _DEADLINE - 25.0:
                break
            
            # Random ring parameters
            params = np.array([
                np.random.uniform(0.15, 0.45),  # r1
                np.random.uniform(0, 2*np.pi),  # a1
                np.random.uniform(0.35, 0.65),  # r2
                np.random.uniform(0, 2*np.pi),  # a2
                np.random.uniform(0.55, 0.8),   # r3
                np.random.uniform(0, 2*np.pi),  # a3
                np.random.uniform(0.75, 0.95),  # r4
                np.random.uniform(0, 2*np.pi),  # a4
            ])
            
            centres = _build_centres(params)
            r = _compute_radius(centres)
            
            if r > best_radius:
                best_radius = r
                best_centres = centres
            
            # Local polish
            if time.time() < _DEADLINE - 20.0:
                polished = _local_optimize(centres)
                polished_radius = _compute_radius(polished)
                if polished_radius > best_radius:
                    best_radius = polished_radius
                    best_centres = polished
    
    # Phase 4: Also try hexagonal initialization with local optimization
    if best_centres is None or time.time() < _DEADLINE - 15.0:
        hex_centres = _hexagonal_init(N)
        hex_radius = _compute_radius(hex_centres)
        if hex_radius > best_radius:
            best_radius = hex_radius
            best_centres = hex_centres
        
        if time.time() < _DEADLINE - 10.0:
            polished = _local_optimize(hex_centres)
            polished_radius = _compute_radius(polished)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centres = polished
    
    # Final safety: ensure all circles are strictly inside
    if best_centres is None:
        # Emergency fallback
        best_centres = _hexagonal_init(N)
        best_radius = _compute_radius(best_centres)
    
    dist_origin = np.hypot(best_centres[:, 0], best_centres[:, 1])
    max_dist = np.max(dist_origin)
    if max_dist > 0.999:
        scale = 0.999 / max_dist
        best_centres = best_centres * scale
    
    # Final radius check
    final_radius = _compute_radius(best_centres)
    if final_radius <= 0:
        # Emergency fallback
        best_centres = _hexagonal_init(N)
        final_radius = _compute_radius(best_centres)
    
    return best_centres
