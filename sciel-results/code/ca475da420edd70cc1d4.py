"""Strategy: 1D Angular Optimization with Grid Search and Nelder-Mead Polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n = 8

    # Objective: Maximize product of distances.
    # Since points are on a circle of radius R (diameter 2R), the product of distances
    # is proportional to R^(n(n-1)) * product of chord lengths.
    # Chord length between angles a_i, a_j is 2R * sin(|a_i - a_j|/2).
    # So Product = (2R)^(n(n-1)) * prod(sin(|a_i - a_j|/2)).
    # We maximize the log of the product of sines (constant factors don't matter for argmax).
    
    def log_prod_sines(angles):
        # angles: array of shape (n,)
        # Compute pairwise differences
        # We want sin(|a_i - a_j|/2) for i != j
        # Note: sin(x) is positive for x in (0, pi). Since angles are in [0, 2pi),
        # |a_i - a_j| is in [0, 2pi). |a_i - a_j|/2 is in [0, pi).
        # sin is non-negative.
        
        diff = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
        # Reduce to [0, pi] effectively by taking sin of half difference
        # Actually, sin(|a-b|/2) is the same as sin(|a-b|/2) for |a-b| in [0, 2pi]
        # because sin(x) for x in [0, pi] is positive.
        # Wait, if |a-b| > pi, sin(|a-b|/2) is still positive?
        # Let a=0, b=3pi/2. |a-b|=3pi/2. |a-b|/2 = 3pi/4. sin(3pi/4) > 0.
        # Let a=0, b=pi. |a-b|=pi. |a-b|/2 = pi/2. sin(pi/2) = 1.
        # Let a=0, b=2pi. |a-b|=2pi. |a-b|/2 = pi. sin(pi) = 0.
        # So yes, sin(|a_i - a_j|/2) is always non-negative for angles in [0, 2pi].
        
        half_diff = diff / 2.0
        sines = np.sin(half_diff)
        
        # Exclude diagonal (distance 0 -> sin(0)=0 -> log(0)=-inf)
        np.fill_diagonal(sines, 1.0)
        
        # To avoid log(0) if points coincide, we can add a small epsilon or handle it.
        # In optimization, points shouldn't coincide if we start well.
        # But for robustness:
        sines = np.maximum(sines, 1e-15)
        
        return np.sum(np.log(sines))

    def neg_log_prod_sines(angles):
        return -log_prod_sines(angles)

    # Bounds for angles: [0, 2*pi]
    # We fix angle[0] = 0 to break rotational symmetry?
    # No, Nelder-Mead handles periodicity poorly if not careful.
    # Better: Use angles in [0, 2*pi] and let the optimizer work.
    # Or, fix one angle to 0 and optimize the rest relative to it?
    # Fixing angle[0]=0 reduces dimensionality to 7.
    
    def neg_log_prod_sines_reduced(angles_reduced):
        # angles_reduced: shape (7,)
        # angles[0] is fixed at 0
        angles = np.zeros(8)
        angles[1:] = angles_reduced
        return neg_log_prod_sines(angles)

    # Initial guess: Regular octagon
    # Angles: 0, pi/4, pi/2, ...
    angles_init = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles_reduced_init = angles_init[1:]

    # Strategy:
    # 1. Grid search for a good starting point? 
    #    A full grid in 7D is too large.
    #    Instead, try multiple random starts + the regular octagon.
    # 2. Polish with Nelder-Mead.
    
    best_score = -np.inf
    best_angles_reduced = angles_reduced_init

    # List of initial guesses
    # 1. Regular octagon
    inits = [angles_reduced_init]
    
    # 2. Random starts
    np.random.seed(42)
    for _ in range(50):
        # Random angles
        rand_angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        # Shift so min is 0 (breaks symmetry, helps NM)
        rand_angles -= rand_angles[0]
        inits.append(rand_angles[1:])
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    # 3. Perturbed regular octagon
    for _ in range(20):
        pert = np.random.normal(0, 0.1, n)
        pert_angles = angles_init + pert
        pert_angles -= pert_angles[0]
        inits.append(pert_angles[1:])
        
        if time.time() > deadline - 5.0:
            break

    # Optimize each start
    for init in inits:
        if time.time() > deadline - 2.0:
            break
            
        try:
            res = minimize(
                neg_log_prod_sines_reduced,
                init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            score = res.fun
            if score < best_score:
                best_score = score
                best_angles_reduced = res.x
        except Exception:
            continue

    # Construct final points
    best_angles = np.zeros(8)
    best_angles[1:] = best_angles_reduced
    
    # Sort angles to ensure consistent ordering? Not strictly necessary for score,
    # but good for distinctness check.
    # Actually, the order doesn't matter for the set of points.
    
    # Points on unit circle (radius 1)
    pts = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    # Scale to diameter 2.
    # Diameter of points on unit circle is 2 (diameter of circle).
    # So pts are already scaled correctly if they span the circle.
    # Let's verify diameter just in case (e.g. if points cluster).
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    diam = np.max(d)
    
    if diam < 1e-10:
        # Fallback
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
    pts_scaled = pts * (2.0 / diam)
    
    # Verify distinctness
    # Recalculate distances on scaled points
    diff = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-10 or not np.all(np.isfinite(pts_scaled)):
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    if np.max(np.abs(pts_scaled)) >= 1e6:
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    return pts_scaled.tolist()
# EVOLVE-BLOCK-END
