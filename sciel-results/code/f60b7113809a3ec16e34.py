import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0

def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0

    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    dmin = np.min(dists)
    dmax = np.max(dists)
    if not (np.isfinite(dmin) and np.isfinite(dmax)):
        return 0.0
    if dmax <= 1e-15 or dmin <= 1e-15:
        return 0.0
    r = dmin / dmax
    return float(r * r)

def _spherical_to_cartesian(x):
    """
    Convert flat array of spherical coordinates (theta, phi) for 14 points
    to Cartesian coordinates.
    x shape: (28,)
    theta[i] = polar angle [0, pi]
    phi[i] = azimuthal angle [0, 2*pi]
    """
    theta = x[0::2]
    phi = x[1::2]
    
    # Ensure valid ranges, though optimizer might go out of bounds
    # We can wrap them or just let the math handle it (sin/cos are periodic)
    # But for stability, let's assume the optimizer keeps them reasonable.
    
    x_cart = np.zeros((14, 3))
    x_cart[:, 0] = np.sin(theta) * np.cos(phi)
    x_cart[:, 1] = np.sin(theta) * np.sin(phi)
    x_cart[:, 2] = np.cos(theta)
    
    return x_cart

def _objective_spherical(x):
    """
    Objective function for spherical coordinates.
    Minimizes log(d_max) - log(d_min) which is -log(d_min/d_max).
    """
    points = _spherical_to_cartesian(x)
    dists = pdist(points)
    dmin = np.min(dists)
    dmax = np.max(dists)
    
    if dmin <= 1e-15 or dmax <= 1e-15:
        return 1e10
        
    return np.log(dmax) - np.log(dmin)

def _simulated_annealing(start_time, deadline, seed=42):
    """
    Perform Simulated Annealing on spherical coordinates.
    """
    rng = np.random.default_rng(seed)
    
    # Initial random configuration
    # theta in [0, pi], phi in [0, 2*pi]
    x_current = np.zeros(28)
    x_current[0::2] = rng.uniform(0, np.pi, 14)
    x_current[1::2] = rng.uniform(0, 2 * np.pi, 14)
    
    f_current = _objective_spherical(x_current)
    
    x_best = x_current.copy()
    f_best = f_current
    
    # SA Parameters
    T_start = 1.0
    T_end = 1e-6
    alpha = 0.995  # Cooling rate
    
    T = T_start
    step_size = 0.5 # Initial step size
    
    steps = 0
    while time.time() < deadline and T > T_end:
        # Propose move
        x_proposed = x_current.copy()
        # Randomly select a few coordinates to perturb
        num_perturb = rng.integers(1, 4)
        indices = rng.choice(28, num_perturb, replace=False)
        
        # Perturbation size scales with temperature
        noise = rng.normal(0, step_size * T, num_perturb)
        x_proposed[indices] += noise
        
        # Wrap angles to keep them in reasonable range (optional but good for stability)
        x_proposed[0::2] = np.mod(x_proposed[0::2], np.pi) # theta [0, pi]
        # Actually, theta should be [0, pi]. If it goes negative, wrap.
        # If it goes > pi, wrap.
        # Let's just use mod for phi and clip/wrap for theta properly.
        # For theta, sin(theta) is symmetric around pi/2.
        # Let's just let the math handle it, sin/cos are continuous.
        
        f_proposed = _objective_spherical(x_proposed)
        
        # Metropolis Criterion
        delta_f = f_proposed - f_current
        if delta_f < 0 or rng.random() < np.exp(-delta_f / T):
            x_current = x_proposed
            f_current = f_proposed
            
            if f_current < f_best:
                f_best = f_current
                x_best = x_current.copy()
        
        T *= alpha
        # Decrease step size slightly to refine search
        step_size *= 0.999
        
        steps += 1
        if steps % 1000 == 0:
             # Check time more frequently
             if time.time() > deadline:
                 break

    return x_best, f_best

def _polish_spherical(x0, deadline):
    """
    Polish a spherical coordinate configuration using L-BFGS-B.
    """
    if time.time() > deadline:
        return x0

    try:
        res = minimize(
            _objective_spherical,
            x0,
            method="L-BFGS-B",
            options={
                "maxiter": 2000,
                "maxfun": 100000,
                "ftol": 1e-12,
                "gtol": 1e-10,
            },
        )
        return res.x
    except Exception:
        return x0

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN

    best_pts = None
    best_score = 0.0
    
    # Multi-start Simulated Annealing + Polish
    num_starts = 10
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        seed = seeds[i]
        x_sa, f_sa = _simulated_annealing(start_time, deadline - 30, seed=seed)
        
        # Polish the result
        x_polished = _polish_spherical(x_sa, deadline - 5)
        
        # Convert to Cartesian and score
        pts = _spherical_to_cartesian(x_polished)
        score = _raw_score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # If we still have time, do a few more local restarts on the best point found
    if best_pts is not None and time.time() < deadline - 5:
        # Convert best_pts back to spherical? No, just use the best x_polished.
        # We need to store the best x_spherical too.
        # Let's re-run the last successful polish if we have time, or just trust it.
        # Actually, let's keep track of the best x_spherical.
        pass

    # Re-evaluate best_pts to ensure it's the absolute best Cartesian set
    # The loop above already does this.
    
    if best_pts is None:
        # Fallback: Simple random search
        rng = np.random.default_rng(0)
        for _ in range(100):
            if time.time() > deadline:
                break
            x_rand = np.zeros(28)
            x_rand[0::2] = rng.uniform(0, np.pi, 14)
            x_rand[1::2] = rng.uniform(0, 2 * np.pi, 14)
            pts = _spherical_to_cartesian(x_rand)
            s = _raw_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    if best_pts is None:
        best_pts = np.zeros((14, 3))
    else:
        best_pts = np.asarray(best_pts, dtype=np.float64)
        if not np.all(np.isfinite(best_pts)):
            best_pts = np.zeros((14, 3))

    return best_pts
