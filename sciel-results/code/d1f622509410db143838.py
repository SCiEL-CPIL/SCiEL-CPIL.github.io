import random
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# Precompute indices for all triples
TRIPLES = np.array([(i, j, k) for i in range(N - 2) for j in range(i + 1, N - 1) for k in range(j + 1, N)])
NUM_TRIPLES = len(TRIPLES)

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def soft_min_objective(theta, tau):
    """
    Objective function for L-BFGS-B.
    theta: flattened array of shape (2*N,) representing x,y coordinates.
    tau: temperature parameter for log-sum-exp soft-min.
    
    We want to maximize min_area / hull_area.
    So we minimize - (soft_min_area / hull_area).
    
    soft_min_area = -tau * log(sum(exp(-areas / tau)))
    
    As tau -> 0, soft_min_area -> min(areas).
    As tau -> inf, soft_min_area -> mean(areas).
    
    Using a soft-min makes the objective differentiable everywhere,
    which is crucial for gradient-based methods like L-BFGS-B.
    """
    points = theta.reshape(N, 2)
    
    # Calculate hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 1e6 # Large penalty
    except Exception:
        return 1e6

    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Avoid log(0) or division by zero issues
    # If any area is very small, the soft min will be small.
    # We add a small epsilon to areas to ensure numerical stability in exp
    areas_safe = areas + 1e-12
    
    # Log-sum-exp trick for numerical stability
    # We want: -tau * log(sum(exp(-areas / tau)))
    # Let m = min(areas_safe)
    # sum(exp(-areas / tau)) = exp(-m/tau) * sum(exp(-(areas-m)/tau))
    # log(sum(...)) = -m/tau + log(sum(exp(-(areas-m)/tau)))
    # soft_min = -tau * (-m/tau + log(...)) = m - tau * log(sum(exp(-(areas-m)/tau)))
    
    m = np.min(areas_safe)
    exp_terms = np.exp(-(areas_safe - m) / tau)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - tau * log_sum
    
    score = soft_min_val / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial guess: Regular 13-gon centered at origin, radius 1
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # We will perform a multi-scale annealing of the soft-min temperature tau.
    # Start with a larger tau (smoother, more global) and decrease it (sharper, more local).
    
    # List of tau values to try. We start high and go low.
    # The scale of areas is roughly O(1) for unit circle points.
    # tau = 1.0 is quite smooth, tau = 0.1 is sharper.
    tau_schedule = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    current_theta = init_points.flatten().copy()
    
    # Multi-start with different initializations
    # 1. Regular polygon
    # 2. Halton sequence (low-discrepancy)
    # 3. Random points in unit disk
    
    def halton_sequence(n, base):
        seq = []
        f = 1.0
        r = 0.0
        for i in range(1, n+1):
            f = 1.0 / base
            while i > 0:
                r += (i % base) * f
                i //= base
                f /= base
            seq.append(r)
        return np.array(seq)
    
    # Generate 3 distinct initializations
    inits = []
    
    # Init 1: Regular polygon
    inits.append(init_points)
    
    # Init 2: Halton sequence mapped to unit disk
    # Halton in 2D
    x_halton = halton_sequence(N, 2)
    y_halton = halton_sequence(N, 3)
    # Map [0,1] to unit disk? Or just use as is and scale?
    # Let's map to unit square then scale to unit disk radius 1
    # Actually, just placing them in the unit square and letting the optimizer move them is fine.
    # But for convexity, let's try to make them somewhat circular.
    # Simple mapping: (x-0.5)*2, (y-0.5)*2 gives square [-1,1]^2.
    # Let's just use a perturbed regular polygon for simplicity and robustness.
    # The prompt suggests Halton, so let's try to map Halton to a circle.
    # r = sqrt(x), theta = 2*pi*y
    r_halton = np.sqrt(x_halton)
    theta_halton = 2 * np.pi * y_halton
    halton_points = np.column_stack((r_halton * np.cos(theta_halton), r_halton * np.sin(theta_halton)))
    inits.append(halton_points)
    
    # Init 3: Random points in unit disk
    random.seed(42)
    r_rand = np.sqrt(np.random.rand(N))
    theta_rand = 2 * np.pi * np.random.rand(N)
    rand_points = np.column_stack((r_rand * np.cos(theta_rand), r_rand * np.sin(theta_rand)))
    inits.append(rand_points)
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        current_theta = init_pts.flatten().copy()
        
        for tau in tau_schedule:
            if time.time() > deadline - 5.0:
                break
                
            # Define objective for this tau
            obj_func = lambda x, t=tau: soft_min_objective(x, t)
            
            try:
                res = minimize(
                    obj_func,
                    current_theta,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < -best_score:
                    current_theta = res.x
                    # Compute actual score (hard min) to compare with global best
                    pts = current_theta.reshape(N, 2)
                    score = compute_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
                        
            except Exception:
                # If optimization fails, continue with next tau or break
                pass
                
    # Final refinement with very small tau (approaching hard min)
    if best_points is not None and time.time() < deadline - 2.0:
        final_tau = 0.001
        obj_final = lambda x: soft_min_objective(x, final_tau)
        try:
            res_final = minimize(
                obj_final,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_final.fun < -best_score:
                final_pts = res_final.x.reshape(N, 2)
                final_score = compute_score(final_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        best_points = init_points
        
    return best_points
