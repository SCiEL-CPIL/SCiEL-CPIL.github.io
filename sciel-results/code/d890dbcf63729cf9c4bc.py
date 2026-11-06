import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(c):
    """Compute the actual packing radius."""
    if c.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(c)):
        return -np.inf
    
    # Distance to center
    r_centers = np.linalg.norm(c, axis=1)
    if np.any(r_centers >= 1.0):
        return -np.inf
        
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(d_sq, np.inf)
    min_d_sq = np.min(d_sq)
    min_d = np.sqrt(min_d_sq)
    
    r = min(min_d / 2.0, 1.0 - np.max(r_centers))
    return r

def _objective_and_grad(c, T):
    """
    Objective: Minimize the log-sum-exp of "badness".
    Badness for pair (i,j) is exp(-d_ij^2 / T).
    Badness for center i is exp(-(1-|c_i|)^2 / T).
    We want to minimize the sum of these badnesses.
    f(c) = (1/T) * ln( sum_badness )
    """
    c = c.reshape(N, 2)
    
    # Pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(d_sq, np.inf)
    
    # Center distances
    r_sq = np.sum(c**2, axis=1)
    dist_to_border_sq = (1.0 - np.sqrt(r_sq))**2
    # Handle points outside (shouldn't happen if constrained, but for safety)
    # If r > 1, dist_to_border_sq is positive, but we want to penalize being outside.
    # Actually, if r > 1, the term (1-r) is negative, squared is positive.
    # But we want to push them in.
    # Let's use (max(0, r-1))^2 + (1-r)^2? No.
    # The term exp(-(1-|c|)^2/T) is large when |c| is close to 1.
    # If |c| > 1, (1-|c|)^2 is positive, so exp is small. This is bad.
    # We need a term that penalizes |c| > 1.
    # Let's define badness_center = exp(-min(0, 1-|c|)^2 / T) ? No.
    # Let's just use a hard constraint in the optimizer if possible, or a penalty.
    # L-BFGS-B supports bounds. But we have 2N variables.
    # Bounds: -1 <= x_i <= 1, -1 <= y_i <= 1.
    # This doesn't enforce x^2+y^2 <= 1.
    
    # Let's modify the center term.
    # We want to maximize 1 - |c_i|.
    # Let h_i = 1 - ||c_i||.
    # Badness = exp(-h_i^2 / T) if h_i > 0.
    # If h_i <= 0 (outside), we want huge badness.
    # Let's use: badness_i = exp(-max(0, h_i)^2 / T) + max(0, -h_i)^2 * 100
    # This is getting complicated for the gradient.
    
    # Alternative: Just use the pairwise term and rely on the fact that if points go outside,
    # the pairwise distances might increase, but the "container" constraint is critical.
    # Let's add a penalty term for being outside.
    
    r_norm = np.sqrt(r_sq)
    h = 1.0 - r_norm
    # If h < 0, it's outside.
    # Let's use a smooth penalty: P_out = sum( max(0, -h)^2 )
    # And the log-sum-exp for the "tightness".
    
    # Actually, a simpler objective for packing is:
    # Minimize: sum_{i<j} exp(-d_ij^2/T) + sum_i exp(-(1-||c_i||)^2/T)
    # This works well if points stay inside.
    # To ensure they stay inside, we can add a large penalty if they go out.
    
    exp_pair = np.exp(-d_sq / T)
    np.fill_diagonal(exp_pair, 0)
    sum_pair = np.sum(exp_pair)
    
    exp_center = np.exp(-h**2 / T)
    sum_center = np.sum(exp_center)
    
    # Penalty for going outside
    penalty = 0.0
    grad_penalty = np.zeros_like(c)
    outside_mask = h < 0
    if np.any(outside_mask):
        # Quadratic penalty
        penalty = 100.0 * np.sum(h[outside_mask]**2)
        # Gradient of penalty: d/dx (100 * (1 - ||x||)^2) for x outside
        # d/dx (1 - r)^2 = 2(1-r) * (-1/r * x) = -2(1-r)/r * x
        # Note: 1-r is negative.
        h_out = h[outside_mask]
        r_out = r_norm[outside_mask]
        # grad = 100 * 2 * h_out * (-c_out / r_out)
        # Wait, h = 1 - r. dh/dr = -1. dr/dx = x/r.
        # d(h^2)/dx = 2h * (-x/r)
        # d(penalty)/dx = 100 * 2 * h * (-x/r)
        grad_penalty[outside_mask] = 100.0 * 2.0 * h_out[:, np.newaxis] * (-c[outside_mask] / r_out[:, np.newaxis])
        
    total_val = (1.0 / T) * np.log(sum_pair + sum_center + 1e-300) + penalty
    
    # Gradient
    # d/dc [ (1/T) ln(S) ] = (1/T) * (1/S) * dS/dc
    # S = sum_pair + sum_center
    # dS/dc_k (from pair) = sum_{j!=k} exp(-d_kj^2/T) * (-1/T) * 2(c_k - c_j)
    # dS/dc_k (from center) = exp(-h_k^2/T) * (-1/T) * 2h_k * (-c_k/r_k) = exp(...) * (2h_k/T) * (c_k/r_k)
    
    S = sum_pair + sum_center
    if S < 1e-300:
        S = 1e-300
        
    grad = np.zeros_like(c)
    
    # Pair contribution
    # exp_pair[k,j] = exp(-d_kj^2/T)
    # term = (1/T) * (1/S) * sum_j exp_pair[k,j] * (-2/T) * (c_k - c_j)
    # This is O(N^2) per point, so O(N^3) total. N=50, 125k ops. Fast.
    
    for k in range(N):
        # Pair
        weights = exp_pair[k, :] / (T * S)
        # Sum of weights * (c_k - c_j)
        # weights has inf on diagonal, but exp_pair diag is 0, so weight is 0.
        vec_sum = np.sum(weights[:, np.newaxis] * (c[k] - c), axis=0)
        grad[k] += -2.0 * vec_sum
        
        # Center
        # dS_center/dc_k = exp_center[k] * (2 h_k / T) * (c_k / r_k)
        if r_norm[k] > 1e-8:
            grad_center = exp_center[k] * (2.0 * h[k] / T) * (c[k] / r_norm[k])
        else:
            grad_center = 0.0
        grad[k] += (1.0 / T) * grad_center / S
        
    grad += grad_penalty
    
    return total_val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -np.inf
    best_centers = None
    
    # Define initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    spacing = 0.1
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            init = points[np.argsort(radius, kind="stable")[:N]]
            inits.append(init)
            break
        spacing *= 0.95
        
    # 2. Concentric rings
    # 1, 6, 12, 18, 13 (50 total)
    rings = [1, 6, 12, 18, 13]
    points = [np.array([[0, 0]])]
    for i, n in enumerate(rings[1:], 1):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        # Estimate radius
        r_est = (i + 0.5) * 0.2 
        pts = np.column_stack([r_est * np.cos(angles), r_est * np.sin(angles)])
        points.append(pts)
    init_rings = np.vstack(points)
    inits.append(init_rings)
    
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)

    # Optimization Loop
    for init_idx, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        c = init.copy()
        # Ensure inside
        r = np.linalg.norm(c, axis=1)
        mask = r > 0.99
        if np.any(mask):
            c[mask] *= 0.99 / r[mask, np.newaxis]
            
        T = 0.1
        # Annealing schedule
        for step in range(50):
            if time.time() > deadline:
                break
                
            try:
                result = minimize(
                    _objective_and_grad,
                    c.ravel(),
                    args=(T,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-6}
                )
                c = result.x.reshape(N, 2)
                
                # Check validity
                r_curr = np.linalg.norm(c, axis=1)
                if np.any(r_curr >= 1.0):
                    # Pull back in
                    mask = r_curr >= 1.0
                    c[mask] *= 0.99 / r_curr[mask, np.newaxis]
                    
            except Exception:
                pass
                
            # Reduce T
            T *= 0.8
            if T < 1e-5:
                break
                
        # Final polish with very small T
        try:
            result = minimize(
                _objective_and_grad,
                c.ravel(),
                args=(1e-5,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            c = result.x.reshape(N, 2)
        except Exception:
            pass
        
        # Evaluate
        score = _score(c)
        if score > best_score:
            best_score = score
            best_centers = c.copy()
            
    if best_centers is None:
        return inits[0]
        
    return best_centers
