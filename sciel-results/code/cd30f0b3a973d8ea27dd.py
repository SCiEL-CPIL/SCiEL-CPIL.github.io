import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_init():
    """
    Generate a high-quality initial hexagonal packing.
    This serves as the starting point for the gradient optimization.
    """
    # We want to fit N=50 circles.
    # A hexagonal lattice with spacing d.
    # Area of hexagon ~ 0.866 * d^2.
    # N * 0.866 * d^2 ~ pi * 1^2 => d ~ sqrt(pi / (50 * 0.866)) ~ sqrt(0.072) ~ 0.27.
    # Let's try d=0.25 to be safe and let optimization expand it.
    
    points = []
    # Generate a grid of hexagonal points
    # Range needs to cover radius 1
    # max radius ~ 1.0
    # hex rows: y = j * sqrt(3)/2 * d
    # hex cols: x = i * d + (j%2)*d/2
    
    d = 0.25
    sqrt3_2 = np.sqrt(3) / 2
    
    for j in range(-6, 7):
        for i in range(-6, 7):
            x = i * d + (j % 2) * d / 2
            y = j * d * sqrt3_2
            # Ensure point is well inside to start, optimization will push out
            if np.hypot(x, y) < 1.0 - d/2 - 0.05:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) < N:
        # If we didn't get enough points, increase range or decrease d
        # But for N=50, d=0.25 should give plenty.
        # Let's just take the closest N to origin
        if len(points) == 0:
             # Fallback: random
             return np.random.uniform(-0.8, 0.8, size=(N, 2))
        # Sort by distance to center
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        return points[idx]
    
    # Sort by distance to center and take first N
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists)[:N]
    return points[idx]

def _soft_min_energy(x, beta):
    """
    Compute the negative soft-min of the constraints (distances and boundary).
    We want to maximize the minimum distance, so we minimize the negative of that.
    Soft min: -ln(sum(exp(-beta * c_i))) / beta
    As beta -> infinity, this approaches -min(c_i).
    """
    x = x.reshape(N, 2)
    
    # Boundary constraints: 1 - |x_i|
    r = np.hypot(x[:, 0], x[:, 1])
    c_boundary = 1.0 - r
    
    # Pairwise constraints: |x_i - x_j| / 2
    # We can compute all pairwise distances
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    
    # We only care about the minimum, but soft-min needs all values.
    # To save memory/time, we can just use the full matrix since N=50 is small.
    c_pair = dists / 2.0
    
    # Combine all constraints
    # c_boundary has N elements. c_pair has N*(N-1)/2 elements (upper triangle).
    # Let's flatten c_pair upper triangle
    iu = np.triu_indices(N, k=1)
    c_pair_flat = c_pair[iu]
    
    all_c = np.concatenate([c_boundary, c_pair_flat])
    
    # Soft min
    # To avoid overflow/underflow in exp, shift by max
    max_c = np.max(all_c)
    exp_vals = np.exp(-beta * (all_c - max_c))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = max_c - log_sum / beta
    
    # We want to maximize soft_min_val, so return negative
    return -soft_min_val

def _soft_min_grad(x, beta):
    """
    Gradient of the soft-min energy.
    """
    x = x.reshape(N, 2)
    
    r = np.hypot(x[:, 0], x[:, 1])
    c_boundary = 1.0 - r
    
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    c_pair = dists / 2.0
    
    iu = np.triu_indices(N, k=1)
    c_pair_flat = c_pair[iu]
    
    all_c = np.concatenate([c_boundary, c_pair_flat])
    
    max_c = np.max(all_c)
    shifted = all_c - max_c
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # Gradient of soft-min w.r.t c_k is:
    # d/dc_k (-ln(sum(exp(-beta*c))) / beta) = -1/beta * (1/sum_exp) * (-beta * exp(-beta*c_k))
    # = exp(-beta*c_k) / sum_exp
    # This is the probability p_k that constraint k is the "active" minimum.
    
    weights = exp_vals / sum_exp
    
    grad_x = np.zeros_like(x)
    
    # Boundary part
    # c_boundary_i = 1 - sqrt(x_i^2 + y_i^2)
    # d c_boundary_i / d x_i = -x_i / r_i
    # d c_boundary_i / d y_i = -y_i / r_i
    
    w_b = weights[:N]
    for i in range(N):
        if r[i] > 1e-8:
            grad_x[i, 0] += w_b[i] * (-x[i, 0] / r[i])
            grad_x[i, 1] += w_b[i] * (-x[i, 1] / r[i])
        else:
            # At center, gradient of distance is undefined, but symmetric.
            # Just skip or add 0.
            pass
            
    # Pairwise part
    # c_pair_ij = 0.5 * sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    # d c_pair_ij / d x_i = 0.5 * (x_i - x_j) / dist_ij
    # d c_pair_ij / d x_j = 0.5 * (x_j - x_i) / dist_ij = - d c_pair_ij / d x_i
    
    w_p = weights[N:]
    idx_i, idx_j = iu
    for k in range(len(w_p)):
        i = idx_i[k]
        j = idx_j[k]
        w = w_p[k]
        d = dists[i, j]
        if d > 1e-8:
            # Gradient w.r.t x_i
            grad_x[i, 0] += w * (0.5 * (x[i, 0] - x[j, 0]) / d)
            grad_x[i, 1] += w * (0.5 * (x[i, 1] - x[j, 1]) / d)
            # Gradient w.r.t x_j
            grad_x[j, 0] += w * (0.5 * (x[j, 0] - x[i, 0]) / d)
            grad_x[j, 1] += w * (0.5 * (x[j, 1] - x[i, 1]) / d)

    # The objective function returns -soft_min.
    # So gradient of objective is - (gradient of soft_min)
    # Wait, let's re-verify signs.
    # f(x) = - soft_min(c(x))
    # df/dx = - (d soft_min / d c) * (dc / dx)
    # d soft_min / d c_k = - p_k (where p_k = exp(-beta c_k)/sum)
    # So df/dx = - (-p_k) * (dc_k/dx) = p_k * (dc_k/dx)
    
    # My derivation above for "weights" was p_k.
    # And I calculated dc_k/dx.
    # So the gradient of the objective (which is -soft_min) is exactly what I computed: sum p_k * dc_k/dx.
    
    return grad_x.flatten()

def construct():
    deadline = time.time() + 145.0
    
    # 1. Generate initial hexagonal packing
    x0 = _hexagonal_init()
    x0_flat = x0.flatten()
    
    # Ensure validity of start
    r_init = _compute_radius(x0)
    if r_init <= 0:
        # Fallback random
        x0 = np.random.uniform(-0.8, 0.8, size=(N, 2))
        x0_flat = x0.flatten()
        r_init = _compute_radius(x0)
        
    best_x = x0.copy()
    best_r = r_init
    
    # 2. Gradient Optimization with Annealing of beta
    # Start with low beta (smooth landscape) and increase to high beta (sharp minimum)
    betas = [10, 50, 100, 200, 500, 1000]
    
    current_x = x0_flat
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Define objective and gradient for this beta
        def obj(x):
            return _soft_min_energy(x, beta)
            
        def grad(x):
            return _soft_min_grad(x, beta)
            
        # Use L-BFGS-B
        # Bounds: keep points within [-1, 1] roughly, though soft-min handles it.
        # Hard bounds prevent flying out.
        bounds = [(-1.0, 1.0)] * (N * 2)
        
        try:
            res = minimize(
                obj, 
                current_x, 
                method='L-BFGS-B', 
                jac=grad, 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < obj(current_x):
                new_x = res.x.reshape(N, 2)
                new_r = _compute_radius(new_x)
                
                if new_r > best_r:
                    best_r = new_r
                    best_x = new_x.copy()
                
                # Update current_x for next beta
                # Only update if we made progress or if it's valid
                if new_r > 0:
                    current_x = new_x.flatten()
                    
        except Exception:
            pass
            
    # 3. Final Polish with Simulated Annealing (optional, if time permits)
    # The gradient method might get stuck in a very local minimum.
    # A short SA run with the current best as start can sometimes jump out.
    
    if time.time() < deadline - 5:
        current = best_x.copy()
        current_r = best_r
        T = 0.01
        min_T = 1e-5
        cooling = 0.95
        
        while time.time() < deadline - 2 and T > min_T:
            # Small random perturbation
            noise = np.random.normal(0, T * 0.05, size=(N, 2))
            new_centers = current + noise
            
            new_r = _compute_radius(new_centers)
            
            if new_r > current_r:
                current = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_x = current.copy()
            else:
                delta = current_r - new_r
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current = new_centers
                    current_r = new_r
            
            T *= cooling
            
    return best_x

# Test/Debug
if __name__ == "__main__":
    c = construct()
    print("Radius:", _compute_radius(c))
    print(c)
