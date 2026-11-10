"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

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
    x = centers[:, 0]
    y = centers[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate a hexagonal lattice packing inside the unit circle."""
    best_points = None
    best_radius = 0.0
    
    # Try a range of spacings to find the best lattice cut
    for spacing in np.linspace(0.15, 0.25, 20):
        k = int(1.0 / spacing) + 2
        # Generate hexagonal lattice points
        # x = i * spacing + (j % 2) * spacing / 2
        # y = j * spacing * sqrt(3) / 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        # Keep points strictly inside (with buffer) to avoid initial boundary violations
        # Buffer is half the spacing to ensure initial non-overlap in the lattice
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            # Pick the N points closest to the origin to form a compact cluster
            points = points[np.argsort(radius, kind='stable')[:N]]
            r = _compute_radius(points)
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    # Fallback: simple grid or random if lattice fails (unlikely)
    spacing = 0.2
    k = int(1.0 / spacing) + 2
    rows = [(i * spacing, j * spacing) for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    radius = radius[mask]
    if len(points) >= N:
        return points[np.argsort(radius, kind='stable')[:N]]
    
    # Last resort: random
    return np.random.rand(N, 2) * 1.5 - 0.75

def _softmin_obj_and_grad(flat_centers, beta):
    """
    Objective: minimize -softmin of (pairwise dists/2, boundary dists)
    Gradient computed analytically.
    
    flat_centers: shape (2*N,)
    beta: float, sharpness of the softmin
    """
    centers = flat_centers.reshape(N, 2)
    
    # Boundary distances
    r_norm = np.hypot(centers[:, 0], centers[:, 1])
    boundary_dists = 1.0 - r_norm
    
    # Pairwise distances / 2
    x = centers[:, 0]
    y = centers[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.hypot(dx, dy)
    np.fill_diagonal(dists, 0.0)
    
    # We only need upper triangle for unique pairs
    idx_i, idx_j = np.triu_indices(N, k=1)
    pair_dists = dists[idx_i, idx_j] / 2.0
    
    # Collect all constraints
    # all_dists[0:N] are boundary dists
    # all_dists[N:] are pair dists
    all_dists = np.concatenate([boundary_dists, pair_dists])
    
    # Soft-min: softmin(d) = -1/beta * log(sum(exp(-beta * d)))
    # We want to maximize softmin, so minimize f = -softmin(d) = 1/beta * log(sum(exp(-beta * d)))
    
    # Numerical stability: subtract max argument from exponent
    # exp(-beta * d)
    # Let M = max(-beta * d) = -beta * min(d)
    # sum(exp(-beta * d)) = exp(M) * sum(exp(-beta * d - M))
    # log(sum(exp(-beta * d))) = M + log(sum(exp(-beta * d - M)))
    
    shifted = -beta * all_dists
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        return 0.0, np.zeros(2*N)
        
    log_sum_exp = np.log(sum_exp)
    
    # f = 1/beta * (max_shifted + log_sum_exp)
    obj = (max_shifted + log_sum_exp) / beta
    
    # Gradient:
    # f = 1/beta * (max_shifted + log(sum_exp))
    # Note: max_shifted depends on all_dists, but in the context of the log-sum-exp trick,
    # the gradient of log(sum(exp(-beta*d))) w.r.t d_i is:
    # d/d d_i [ log(sum_j exp(-beta*d_j)) ] = (1/sum_j exp(-beta*d_j)) * (-beta * exp(-beta*d_i))
    # = -beta * (exp(-beta*d_i) / sum_j exp(-beta*d_j))
    # = -beta * w_i  where w_i is the softmax weight.
    
    # So d f / d d_i = 1/beta * (-beta * w_i) = -w_i
    
    weights = exp_shifted / sum_exp
    
    # Now map gradient from all_dists to centers
    grad_centers = np.zeros((N, 2))
    
    # 1. Boundary part: all_dists[0:N]
    # boundary_dists[i] = 1 - hypot(centers[i])
    # d(boundary_dists[i])/d(centers[i]) = -centers[i] / hypot(centers[i])
    for i in range(N):
        if r_norm[i] > 1e-12:
            # grad_all_dists[i] = -weights[i]
            # grad_centers[i] += grad_all_dists[i] * d(boundary_dists[i])/d(centers[i])
            grad_centers[i] += (-weights[i]) * (-centers[i] / r_norm[i])
    
    # 2. Pairwise part: all_dists[N:]
    # pair_dists[k] corresponds to pair (idx_i[k], idx_j[k])
    # pair_dists[k] = hypot(centers[i]-centers[j]) / 2
    # Let d_ij = hypot(centers[i]-centers[j])
    # d(pair_dists[k])/d(centers[i]) = (centers[i]-centers[j]) / (2 * d_ij)
    # d(pair_dists[k])/d(centers[j]) = (centers[j]-centers[i]) / (2 * d_ij) = - (centers[i]-centers[j]) / (2 * d_ij)
    
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        d_ij = dists[i, j]
        if d_ij < 1e-12:
            continue
        
        # grad_all_dists[N+k] = -weights[N+k]
        w = -weights[N + k]
        
        diff = centers[i] - centers[j]
        grad_vec = diff / (2.0 * d_ij)
        
        grad_centers[i] += w * grad_vec
        grad_centers[j] += w * (-grad_vec)
    
    grad_flat = grad_centers.flatten()
    
    return obj, grad_flat

def _gradient_polish(centers, beta, max_time=10.0):
    """Use L-BFGS-B to polish the packing with a specific beta."""
    start = time.time()
    
    def objective(flat):
        if time.time() - start > max_time:
            # Return a large value to signal termination or just stop
            return 0.0, np.zeros(2*N)
        obj, grad = _softmin_obj_and_grad(flat, beta)
        return obj, grad
    
    flat_init = centers.flatten()
    
    # Bounds: coordinates must be within [-1, 1] roughly, but circle constraint is handled by objective
    # Using box bounds is safe for L-BFGS-B
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        result = minimize(
            objective,
            flat_init,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        
        if result.success or result.fun < objective(flat_init)[0]:
            polished = result.x.reshape(N, 2)
            if np.all(np.isfinite(polished)):
                return polished
    except Exception:
        pass
    
    return centers

def construct():
    """Construct the packing of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Initial Configuration: Hexagonal Packing
    centers = _hexagonal_packing()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # 2. Multi-stage Optimization
    # Start with smooth objective (low beta) to escape local minima, then sharpen
    betas = [10.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        current_time = time.time()
        time_remaining = deadline - current_time - 5  # 5s safety margin
        
        if time_remaining < 2:
            break
            
        # Allocate time based on beta: earlier stages get more time
        max_iter_time = min(20.0, time_remaining / (len(betas) + 1))
        
        polished = _gradient_polish(best_centers, beta, max_time=max_iter_time)
        polished_radius = _compute_radius(polished)
        
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished.copy()
    
    # 3. Final High-Precision Polish
    # If time remains, do a final polish with a high beta and tight tolerances
    if time.time() < deadline - 5:
        time_remaining = deadline - time.time() - 2
        if time_remaining > 2:
            polished = _gradient_polish(best_centers, 500.0, max_time=time_remaining)
            polished_radius = _compute_radius(polished)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished.copy()
    
    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius > 0 and np.all(np.isfinite(best_centers)):
        return best_centers
    
    # Fallback
    return _hexagonal_packing()

# EVOLVE-BLOCK-END
