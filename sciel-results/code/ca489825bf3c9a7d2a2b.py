"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _score(centers):
    """Compute the packing radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    dist_boundary = np.minimum(0.5 - np.abs(centers), axis=1).min()
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    return min(dist_boundary, min_pair / 2.0)

def _soft_min_and_grad(centers, beta=50.0):
    """Compute soft-min of all constraints and its gradient."""
    centers = np.asarray(centers).reshape(-1, 2)
    n = centers.shape[0]
    
    # Distances to boundary
    # For each point, distance to nearest edge
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    dists_boundary = np.minimum(dist_x, dist_y)  # (n,)
    
    # Pairwise distances / 2
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    pair_dists = d / 2.0  # (n, n)
    
    # All constraint values
    all_vals = np.concatenate([dists_boundary, pair_dists.reshape(-1)])
    all_vals = np.maximum(all_vals, 1e-12)  # Avoid log(0)
    
    # Soft-min: m - (1/beta) * log(sum(exp(beta*(v-m))))
    m = np.min(all_vals)
    exp_vals = np.exp(beta * (all_vals - m))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = m - log_sum / beta
    
    # Gradient of soft-min w.r.t. centers
    # d(soft_min)/d(all_vals) = -exp_vals / sum(exp_vals)
    weights = -exp_vals / np.sum(exp_vals)
    
    grad = np.zeros((n, 2))
    
    # Gradient from boundary distances
    # dist_x = 0.5 - |x|, d(dist_x)/dx = -sign(x)
    # dist_y = 0.5 - |y|, d(dist_y)/dy = -sign(y)
    # dist_boundary = min(dist_x, dist_y)
    # This is non-differentiable at min, but soft-min handles it
    
    # For soft-min, we need gradient of each constraint
    # Let's compute gradient of each boundary dist and pair dist
    
    # Boundary: for point i, constraint is min(0.5-|xi|, 0.5-|yi|)
    # Gradient of 0.5-|xi| w.r.t. xi = -sign(xi)
    # Gradient of 0.5-|yi| w.r.t. yi = -sign(yi)
    
    # Since we use soft-min over all constraints, the gradient is:
    # sum over all constraints of weight_j * grad(constraint_j)
    
    # Boundary constraints: 2 per point (one for x, one for y)
    # Actually, the constraint is min(dist_x, dist_y), which is non-smooth.
    # Let's use separate x and y boundary constraints for smoothness.
    
    # Redo: use 2 boundary constraints per point (x and y)
    # and all pairwise constraints
    
    # Recompute with separate x, y boundary constraints
    all_vals2 = np.concatenate([dist_x, dist_y, pair_dists.reshape(-1)])
    all_vals2 = np.maximum(all_vals2, 1e-12)
    
    m2 = np.min(all_vals2)
    exp_vals2 = np.exp(beta * (all_vals2 - m2))
    log_sum2 = np.log(np.sum(exp_vals2))
    soft_min_val2 = m2 - log_sum2 / beta
    
    weights2 = -exp_vals2 / np.sum(exp_vals2)
    
    grad2 = np.zeros((n, 2))
    
    # x boundary: 0.5 - |xi|, grad = -sign(xi)
    for i in range(n):
        w = weights2[i]
        grad2[i, 0] += w * (-np.sign(centers[i, 0]))
        # y boundary
        wy = weights2[n + i]
        grad2[i, 1] += wy * (-np.sign(centers[i, 1]))
    
    # Pairwise: for pair (i,j), constraint is |ci-cj|/2
    # d(|ci-cj|/2)/dci = (ci-cj)/|ci-cj|
    # d(|ci-cj|/2)/dcj = (cj-ci)/|ci-cj|
    pair_weights = weights2[2*n:].reshape(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_weights[i, j]
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                grad2[i] += w * diff / dist
                grad2[j] += w * (-diff) / dist
    
    return -soft_min_val2, -grad2  # Negate because we want to maximize

def _hex_grid_init():
    """Generate a hexagonal grid initial configuration for N circles."""
    points = []
    best_points = None
    best_count = 0
    
    for s in np.linspace(0.03, 0.12, 150):
        vs = s * np.sqrt(3) / 2
        n_rows = int(np.floor(1.0 / vs)) + 1
        n_cols = int(np.floor(1.0 / s)) + 1
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + vs * (row + 0.5)
            if y > 0.5:
                break
            offset = 0.0 if row % 2 == 0 else s / 2.0
            for col in range(n_cols):
                x = -0.5 + s * (col + 0.5) + offset
                if x > 0.5:
                    break
                pts.append((x, y))
        
        if len(pts) > best_count:
            best_count = len(pts)
            best_points = pts
    
    if best_points is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    pts = np.array(best_points[:N]) if len(best_points) >= N else np.array(best_points)
    
    if len(pts) < N:
        rng = np.random.RandomState(42)
        while len(pts) < N:
            p = rng.uniform(-0.45, 0.45, size=2)
            pts = np.vstack([pts, p])
    
    return pts[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    centers = _hex_grid_init()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Multi-stage optimization
    betas = [5.0, 15.0, 40.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 15.0:
            break
        
        bounds = [(-0.4999, 0.4999)] * (2 * N)
        
        def objective(x):
            if time.time() > deadline - 5.0:
                return 1e10
            c = x.reshape(-1, 2)
            val, _ = _soft_min_and_grad(c, beta=beta)
            return val
        
        def gradient(x):
            if time.time() > deadline - 5.0:
                return np.zeros_like(x)
            c = x.reshape(-1, 2)
            _, grad = _soft_min_and_grad(c, beta=beta)
            return grad.flatten()
        
        try:
            result = minimize(
                objective, 
                centers.flatten(), 
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 40}
            )
            
            new_centers = result.x.reshape(-1, 2)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                centers = new_centers.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    # Final local search with exact score
    if time.time() < deadline - 3.0:
        rng = np.random.RandomState(999)
        step_size = 0.005
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            direction = rng.normal(0, 1, size=(N, 2))
            perturbed = best_centers + step_size * direction
            perturbed = np.clip(perturbed, -0.4999, 0.4999)
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_centers = perturbed.copy()
            else:
                step_size *= 0.9
    
    return best_centers
# EVOLVE-BLOCK-END
