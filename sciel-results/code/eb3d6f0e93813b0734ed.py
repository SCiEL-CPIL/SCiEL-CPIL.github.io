"""Rewritten program for circle packing in unit square."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _soft_min_dist(params, beta):
    """Compute a smooth surrogate for the minimum pair distance minus sum of radii.
    params: (x, y, r) arrays each of length N.
    Returns a scalar that is ~0 when all circles are non-overlapping and negative when overlapping.
    We want to maximize this, so we minimize its negative.
    """
    x, y, r = params
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    # Required separation: r_i + r_j
    sep = r[:, None] + r[None, :]
    # Violation: sep - dist (positive means overlap)
    viol = sep - dist
    # Zero out diagonal
    np.fill_diagonal(viol, 0.0)
    # Soft-min of the negative violations (i.e., soft-max of violations)
    # We want a measure of how much the circles overlap.
    # Use log-sum-exp for numerical stability: soft_max(viol) = log(sum(exp(beta*viol))) / beta
    # This is a smooth approximation to max(viol).
    # If max(viol) <= 0, circles are mostly non-overlapping.
    # We want to penalize positive violations.
    # Let's use: penalty = (1/beta) * log(sum(exp(beta * viol))) for viol > some threshold
    # Actually, let's use a simpler approach: sum of soft penalties
    # penalty_i_j = max(0, viol_i_j)^2 is nonsmooth.
    # Use: (1/beta) * log(1 + exp(beta * viol)) as a smooth max(0, viol)
    # This is the softplus function.
    # Sum over all pairs.
    upper = np.triu_indices(N, k=1)
    v = viol[upper]
    # softplus: log(1 + exp(beta * v)) / beta
    # For large beta*v, this is ~v. For small, it's ~log(2)/beta.
    # We want to penalize overlaps, so we minimize the sum of softplus(beta * v) / beta
    # But we also need to keep circles inside the square.
    # Boundary violations:
    # x - r < 0, 1 - x - r < 0, y - r < 0, 1 - y - r < 0
    bx = np.maximum(r - x, 0) + np.maximum(r - (1 - x), 0)
    by = np.maximum(r - y, 0) + np.maximum(r - (1 - y), 0)
    bviol = np.concatenate([bx, by])
    # Sum of softplus for all violations
    # For pair violations:
    sp_pairs = np.logaddexp(0, beta * v) / beta
    # For boundary violations:
    sp_bound = np.logaddexp(0, beta * bviol) / beta
    # Total penalty
    penalty = np.sum(sp_pairs) + np.sum(sp_bound)
    # We want to maximize sum(r), so objective = -sum(r) + penalty
    return -np.sum(r) + penalty


def _objective(params, beta):
    """Objective function for minimization."""
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    return _soft_min_dist((x, y, r), beta)


def _objective_grad(params, beta):
    """Gradient of the objective function."""
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Compute pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, 1.0)  # Avoid division by zero
    
    sep = r[:, None] + r[None, :]
    viol = sep - dist
    
    # Softplus derivative: sigmoid(beta * v)
    # d/dv [log(1+exp(beta*v))/beta] = exp(beta*v)/(1+exp(beta*v)) = sigmoid(beta*v)
    sig_pairs = 1.0 / (1.0 + np.exp(-beta * viol))
    # Zero out diagonal
    np.fill_diagonal(sig_pairs, 0.0)
    
    # Gradient with respect to x
    # d(viol)/dx_i = 1 - (x_i - x_j)/dist_ij for j > i, and 1 - (x_j - x_i)/dist_ji for j < i
    # More carefully:
    # viol_ij = r_i + r_j - dist_ij
    # d(viol_ij)/dx_i = -(x_i - x_j)/dist_ij
    # d(viol_ij)/dx_j = +(x_i - x_j)/dist_ij
    
    # grad_x[i] = sum_j sig_ij * d(viol_ij)/dx_i
    # For i != j: d(viol_ij)/dx_i = -(x_i - x_j)/dist_ij
    # So grad_x[i] = -sum_{j!=i} sig_ij * (x_i - x_j)/dist_ij
    
    # Vectorized:
    # dx/dist is the unit vector from j to i
    ux = dx / dist  # (i,j) component: (x_i - x_j)/dist_ij
    uy = dy / dist
    
    # grad_x[i] = -sum_j sig_ij * ux_ij
    grad_x = -np.sum(sig_pairs * ux, axis=1)
    grad_y = -np.sum(sig_pairs * uy, axis=1)
    
    # Gradient with respect to r
    # d(viol_ij)/dr_i = 1, d(viol_ij)/dr_j = 1
    grad_r = np.sum(sig_pairs, axis=1)  # Each r_i appears in N-1 pairs
    
    # Boundary gradients
    bx_left = np.maximum(r - x, 0)
    bx_right = np.maximum(r - (1 - x), 0)
    by_left = np.maximum(r - y, 0)
    by_right = np.maximum(r - (1 - y), 0)
    
    bviol = np.concatenate([bx_left, bx_right, by_left, by_right])
    sig_bound = 1.0 / (1.0 + np.exp(-beta * bviol))
    
    # d(bx_left)/dx = -1 if r-x > 0, else 0 -> -sig * 1
    # d(bx_left)/dr = 1 if r-x > 0, else 0 -> sig * 1
    # d(bx_right)/dx = 1 if 1-x-r > 0, else 0 -> sig * 1
    # d(bx_right)/dr = 1 if 1-x-r > 0, else 0 -> sig * 1
    
    grad_x += -sig_bound[:N] + sig_bound[N:2*N]
    grad_y += -sig_bound[2*N:3*N] + sig_bound[3*N:]
    grad_r += sig_bound[:N] + sig_bound[N:2*N] + sig_bound[2*N:3*N] + sig_bound[3*N:]
    
    # Gradient of -sum(r)
    grad_r -= 1.0
    
    return np.concatenate([grad_x, grad_y, grad_r])


def _construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Initial configurations
    def _hex_init():
        # Hexagonal packing initialization
        pts = []
        r0 = 0.1
        rows = 0
        while len(pts) < N:
            for col in range(0, 10):
                x = (col + 0.5) * r0 * 2
                y = (rows + 0.5) * r0 * np.sqrt(3)
                if x < 1 and y < 1:
                    pts.append((x, y))
                if len(pts) >= N:
                    break
            rows += 1
        pts = pts[:N]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        r = np.full(N, r0 * 0.5)
        return np.concatenate([x, y, r])
    
    def _grid_init():
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:N])
        y = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:N])
        r = np.full(N, 0.5 / side - 0.01)
        return np.concatenate([x, y, r])
    
    def _random_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = rng.uniform(0.05, 0.15, N)
        return np.concatenate([x, y, r])
    
    inits = [
        _hex_init(),
        _grid_init(),
        _random_init(42),
        _random_init(123),
        _random_init(456),
    ]
    
    # Beta schedule: start low, increase
    beta_schedule = [10, 50, 100, 500, 1000, 2000, 5000]
    
    for init_idx, init_params in enumerate(inits):
        if time.time() > deadline:
            break
        
        params = init_params.copy()
        
        for beta in beta_schedule:
            if time.time() > deadline:
                break
            
            # Run L-BFGS
            try:
                res = minimize(
                    _objective,
                    params,
                    args=(beta,),
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                params = res.x
            except Exception:
                pass
            
            # Evaluate actual score
            x = params[:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            
            # Check validity
            valid = True
            if np.any(r <= 0) or np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
                valid = False
            if valid:
                for i in range(N):
                    if x[i] - r[i] < -1e-9 or 1 - x[i] - r[i] < -1e-9 or y[i] - r[i] < -1e-9 or 1 - y[i] - r[i] < -1e-9:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
    
    # Final high-precision polish
    if best_params is not None and time.time() < deadline:
        for beta in [200, 1000, 5000]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    _objective,
                    best_params,
                    args=(beta,),
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                best_params = res.x
            except Exception:
                pass
    
    if best_params is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Clamp to valid range
    r = np.maximum(r, 1e-10)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    r = np.minimum(r, np.minimum(x, 1 - x))
    r = np.minimum(r, np.minimum(y, 1 - y))
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END


def construct():
    return _construct()
