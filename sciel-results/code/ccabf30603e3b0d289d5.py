"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 14
D = 3
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

# EVOLVE-BLOCK-START
def _pairwise_dists_sq(P):
    """Compute squared pairwise distances. Returns array of shape (N, N)."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)

def _score(P):
    """Compute the squared ratio of max to min pairwise distance."""
    d2 = _pairwise_dists_sq(P)
    # Exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    d2_masked = d2[mask]
    if d2_masked.min() < 1e-12:
        return 1e10
    return (d2_masked.max() / d2_masked.min())

def _objective(P_flat, tau=100.0):
    """Smooth surrogate objective using log-sum-exp for max and soft-min for min."""
    P = P_flat.reshape(N, D)
    d2 = _pairwise_dists_sq(P)
    mask = ~np.eye(N, dtype=bool)
    d2_masked = d2[mask]
    
    # Soft max via log-sum-exp
    max_d2 = np.log(np.sum(np.exp(d2_masked / tau))) * tau
    # Soft min via negative log-sum-exp
    min_d2 = -np.log(np.sum(np.exp(-d2_masked / tau))) * tau
    
    if min_d2 < 1e-12:
        return 1e10
    return max_d2 / min_d2

def _objective_grad(P_flat, tau=100.0):
    """Gradient of the smooth surrogate objective."""
    P = P_flat.reshape(N, D)
    d2 = _pairwise_dists_sq(P)
    mask = ~np.eye(N, dtype=bool)
    d2_masked = d2[mask]
    
    # Weights for soft max
    w_max = np.exp(d2_masked / tau)
    w_max /= np.sum(w_max)
    
    # Weights for soft min
    w_min = np.exp(-d2_masked / tau)
    w_min /= np.sum(w_min)
    
    max_d2 = np.log(np.sum(np.exp(d2_masked / tau))) * tau
    min_d2 = -np.log(np.sum(np.exp(-d2_masked / tau))) * tau
    
    if min_d2 < 1e-12:
        return np.zeros_like(P_flat)
    
    # d/dP of (max/min) = (1/min) * d(max)/dP - (max/min^2) * d(min)/dP
    # d(max_d2)/dP_i = (2/tau) * sum_j w_max_ij * (P_i - P_j) for i != j
    # d(min_d2)/dP_i = (2/tau) * sum_j w_min_ij * (P_i - P_j) for i != j
    
    grad = np.zeros_like(P)
    # Reshape weights to (N, N)
    w_max_full = np.zeros((N, N))
    w_min_full = np.zeros((N, N))
    idx = np.where(mask)
    w_max_full[idx] = w_max
    w_min_full[idx] = w_min
    
    # For each point i, grad_i = (2/tau) * [ (1/min_d2) * sum_j w_max_ij (P_i - P_j) - (max_d2/min_d2^2) * sum_j w_min_ij (P_i - P_j) ]
    for i in range(N):
        sum_max = np.zeros(D)
        sum_min = np.zeros(D)
        for j in range(N):
            if i != j:
                diff = P[i] - P[j]
                sum_max += w_max_full[i, j] * diff
                sum_min += w_min_full[i, j] * diff
        
        grad[i] = (2.0 / tau) * ((1.0 / min_d2) * sum_max - (max_d2 / (min_d2 ** 2)) * sum_min)
    
    return grad.ravel()

def _construct_initial_grid():
    """Create initial grid of N points."""
    side = int(np.ceil(N ** (1 / D)))
    import itertools
    grid = list(itertools.product(range(side), repeat=D))[:N]
    return np.array(grid, dtype=float)

def _fibonacci_sphere(n, radius=1.0):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)

def construct():
    start_time = time.time()
    best_P = None
    best_score = np.inf
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Grid
    inits.append(_construct_initial_grid())
    
    # 2. Fibonacci sphere
    inits.append(_fibonacci_sphere(N, radius=1.0))
    
    # 3. Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.standard_normal((N, D)))
    
    # 4. Regular simplex-like in 3D (project 14 points)
    # Use a regular polygon in xy plane plus z variations
    angles = 2 * np.pi * np.arange(N) / N
    for r in [0.5, 1.0, 1.5]:
        pts = np.column_stack([
            r * np.cos(angles),
            r * np.sin(angles),
            np.sin(angles * 2) * 0.3
        ])
        inits.append(pts)
    
    tau_schedule = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    
    for P0 in inits:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        P_curr = P0.copy()
        score_curr = _score(P_curr)
        
        # Anneal through tau values
        for tau in tau_schedule:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            P_flat = P_curr.ravel()
            
            try:
                result = minimize(
                    lambda x: _objective(x, tau=tau),
                    P_flat,
                    jac=lambda x: _objective_grad(x, tau=tau),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < _score(P_curr):
                    P_new = result.x.reshape(N, D)
                    score_new = _score(P_new)
                    
                    if score_new < score_curr:
                        P_curr = P_new
                        score_curr = score_new
                        
                        if score_curr < best_score:
                            best_score = score_curr
                            best_P = P_curr.copy()
            except Exception:
                pass
        
        # Final polish with high-precision
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        try:
            P_flat = P_curr.ravel()
            result = minimize(
                lambda x: _objective(x, tau=10000.0),
                P_flat,
                jac=lambda x: _objective_grad(x, tau=10000.0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < _score(P_curr):
                P_new = result.x.reshape(N, D)
                score_new = _score(P_new)
                
                if score_new < score_curr:
                    P_curr = P_new
                    score_curr = score_new
                    
                    if score_curr < best_score:
                        best_score = score_curr
                        best_P = P_curr.copy()
        except Exception:
            pass
    
    # Return best found
    if best_P is None:
        best_P = _construct_initial_grid()
    
    return best_P
# EVOLVE-BLOCK-END
