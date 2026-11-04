"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 21

# EVOLVE-BLOCK-START

def _pairwise_distances(x):
    """x is (2N, 1) or (2N,). Returns (N, N) matrix of distances between circle centers."""
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    return dist, radii

def _feasibility_penalty(x, w=100.0):
    """Penalty for constraint violations (circle-circle overlap, circle-wall)."""
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Rectangle dimensions: perimeter = 2*(W+H) = 4, so W+H = 2
    # We fix W and H via a parameter, or we can let the optimizer choose W,H with W+H=2
    # Actually, let's parameterize: W = w, H = 2 - w, where w in (0, 2)
    # But to keep it simpler, let's fix the rectangle aspect ratio or optimize it.
    # Let's include W as an additional variable.
    
    # For now, let's fix W=1, H=1 (square) as a starting point, but we should optimize W too.
    # Let's add W as variable index 2N.
    
    W = x[2*N]
    H = 2.0 - W
    
    # Wall constraints: circles must be inside rectangle
    # x_i >= r_i, x_i <= W - r_i, y_i >= r_i, y_i <= H - r_i
    wall_viol = 0.0
    for i in range(N):
        cx, cy = centers[i]
        r = radii[i]
        if cx < r:
            wall_viol += (r - cx)**2
        if cx > W - r:
            wall_viol += (cx - (W - r))**2
        if cy < r:
            wall_viol += (r - cy)**2
        if cy > H - r:
            wall_viol += (cy - (H - r))**2
    
    # Circle-circle constraints: dist(i,j) >= r_i + r_j
    cc_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            min_d = radii[i] + radii[j]
            if d < min_d:
                cc_viol += (min_d - d)**2
    
    return w * (wall_viol + cc_viol)

def _objective(x):
    """Minimize negative sum of radii + penalty."""
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    radii = x[2*N:2*N+N]
    penalty = _feasibility_penalty(x, w=100.0)
    return -np.sum(radii) + penalty

def _objective_grad(x):
    """Numerical gradient."""
    eps = 1e-7
    g = np.zeros_like(x)
    f0 = _objective(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_objective(xp) - f0) / eps
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Strategy: Use differential evolution for global search, then L-BFGS-B for local polish.
    # Variables: x_1..x_N (cx), y_1..y_N (cy), r_1..r_N, W
    # Total: 2N + N + 1 = 3N + 1 = 64 variables
    
    # Bounds:
    # cx, cy in [0, 2] (loose)
    # r in [0.001, 0.5]
    # W in [0.1, 1.9]
    
    def run_local_polish(x0, time_budget):
        """Run L-BFGS-B local optimization."""
        x = x0.copy()
        try:
            res = minimize(_objective, x, method='L-BFGS-B', 
                          jac=_objective_grad,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            if res.success or res.fun < best_score_global[0]:
                # Extract result
                x_opt = res.x
                W = x_opt[2*N]
                H = 2.0 - W
                centers = x_opt[:2*N].reshape(N, 2)
                radii = x_opt[2*N:2*N+N]
                
                # Check feasibility
                feasible = True
                for i in range(N):
                    cx, cy = centers[i]
                    r = radii[i]
                    if cx < r - 1e-6 or cx > W - r + 1e-6 or cy < r - 1e-6 or cy > H - r + 1e-6:
                        feasible = False
                        break
                if feasible:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                            if d < radii[i] + radii[j] - 1e-6:
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible:
                    score = np.sum(radii)
                    if score > best_score_global[0]:
                        best_score_global[0] = score
                        best_result_global[0] = np.column_stack([centers, radii])
        except Exception:
            pass
    
    best_score_global = [-np.inf]
    best_result_global = [None]
    
    # Initial guess: grid-like placement
    side = int(np.ceil(np.sqrt(N)))
    W_init = 1.0
    H_init = 1.0
    x0 = np.zeros(3*N + 1)
    for i in range(N):
        col = i % side
        row = i // side
        x0[i] = (col + 0.5) * W_init / side  # cx
        x0[N + i] = (row + 0.5) * H_init / side  # cy
        x0[2*N + i] = 0.5 * W_init / side - 0.01  # r
    x0[2*N + N] = W_init  # W
    
    # Run differential evolution
    bounds = []
    for i in range(2*N):
        bounds.append((0.0, 2.0))
    for i in range(N):
        bounds.append((0.001, 0.5))
    bounds.append((0.1, 1.9))
    
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(_objective, bounds, seed=42, maxiter=50, 
                                               popsize=15, tol=1e-6, polish=False,
                                               update_interval=1, init='sobol')
            if time.time() < deadline - 10:
                run_local_polish(de_result.x, 10)
        except Exception:
            pass
    
    # Try multiple random restarts with local polish
    np.random.seed(123)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        W_r = 0.5 + np.random.rand() * 1.0  # W in [0.5, 1.5]
        H_r = 2.0 - W_r
        x_r = np.zeros(3*N + 1)
        for i in range(N):
            x_r[i] = np.random.rand() * W_r
            x_r[N + i] = np.random.rand() * H_r
            x_r[2*N + i] = 0.05 + np.random.rand() * 0.15
        x_r[2*N + N] = W_r
        
        if time.time() < deadline - 5:
            run_local_polish(x_r, 5)
    
    # Final polish on best result
    if best_result_global[0] is not None and time.time() < deadline - 2:
        # Try to further improve
        x_best = np.concatenate([best_result_global[0][:, 0], best_result_global[0][:, 1], 
                                  best_result_global[0][:, 2], [1.0]])
        # We don't have W stored, let's just return what we have
        pass
    
    if best_result_global[0] is not None:
        return best_result_global[0]
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(N):
        col = i % side
        row = i // side
        cx = (col + 0.5) / side
        cy = (row + 0.5) / side
        r = 0.5 / side - 0.001
        cells.append([cx, cy, r])
    return np.array(cells)

# EVOLVE-BLOCK-END
