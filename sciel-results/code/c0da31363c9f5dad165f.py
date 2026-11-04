"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import Delaunay
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n):
    """
    Compute maximum radii for given positions via iterative relaxation.
    """
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
    
    for _ in range(80):
        changed = False
        for i in range(n):
            max_r = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_max_j = dist - radii[j]
                if r_max_j < max_r:
                    max_r = r_max_j
            if max_r < 0:
                max_r = 0.0
            if abs(max_r - radii[i]) > 1e-13:
                radii[i] = max_r
                changed = True
        if not changed:
            break
    return radii

def _sum_radii(pos, n):
    """Compute sum of max radii for given positions."""
    radii = _compute_radii(pos, n)
    return np.sum(radii)

def _obj_pos(p_flat, n):
    """Objective: minimize -sum(radii)."""
    p = p_flat.reshape(n, 2)
    r = _compute_radii(p, n)
    return -np.sum(r)

def _grad_pos(p_flat, n, eps=1e-7):
    """Numerical gradient of -sum(radii) w.r.t. positions."""
    p = p_flat.reshape(n, 2)
    base = -_sum_radii(p, n)
    grad = np.zeros(2*n)
    for k in range(2*n):
        pf = p_flat.copy()
        pf[k] += eps
        val = _obj_pos(pf, n)
        grad[k] = (val - base) / eps
    return grad

def _gradient_descent(pos, n, deadline, max_iter=500, lr=0.005, tol=1e-10):
    """
    Custom gradient descent on positions with line search.
    """
    current_pos = pos.copy()
    current_val = _obj_pos(current_pos.flatten(), n)
    step = lr
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        grad = _grad_pos(current_pos.flatten(), n)
        
        # If gradient is tiny, stop
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search
        best_val = current_val
        best_pos = current_pos.copy()
        
        for alpha in [step, step*0.5, step*0.25, step*0.1, step*0.05]:
            new_pos = current_pos - alpha * grad.reshape(n, 2)
            # Clip to [0,1]
            new_pos = np.clip(new_pos, 0.0, 1.0)
            new_val = _obj_pos(new_pos.flatten(), n)
            if new_val < best_val - 1e-12:
                best_val = new_val
                best_pos = new_pos
                break
        
        if best_val < current_val - 1e-12:
            current_pos = best_pos
            current_val = best_val
        else:
            step *= 0.7
            if step < 1e-8:
                break
                
    return current_pos

def _l_bfgs_optimize(pos, n, deadline):
    """
    Use L-BFGS-B to optimize positions.
    """
    x0 = pos.flatten()
    bounds = [(0.0, 1.0)] * (2*n)
    
    def obj(x):
        if time.time() > deadline:
            return 1e6
        return _obj_pos(x, n)
    
    res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 300, 'maxfun': 2000, 'gtol': 1e-6})
    return res.x.reshape(n, 2)

def _high_precision_polish(pos, n, deadline):
    """
    Final polish: alternate between L-BFGS-B and manual gradient descent
    with very tight tolerances.
    """
    current_pos = pos.copy()
    current_val = _obj_pos(current_pos.flatten(), n)
    
    for round in range(3):
        if time.time() > deadline - 5:
            break
        
        # L-BFGS-B pass
        new_pos = _l_bfgs_optimize(current_pos, n, deadline - 5)
        new_val = _obj_pos(new_pos.flatten(), n)
        if new_val < current_val:
            current_pos = new_pos
            current_val = new_val
        
        # Gradient descent pass with small steps
        new_pos = _gradient_descent(current_pos, n, deadline - 5, max_iter=100, lr=0.001)
        new_val = _obj_pos(new_pos.flatten(), n)
        if new_val < current_val:
            current_pos = new_pos
            current_val = new_val
            
    return current_pos

def _get_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Grid 5x5 + 1
    pos = []
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            pos.append([x, y])
    pos.append([0.5, 0.5])
    pos = np.array(pos[:N])
    configs.append(pos)
    
    # 2. Hexagonal packing
    pos = []
    dx = 0.18
    dy = 0.155
    for r in range(8):
        for c in range(7):
            if len(pos) >= N:
                break
            x = (c + 0.5) * dx
            y = (r + 0.5) * dy
            if r % 2 == 1:
                x += dx / 2.0
            if 0 <= x <= 1 and 0 <= y <= 1:
                pos.append([x, y])
    pos = np.array(pos[:N])
    configs.append(pos)
    
    # 3. Random inits with rejection sampling
    for seed in range(10):
        rng = np.random.RandomState(seed)
        pos = []
        min_dist = 0.12
        attempts = 0
        while len(pos) < N and attempts < 5000:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            ok = True
            for px, py in pos:
                if (x - px)**2 + (y - py)**2 < min_dist**2:
                    ok = False
                    break
            if ok:
                pos.append([x, y])
            attempts += 1
        while len(pos) < N:
            pos.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
        pos = np.array(pos)
        configs.append(pos)
        
    return configs

def _validate_result(result):
    """Check if result is valid (disjoint circles in unit square)."""
    n = result.shape[0]
    for i in range(n):
        x, y, r = result[i]
        if x - r < -1e-9 or y - r < -1e-9:
            return False
        if x + r > 1.0 + 1e-9 or y + r > 1.0 + 1e-9:
            return False
        if r < 0:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = result[i,0] - result[j,0]
            dy = result[i,1] - result[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < result[i,2] + result[j,2] - 1e-9:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    configs = _get_initial_configs()
    
    for idx, pos in enumerate(configs):
        if time.time() > deadline - 15:
            break
        
        try:
            # Stage 1: L-BFGS-B optimization
            opt_pos = _l_bfgs_optimize(pos, N, deadline - 10)
            score = _sum_radii(opt_pos, N)
            
            # Stage 2: Gradient descent polish
            opt_pos = _gradient_descent(opt_pos, N, deadline - 8, max_iter=200, lr=0.003)
            score = _sum_radii(opt_pos, N)
            
            # Stage 3: High precision polish
            opt_pos = _high_precision_polish(opt_pos, N, deadline - 3)
            score = _sum_radii(opt_pos, N)
            
            if score > best_score:
                best_score = score
                radii = _compute_radii(opt_pos, N)
                result = np.zeros((N, 3))
                result[:, :2] = opt_pos
                result[:, 2] = radii
                best_result = result
                
        except Exception:
            continue
            
    if best_result is None:
        # Fallback: grid
        result = np.zeros((N, 3))
        for i in range(N):
            result[i, 0] = (i % 5 + 0.5) / 5.0
            result[i, 1] = (i // 5 + 0.5) / 5.0
            result[i, 2] = 0.09
        return result
        
    return best_result

# EVOLVE-BLOCK-END
