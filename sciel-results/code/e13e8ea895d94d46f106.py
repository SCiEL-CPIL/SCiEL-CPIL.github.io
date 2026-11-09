import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 620
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid(centers, sides, tol=1e-7):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True
    
    def score(centers, sides):
        if not is_valid(centers, sides):
            return -np.inf
        return np.sum(sides)
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if centers is None or sides is None:
            return
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return
        if np.any(sides <= 0):
            return
        s = score(centers, sides)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # 1. Construct the known 3.0 solution
    c_known = np.zeros((n, 2))
    s_known = np.zeros(n)
    # 2 large
    c_known[0] = [0.25, 0.75]
    s_known[0] = 0.5
    c_known[1] = [0.75, 0.75]
    s_known[1] = 0.5
    # 8 small
    for r in range(2):
        for c in range(4):
            idx = 2 + r*4 + c
            c_known[idx, 0] = 0.125 + c*0.25
            c_known[idx, 1] = 0.125 + r*0.25
            s_known[idx] = 0.25
    update_best(c_known, s_known)
    
    # 2. Smooth soft-constraint objective
    # We minimize -sum(sides) + penalty
    # Penalty uses smooth max (softplus-like) for overlap
    beta = 200.0  # temperature for soft max
    
    def soft_max(x, beta=200.0):
        """Smooth approximation of max(0, x) using softplus"""
        # softplus(x) = log(1 + exp(x)) approximates max(0, x)
        # For large x, softplus(x) ≈ x
        # For x near 0, it's smooth
        # We want max(0, overlap_amount)
        # Use: softplus(x * beta) / beta
        # This approximates max(0, x) with smoothness ~ 1/beta
        x = np.clip(x, -50, 50)
        return np.log1p(np.exp(x * beta)) / beta
    
    def objective(params):
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]
        
        # Sides must be positive
        if np.any(sides <= 0):
            return 1e6
            
        obj = -np.sum(sides)
        
        # Boundary penalty using smooth max
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Left: violation if cx - s/2 < 0, i.e., s/2 - cx > 0
            v = s/2 - cx
            obj += 50 * soft_max(v, beta)
            # Right: violation if cx + s/2 > 1, i.e., cx + s/2 - 1 > 0
            v = cx + s/2 - 1
            obj += 50 * soft_max(v, beta)
            # Bottom
            v = s/2 - cy
            obj += 50 * soft_max(v, beta)
            # Top
            v = cy + s/2 - 1
            obj += 50 * soft_max(v, beta)
            
        # Overlap penalty using smooth max
        tol = 1e-7
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t <= 0:
                    continue
                # Overlap in x: t - dx if positive
                ox = t - dx
                oy = t - dy
                # Both must be positive for overlap
                # Use product of soft_max to get smooth overlap volume
                if ox > -0.01 and oy > -0.01:
                    obj += 50 * soft_max(ox, beta) * soft_max(oy, beta)
                    
        return obj
    
    # 3. Run optimization from multiple starts
    # Start 1: Known solution
    starts = [np.concatenate([c_known.flatten(), s_known])]
    
    # Start 2: Random valid-ish packing
    c_rand = np.zeros((n, 2))
    s_rand = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c_rand[i, 0] = (c + 0.5) * 0.2
        c_rand[i, 1] = (r + 0.5) * 0.5
        s_rand[i] = 0.2
    starts.append(np.concatenate([c_rand.flatten(), s_rand]))
    
    # Start 3: Perturbed known solution
    np.random.seed(42)
    c_pert = c_known + np.random.normal(0, 0.01, (n, 2))
    s_pert = s_known + np.random.normal(0, 0.01, n)
    s_pert = np.clip(s_pert, 0.1, 0.6)
    starts.append(np.concatenate([c_pert.flatten(), s_pert]))
    
    # Start 4: Larger squares attempt
    c_large = np.zeros((n, 2))
    s_large = np.zeros(n)
    # Try 2x5 grid with slightly larger squares
    for i in range(n):
        r = i // 5
        c = i % 5
        c_large[i, 0] = (c + 0.5) * 0.2
        c_large[i, 1] = (r + 0.5) * 0.5
        s_large[i] = 0.22
    starts.append(np.concatenate([c_large.flatten(), s_large]))
    
    # Start 5: Mixed sizes
    c_mixed = np.zeros((n, 2))
    s_mixed = np.zeros(n)
    # 4 large in corners, 6 medium in middle
    c_mixed[0] = [0.2, 0.2]
    s_mixed[0] = 0.4
    c_mixed[1] = [0.8, 0.2]
    s_mixed[1] = 0.4
    c_mixed[2] = [0.2, 0.8]
    s_mixed[2] = 0.4
    c_mixed[3] = [0.8, 0.8]
    s_mixed[3] = 0.4
    for i in range(4, 10):
        r = (i - 4) // 3
        c = (i - 4) % 3
        c_mixed[i, 0] = 0.5 + (c - 1) * 0.15
        c_mixed[i, 1] = 0.5 + (r - 0.5) * 0.15
        s_mixed[i] = 0.14
    starts.append(np.concatenate([c_mixed.flatten(), s_mixed]))
    
    for init in starts:
        if elapsed() > 580:
            break
        try:
            res = minimize(objective, init, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8})
            if res.fun < 0:
                c_opt = res.x[:2*n].reshape(n, 2)
                s_opt = np.maximum(res.x[2*n:], 1e-6)
                update_best(c_opt, s_opt)
        except Exception:
            pass
            
    # 4. High-precision polish pass on the best solution found
    if best_centers is not None and elapsed() < 590:
        try:
            init_polish = np.concatenate([best_centers.flatten(), best_sides])
            # Use higher beta for sharper constraints
            beta_polish = 500.0
            def objective_polish(params):
                centers = params[:2*n].reshape(n, 2)
                sides = params[2*n:]
                if np.any(sides <= 0):
                    return 1e6
                obj = -np.sum(sides)
                for i in range(n):
                    cx, cy = centers[i]
                    s = sides[i]
                    v = s/2 - cx
                    obj += 50 * soft_max(v, beta_polish)
                    v = cx + s/2 - 1
                    obj += 50 * soft_max(v, beta_polish)
                    v = s/2 - cy
                    obj += 50 * soft_max(v, beta_polish)
                    v = cy + s/2 - 1
                    obj += 50 * soft_max(v, beta_polish)
                tol = 1e-7
                for i in range(n):
                    for j in range(i+1, n):
                        dx = abs(centers[i, 0] - centers[j, 0])
                        dy = abs(centers[i, 1] - centers[j, 1])
                        t = (sides[i] + sides[j]) / 2.0 - tol
                        if t <= 0:
                            continue
                        ox = t - dx
                        oy = t - dy
                        if ox > -0.01 and oy > -0.01:
                            obj += 50 * soft_max(ox, beta_polish) * soft_max(oy, beta_polish)
                return obj
            
            res_polish = minimize(objective_polish, init_polish, method='L-BFGS-B',
                                  options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-10})
            if res_polish.fun < 0:
                c_polish = res_polish.x[:2*n].reshape(n, 2)
                s_polish = np.maximum(res_polish.x[2*n:], 1e-6)
                update_best(c_polish, s_polish)
        except Exception:
            pass
            
    if best_centers is None:
        best_centers = c_known
        best_sides = s_known
        
    return best_centers, best_sides
