import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
DEADLINE = 140.0  # Safety margin before 150s kill

def _is_valid(x, y, r, w, h, tol=1e-7):
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    # Check overlaps
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _force_relax(x, y, r, w, h, max_iter=200):
    """
    Simple force-directed relaxation to resolve overlaps and push circles inside.
    Returns relaxed positions and radii.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # We want to maximize radii, so we treat them as fixed during relaxation
    # and just move centers to resolve overlaps.
    # Actually, to be more robust, we can allow slight radius adjustment or just move centers.
    # Let's just move centers.
    
    dt = 0.01
    for _ in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 10
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 10
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist == 0:
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Repulsive force
                    f = overlap * 5.0
                    fx[i] += (dx / dist) * f
                    fy[i] += (dy / dist) * f
                    fx[j] -= (dx / dist) * f
                    fy[j] -= (dy / dist) * f
        
        # Update positions
        x += fx * dt
        y += fy * dt
        
        # Clamp to bounds
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
            
    return x, y

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    # Aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.1:
            continue
        
        max_restarts = 3 # Reduced to allow more time per restart
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Initialization
            cols = int(np.sqrt(N * w / h))
            if cols < 2: cols = 2
            rows = int(np.ceil(N / cols))
            
            while cols * rows < N:
                if cols <= rows:
                    cols += 1
                else:
                    rows += 1
            
            positions = np.zeros((N, 2))
            radii = np.zeros(N)
            
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= N:
                        break
                    x = (c_idx + 0.5) * w / cols
                    y = (r_idx + 0.5) * h / rows
                    
                    # Jitter for diversity
                    if restart > 0:
                        jitter = 0.15 * min(w/cols, h/rows)
                        x += np.random.uniform(-jitter, jitter)
                        y += np.random.uniform(-jitter, jitter)
                    
                    x = np.clip(x, 0.01, w - 0.01)
                    y = np.clip(y, 0.01, h - 0.01)
                    
                    positions[idx] = [x, y]
                    idx += 1
            
            # Initial radii
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            
            r0 = 0.4 * min_dist
            radii = np.full(N, r0)
            
            # Force relaxation to get a valid starting point
            x_rel, y_rel = _force_relax(positions[:,0], positions[:,1], radii, w, h)
            
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = x_rel[i]
                x0[3*i+1] = y_rel[i]
                x0[3*i+2] = radii[i]
            
            # Optimization
            def objective(vars_):
                r = vars_[2::3]
                return -np.sum(r)
            
            def cons_inside(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                return np.concatenate([
                    x - r,
                    w - x - r,
                    y - r,
                    h - y - r
                ])
            
            def cons_non_overlap(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                # Compute all pairwise distances
                # i < j
                n = len(x)
                cons = np.zeros((n*(n-1)//2,))
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        cons[idx] = dist - r[i] - r[j]
                        idx += 1
                return cons
            
            def cons_positive_r(vars_):
                r = vars_[2::3]
                return r
            
            constraints = [
                {'type': 'ineq', 'fun': cons_inside},
                {'type': 'ineq', 'fun': cons_non_overlap},
                {'type': 'ineq', 'fun': cons_positive_r}
            ]
            
            try:
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-9}
                )
                
                x_res = result.x[0::3]
                y_res = result.x[1::3]
                r_res = result.x[2::3]
                
                # Check validity
                if _is_valid(x_res, y_res, r_res, w, h):
                    score = np.sum(r_res)
                    
                    # Polish phase with tighter tolerance
                    if time.time() - start_time < DEADLINE - 15:
                        try:
                            polished = minimize(
                                objective, result.x, method='SLSQP',
                                constraints=constraints,
                                options={'maxiter': 100, 'ftol': 1e-12}
                            )
                            x_pol = polished.x[0::3]
                            y_pol = polished.x[1::3]
                            r_pol = polished.x[2::3]
                            if _is_valid(x_pol, y_pol, r_pol, w, h):
                                x_res, y_res, r_res = x_pol, y_pol, r_pol
                                score = np.sum(r_pol)
                        except:
                            pass
                    
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_res, y_res, r_res)).copy()
                        
            except Exception:
                continue

    # Fallback
    if best_result is None or best_score < 0:
        w = 1.0
        h = 1.0
        r = 0.095
        positions = []
        radii = []
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                x = (i + 0.5) * w / 5
                y = (j + 0.5) * h / 5
                positions.append([x, y])
                radii.append(r)
                count += 1
        best_result = np.column_stack((np.array(positions), np.array(radii)))
        best_score = np.sum(radii)
    
    return best_result
