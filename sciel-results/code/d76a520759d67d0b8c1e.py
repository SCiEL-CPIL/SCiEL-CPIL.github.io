import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5]
    
    pair_indices = []
    for i in range(N):
        for j in range(i+1, N):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    i_idx = pair_indices[:, 0]
    j_idx = pair_indices[:, 1]
    num_pairs = len(pair_indices)
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.1:
            continue
            
        max_restarts = 4
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
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
                    
                    if restart > 0:
                        jitter = 0.1 * min(w/cols, h/rows)
                        x += np.random.uniform(-jitter, jitter)
                        y += np.random.uniform(-jitter, jitter)
                    
                    x = np.clip(x, 0.01, w - 0.01)
                    y = np.clip(y, 0.01, h - 0.01)
                    
                    positions[idx] = [x, y]
                    idx += 1
            
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            
            r0 = 0.4 * min_dist
            radii = np.full(N, r0)
            
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = radii[i]
            
            # RBF kernel width - controls smoothness of the distance field
            sigma = 0.15 * min(w, h)
            
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
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                dist = np.sqrt(dx*dx + dy*dy)
                # Smooth surrogate: use soft-min approximation
                # dist - r_i - r_j >= 0 becomes a smooth constraint
                # We use the exact distance but with a small smoothing parameter
                eps = 1e-4
                smooth_dist = np.sqrt(dist*dist + eps*eps)
                return smooth_dist - r[i_idx] - r[j_idx]
            
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
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                
                x_res = result.x[0::3]
                y_res = result.x[1::3]
                r_res = result.x[2::3]
                
                valid = True
                if np.any(r_res < 0):
                    valid = False
                else:
                    if np.any(x_res < r_res - 1e-6) or np.any(x_res > w - r_res + 1e-6):
                        valid = False
                    if np.any(y_res < r_res - 1e-6) or np.any(y_res > h - r_res + 1e-6):
                        valid = False
                    else:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                                if dist < r_res[i] + r_res[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    x_p = x_res.copy()
                    y_p = y_res.copy()
                    r_p = r_res.copy()
                    
                    # Enhanced polish: iterative radius growth with position relaxation
                    for polish_iter in range(20):
                        if time.time() - start_time > DEADLINE - 2:
                            break
                        
                        grew = False
                        for i in range(N):
                            slack = min(
                                x_p[i] - r_p[i],
                                w - x_p[i] - r_p[i],
                                y_p[i] - r_p[i],
                                h - y_p[i] - r_p[i]
                            )
                            for j in range(N):
                                if i != j:
                                    d = np.sqrt((x_p[i]-x_p[j])**2 + (y_p[i]-y_p[j])**2)
                                    slack = min(slack, d - r_p[i] - r_p[j])
                            
                            if slack > 1e-7:
                                growth = min(0.003, slack * 0.6)
                                r_p[i] += growth
                                grew = True
                        
                        if not grew:
                            break
                        
                        # Relax positions
                        for relax_iter in range(8):
                            fx = np.zeros(N)
                            fy = np.zeros(N)
                            
                            for i in range(N):
                                if x_p[i] - r_p[i] < 0:
                                    fx[i] += (x_p[i] - r_p[i]) * 25
                                if x_p[i] + r_p[i] > w:
                                    fx[i] -= (x_p[i] + r_p[i] - w) * 25
                                if y_p[i] - r_p[i] < 0:
                                    fy[i] += (y_p[i] - r_p[i]) * 25
                                if y_p[i] + r_p[i] > h:
                                    fy[i] -= (y_p[i] + r_p[i] - h) * 25
                            
                            for i in range(N):
                                for j in range(i + 1, N):
                                    dx = x_p[i] - x_p[j]
                                    dy = y_p[i] - y_p[j]
                                    d = np.sqrt(dx * dx + dy * dy)
                                    min_d = r_p[i] + r_p[j]
                                    if d < min_d and d > 1e-12:
                                        overlap = min_d - d
                                        fx[i] += (dx / d) * overlap * 12
                                        fy[i] += (dy / d) * overlap * 12
                                        fx[j] -= (dx / d) * overlap * 12
                                        fy[j] -= (dy / d) * overlap * 12
                            
                            dt = 0.0015
                            x_p += dt * fx
                            y_p += dt * fy
                            
                            for i in range(N):
                                x_p[i] = np.clip(x_p[i], r_p[i], w - r_p[i])
                                y_p[i] = np.clip(y_p[i], r_p[i], h - r_p[i])
                    
                    valid_p = True
                    if np.any(r_p < 0):
                        valid_p = False
                    else:
                        if np.any(x_p < r_p - 1e-6) or np.any(x_p > w - r_p + 1e-6):
                            valid_p = False
                        if np.any(y_p < r_p - 1e-6) or np.any(y_p > h - r_res + 1e-6):
                            valid_p = False
                        else:
                            for i in range(N):
                                for j in range(i+1, N):
                                    dist = np.sqrt((x_p[i]-x_p[j])**2 + (y_p[i]-y_p[j])**2)
                                    if dist < r_p[i] + r_p[j] - 1e-6:
                                        valid_p = False
                                        break
                                if not valid_p:
                                    break
                    
                    if valid_p:
                        score_p = np.sum(r_p)
                        if score_p > best_score:
                            best_score = score_p
                            best_result = np.column_stack((x_p, y_p, r_p)).copy()
                    else:
                        score = np.sum(r_res)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack((x_res, y_res, r_res)).copy()
                        
            except Exception:
                continue

    if best_result is None or best_score < 0:
        w = 1.0
        h = 1.0
        side = 5
        x = []
        y = []
        r = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x.append((j + 0.5) / side)
                y.append((i + 0.5) / side)
                r.append(0.5 / side - 1e-5)
                count += 1
            if count >= N:
                break
        return np.array([x, y, r]).T
        
    return best_result
