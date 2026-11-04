import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 140  # seconds, safety margin

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = -1
    
    # Try different grid configurations
    configs = [(7, 3), (3, 7), (5, 5), (6, 4), (4, 6)]
    
    for cols, rows in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Aspect ratio matching grid
        W = 2 * cols / (cols + rows)
        H = 2 * rows / (cols + rows)
        
        # Grid positions
        positions = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x = (i + 0.5) * W / cols
                y = (j + 0.5) * H / rows
                positions.append([x, y])
                count += 1
            if count >= N:
                break
        
        positions = np.array(positions)
        
        # Initial radii: slightly less than grid spacing
        r_init = np.full(N, 0.5 * min(W / cols, H / rows) * 0.85)
        
        result = _optimize(positions, r_init, W, H, start_time, DEADLINE)
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
    
    if best_result is None:
        # Fallback: 5x5 grid
        W, H = 1.0, 1.0
        positions = []
        for i in range(5):
            for j in range(5):
                positions.append([(i+0.5)*0.2, (j+0.5)*0.2])
        positions = np.array(positions[:N])
        r = np.full(N, 0.099)
        best_result = np.column_stack([positions[:,0], positions[:,1], r])
    
    return best_result

def _optimize(init_pos, init_r, W0, H0, start_time, deadline):
    # Variables: x0,y0,r0, ..., x20,y20,r20, W, H (65 vars)
    
    def objective(v):
        pos = v[:2*N].reshape(N, 2)
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        penalty = 0.0
        
        # Perimeter: W + H = 2
        penalty += 1000 * (W + H - 2)**2
        
        # Wall constraints (smooth penalty)
        for i in range(N):
            x, y = pos[i]
            ri = r[i]
            # x >= ri
            if x < ri:
                penalty += 1000 * (ri - x)**2
            # x <= W - ri
            if x > W - ri:
                penalty += 1000 * (x - (W - ri))**2
            # y >= ri
            if y < ri:
                penalty += 1000 * (ri - y)**2
            # y <= H - ri
            if y > H - ri:
                penalty += 1000 * (y - (H - ri))**2
            # ri >= 0
            if ri < 0:
                penalty += 1000 * ri**2
        
        # Non-overlap (smooth penalty)
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_d = r[i] + r[j]
                if dist < min_d:
                    penalty += 1000 * (min_d - dist)**2
        
        return -np.sum(r) + penalty
    
    v0 = np.concatenate([init_pos.ravel(), init_r, [W0, H0]])
    
    bounds = []
    for _ in range(N):
        bounds.append((0.001, 1.5))  # x
        bounds.append((0.001, 1.5))  # y
        bounds.append((0.001, 0.5))  # r
    bounds.append((0.1, 1.9))  # W
    bounds.append((0.1, 1.9))  # H
    
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() - start_time > deadline - 10:
            return None
        
        v = res.x
        pos = v[:2*N].reshape(N, 2)
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        # Polish: clip and validate
        r = np.maximum(r, 0.001)
        
        if not _is_valid(pos, r, W, H):
            # Try to repair by shrinking radii
            for scale in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
                if _is_valid(pos, r * scale, W, H):
                    r = r * scale
                    break
            else:
                return None
        
        # High-precision polish with progressive smooth-max eps
        result = _high_precision_polish(pos, r, W, H, start_time, deadline)
        if result is not None:
            pos, r, W, H = result
        
        return np.column_stack([pos[:,0], pos[:,1], r])
    
    except Exception:
        return None

def _high_precision_polish(pos, r, W, H, start_time, deadline):
    """
    Progressive smooth-max eps polish: start with coarse eps, tighten over passes.
    This lets L-BFGS-B resolve near-tangent contacts more precisely.
    """
    if time.time() - start_time > deadline - 5:
        return None
    
    n = N
    # Variable vector: [x0,y0,r0, ..., x20,y20,r20, W, H]
    v = np.concatenate([pos.ravel(), r, [W, H]])
    
    # Progressive eps values for smooth-max
    eps_schedule = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    penalty_weights = [100.0, 500.0, 2000.0, 10000.0, 50000.0]
    
    for eps, weight in zip(eps_schedule, penalty_weights):
        if time.time() - start_time > deadline - 3:
            break
        
        def obj_grad(v, eps=eps, weight=weight):
            pos = v[:2*n].reshape(n, 2)
            r = v[2*n:3*n]
            W = v[3*n]
            H = v[3*n+1]
            
            x = pos[:, 0]
            y = pos[:, 1]
            
            obj = -np.sum(r)
            grad = np.zeros_like(v)
            grad[2*n:3*n] = -1.0  # d(-sum(r))/dr
            
            # Perimeter penalty
            perim_viol = W + H - 2.0
            obj += weight * perim_viol**2
            grad[3*n] += 2.0 * weight * perim_viol
            grad[3*n+1] += 2.0 * weight * perim_viol
            
            # Boundary penalties using smooth max
            def smooth_max(val, eps):
                return 0.5 * (val + np.sqrt(val**2 + eps**2))
            
            def smooth_max_grad(val, eps):
                return 0.5 * (1.0 + val / np.sqrt(val**2 + eps**2))
            
            # x >= r: violation = r - x
            viol = r - x
            sm = smooth_max(viol, eps)
            smg = smooth_max_grad(viol, eps)
            obj += weight * np.sum(sm**2)
            grad[0::3] += -2.0 * weight * sm * smg
            grad[2::3] += 2.0 * weight * sm * smg
            
            # y >= r
            viol = r - y
            sm = smooth_max(viol, eps)
            smg = smooth_max_grad(viol, eps)
            obj += weight * np.sum(sm**2)
            grad[1::3] += -2.0 * weight * sm * smg
            grad[2::3] += 2.0 * weight * sm * smg
            
            # x + r <= W: violation = x + r - W
            viol = x + r - W
            sm = smooth_max(viol, eps)
            smg = smooth_max_grad(viol, eps)
            obj += weight * np.sum(sm**2)
            grad[0::3] += 2.0 * weight * sm * smg
            grad[2::3] += 2.0 * weight * sm * smg
            
            # y + r <= H
            viol = y + r - H
            sm = smooth_max(viol, eps)
            smg = smooth_max_grad(viol, eps)
            obj += weight * np.sum(sm**2)
            grad[1::3] += 2.0 * weight * sm * smg
            grad[2::3] += 2.0 * weight * sm * smg
            
            # Overlap penalties: vectorized
            dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
            dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
            dist_mat = np.sqrt(dx_mat**2 + dy_mat**2 + 1e-15)
            np.fill_diagonal(dist_mat, np.inf)
            
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            overlap_viol = r_sum - dist_mat
            
            # Only upper triangle
            iu, ju = np.triu_indices(n, k=1)
            ov_viol = overlap_viol[iu, ju]
            dist_ij = dist_mat[iu, ju]
            dx_ij = dx_mat[iu, ju]
            dy_ij = dy_mat[iu, ju]
            
            sm_vec = smooth_max(ov_viol, eps)
            smg_vec = smooth_max_grad(ov_viol, eps)
            
            obj += weight * np.sum(sm_vec**2)
            
            factor = 2.0 * weight * sm_vec * smg_vec
            # Gradient w.r.t. x_i: factor * (x_i - x_j)/dist
            gx_i = factor * dx_ij / dist_ij
            gy_i = factor * dy_ij / dist_ij
            gx_j = -factor * dx_ij / dist_ij
            gy_j = -factor * dy_ij / dist_ij
            
            np.add.at(grad, iu, gx_i)
            np.add.at(grad, n + iu, gy_i)
            np.add.at(grad, 2*n + iu, factor)
            np.add.at(grad, ju, gx_j)
            np.add.at(grad, n + ju, gy_j)
            np.add.at(grad, 2*n + ju, factor)
            
            return obj, grad
        
        # Bounds for this polish phase
        bounds = []
        for _ in range(n):
            bounds.append((1e-6, W - 1e-6))
            bounds.append((1e-6, H - 1e-6))
            bounds.append((1e-6, min(W, H)/2))
        bounds.append((0.1, 1.9))
        bounds.append((0.1, 1.9))
        
        try:
            res = minimize(obj_grad, v, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
            v = res.x
        except Exception:
            break
    
    # Extract final result
    pos = v[:2*n].reshape(n, 2)
    r = v[2*n:3*n]
    W = v[3*n]
    H = v[3*n+1]
    
    r = np.maximum(r, 0.001)
    
    if not _is_valid(pos, r, W, H):
        for scale in [0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8]:
            if _is_valid(pos, r * scale, W, H):
                r = r * scale
                break
        else:
            return None
    
    return pos, r, W, H

def _is_valid(pos, r, W, H, tol=1e-5):
    if abs(W + H - 2) > 0.01:
        return False
    if W < 0.01 or H < 0.01:
        return False
    for i in range(N):
        x, y = pos[i]
        ri = r[i]
        if ri < -tol:
            return False
        if x < ri - tol or x > W - ri + tol:
            return False
        if y < ri - tol or y > H - ri + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True
