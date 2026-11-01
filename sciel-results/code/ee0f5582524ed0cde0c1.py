import numpy as np
import time
from scipy.spatial import KDTree
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    r_low = 0.0350
    r_high = 0.0370
    
    best_r = r_low
    best_positions = None
    
    def is_valid(pos, r):
        if np.any(pos[:, 0] < -0.5 + r - 1e-9) or np.any(pos[:, 0] > 0.5 - r + 1e-9):
            return False
        if np.any(pos[:, 1] < -0.5 + r - 1e-9) or np.any(pos[:, 1] > 0.5 - r + 1e-9):
            return False
        tree = KDTree(pos)
        pairs = tree.query_pairs(r=2*r + 1e-9)
        for i, j in pairs:
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < 2*r - 1e-9:
                return False
        return True

    def relax(pos, r, max_iters=500):
        pos = pos.copy()
        eta = 0.01
        decay = 0.999
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            tree = KDTree(pos)
            pairs = tree.query_pairs(r=2*r + 0.01)
            
            forces = np.zeros_like(pos)
            
            for i, j in pairs:
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 2*r:
                    overlap = 2*r - dist
                    if dist > 1e-10:
                        f = overlap * d / dist
                    else:
                        f = np.array([np.random.randn(), np.random.randn()])
                    forces[i] += f
                    forces[j] -= f
            
            mask_left = pos[:, 0] < -0.5 + r
            forces[mask_left, 0] += (-0.5 + r - pos[mask_left, 0]) * 10.0
            mask_right = pos[:, 0] > 0.5 - r
            forces[mask_right, 0] -= (pos[mask_right, 0] - (0.5 - r)) * 10.0
            mask_bottom = pos[:, 1] < -0.5 + r
            forces[mask_bottom, 1] += (-0.5 + r - pos[mask_bottom, 1]) * 10.0
            mask_top = pos[:, 1] > 0.5 - r
            forces[mask_top, 1] -= (pos[mask_top, 1] - (0.5 - r)) * 10.0
            
            pos += eta * forces
            pos[:, 0] = np.clip(pos[:, 0], -0.5 + r, 0.5 - r)
            pos[:, 1] = np.clip(pos[:, 1], -0.5 + r, 0.5 - r)
            
            eta *= decay
            
        return pos

    def polish(pos, r):
        def objective(x):
            p = x.reshape(-1, 2)
            pen = 0.0
            viol_x_left = np.clip(-0.5 + r - p[:, 0], 0, None)
            viol_x_right = np.clip(p[:, 0] - (0.5 - r), 0, None)
            viol_y_left = np.clip(-0.5 + r - p[:, 1], 0, None)
            viol_y_right = np.clip(p[:, 1] - (0.5 - r), 0, None)
            pen += np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + np.sum(viol_y_left**2) + np.sum(viol_y_right**2)
            
            tree = KDTree(p)
            pairs = tree.query_pairs(r=2*r + 0.01)
            pair_pen = 0.0
            for i, j in pairs:
                dist = np.linalg.norm(p[i] - p[j])
                if dist < 2*r:
                    pair_pen += (2*r - dist)**2
            pen += pair_pen
            return pen

        def gradient(x):
            p = x.reshape(-1, 2)
            grad = np.zeros_like(p)
            
            mask_left = p[:, 0] < -0.5 + r
            grad[mask_left, 0] -= 2 * (-0.5 + r - p[mask_left, 0])
            mask_right = p[:, 0] > 0.5 - r
            grad[mask_right, 0] += 2 * (p[mask_right, 0] - (0.5 - r))
            mask_bottom = p[:, 1] < -0.5 + r
            grad[mask_bottom, 1] -= 2 * (-0.5 + r - p[mask_bottom, 1])
            mask_top = p[:, 1] > 0.5 - r
            grad[mask_top, 1] += 2 * (p[mask_top, 1] - (0.5 - r))
            
            tree = KDTree(p)
            pairs = tree.query_pairs(r=2*r + 0.01)
            for i, j in pairs:
                d = p[i] - p[j]
                dist = np.linalg.norm(d)
                if dist < 2*r and dist > 1e-10:
                    factor = 2 * (2*r - dist) / dist
                    grad[i] -= factor * d
                    grad[j] += factor * d
            
            return grad.flatten()

        bounds = [(-0.5 + r, 0.5 - r)] * (2 * N)
        
        try:
            res = minimize(objective, pos.flatten(), jac=gradient, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < 1e-6:
                return res.x.reshape(-1, 2)
        except:
            pass
        return pos

    rng = np.random.default_rng(42)
    
    # Generate initializations
    initializations = []
    
    # 1. Grid with jitter
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    initializations.append(grid + rng.normal(0, 0.01, size=(N, 2)))
    
    # 2. Random uniform
    initializations.append(rng.uniform(-0.5, 0.5, size=(N, 2)))
    
    # 3. Grid with different aspect
    m_x = 14
    m_y = int(np.ceil(N / m_x))
    xs = (np.arange(m_x) + 0.5) / m_x - 0.5
    ys = (np.arange(m_y) + 0.5) / m_y - 0.5
    grid2 = np.array([(x, y) for y in ys for x in xs])[:N]
    initializations.append(grid2 + rng.normal(0, 0.01, size=(N, 2)))
    
    # 4. Random cluster
    centers = rng.uniform(-0.3, 0.3, size=(4, 2))
    points = []
    for c in centers:
        pts = rng.normal(c, 0.1, size=(50, 2))
        points.append(pts)
    initializations.append(np.vstack(points))
    
    # 5. Grid with larger jitter
    initializations.append(grid + rng.normal(0, 0.05, size=(N, 2)))

    for init_pos in initializations:
        if time.time() > deadline - 30:
            break
            
        r_l = r_low
        r_h = r_high
        best_r_local = r_l
        best_pos_local = init_pos
        
        for _ in range(15):
            if time.time() > deadline - 10:
                break
            r_mid = (r_l + r_h) / 2
            
            relaxed_pos = relax(init_pos, r_mid, max_iters=300)
            
            if is_valid(relaxed_pos, r_mid):
                best_r_local = r_mid
                best_pos_local = relaxed_pos
                r_l = r_mid
            else:
                r_h = r_mid
        
        if best_r_local > best_r:
            if time.time() < deadline - 5:
                polished_pos = polish(best_pos_local, best_r_local)
                if is_valid(polished_pos, best_r_local):
                    best_r = best_r_local
                    best_positions = polished_pos
                else:
                    best_r = best_r_local
                    best_positions = best_pos_local
            else:
                best_r = best_r_local
                best_positions = best_pos_local

    if best_positions is not None and time.time() < deadline - 2:
        polished_pos = polish(best_positions, best_r)
        if is_valid(polished_pos, best_r):
            best_positions = polished_pos
        
        for delta_r in [0.0001, 0.00005, 0.00001]:
            if time.time() > deadline - 2:
                break
            r_test = best_r + delta_r
            test_pos = polish(best_positions, r_test)
            if is_valid(test_pos, r_test):
                best_r = r_test
                best_positions = test_pos
            else:
                break

    return best_positions
