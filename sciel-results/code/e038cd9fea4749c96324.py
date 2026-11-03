import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _soft_min_obj(x, beta):
    """
    Smooth surrogate: -log-sum-exp of all constraints (pair distances and wall distances).
    We want to maximize the minimum, so minimize the negative smooth minimum.
    
    x: flat array of shape (N*3,)
    beta: temperature parameter for log-sum-exp approximation
    """
    pts = x.reshape(N, 3)
    
    # Wall constraints: distance from each point to each wall
    wall_dists = 0.5 - np.abs(pts)  # (N, 3)
    
    # Pair constraints: distances between all pairs / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_dists[iu]
    
    # Combine all constraint values
    all_vals = np.concatenate([wall_dists.ravel(), pair_vals])
    
    # Log-sum-exp for smooth minimum approximation
    # min(x) ≈ -(1/beta) * log(sum(exp(-beta * x)))
    # We want to maximize min, so minimize -min ≈ (1/beta) * log(sum(exp(-beta * x)))
    m = np.min(all_vals)
    if m < -10:
        return 100.0  # Penalize heavily if any constraint is very negative
    shifted = -beta * (all_vals - m)
    shifted = np.clip(shifted, -500, 500)
    lse = m + np.log(np.sum(np.exp(shifted))) / beta
    return -lse

def _soft_min_grad(x, beta):
    """Gradient of the soft-min objective."""
    pts = x.reshape(N, 3)
    
    # Wall gradients
    wall_dists = 0.5 - np.abs(pts)
    signs = np.sign(pts)
    signs[signs == 0] = 1.0
    
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e-10)
    pair_dists = dists / 2.0
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_dists[iu]
    
    all_vals = np.concatenate([wall_dists.ravel(), pair_vals])
    
    m = np.min(all_vals)
    if m < -10:
        return np.zeros_like(x)
    shifted = -beta * (all_vals - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # (N*3 + N*(N-1)/2,)
    
    grad = np.zeros_like(x)
    
    # Wall part: d/dx_{i,k} of -lse
    # wall_dists[i,k] = 0.5 - |x_{i,k}|
    # d(wall_dists[i,k])/dx_{i,k} = -sign(x_{i,k})
    # d(-lse)/d(wall_dists[i,k]) = weights[i*3+k] (from log-sum-exp derivative)
    # So d(-lse)/dx_{i,k} = weights[i*3+k] * (-sign(x_{i,k}))
    wall_grad = -weights[:N*3].reshape(N, 3) * signs
    grad += wall_grad.ravel()
    
    # Pair part
    # pair_vals[idx] = dists[i,j]/2 for (i,j) in iu
    # d(pair_vals)/dx_i = (x_i - x_j) / (2 * dists[i,j])  (for the i component)
    # d(pair_vals)/dx_j = (x_j - x_i) / (2 * dists[i,j])  (for the j component)
    
    for idx in range(len(pair_vals)):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[N*3 + idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        dir_ij = (pts[i] - pts[j]) / (2 * d)
        grad[i*3:(i+1)*3] += w * dir_ij
        grad[j*3:(j+1)*3] -= w * dir_ij
    
    return grad

def _generate_init(type_id):
    """Generate diverse initial configurations."""
    np.random.seed(type_id * 12345 + 678)
    
    if type_id % 5 == 0:
        # Simple grid
        coords = np.array(np.meshgrid(np.linspace(-0.45, 0.45, 4),
                                       np.linspace(-0.45, 0.45, 4),
                                       np.linspace(-0.45, 0.45, 4))).T.reshape(-1, 3)
        indices = np.random.choice(len(coords), N, replace=False)
        points = coords[indices]
        points += np.random.randn(N, 3) * 0.01
    
    elif type_id % 5 == 1:
        # BCC-like
        pts = []
        a = 0.5
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    pts.append([(i-1)*a, (j-1)*a, (k-1)*a])
                    pts.append([(i-0.5)*a, (j-0.5)*a, (k-0.5)*a])
        pts = np.array(pts)
        # Remove 4 outermost
        d_center = np.linalg.norm(pts, axis=1)
        remove = np.argsort(d_center)[-4:]
        keep = [i for i in range(len(pts)) if i not in remove]
        points = pts[keep[:N]]
        points += np.random.randn(N, 3) * 0.02
    
    elif type_id % 5 == 2:
        # FCC-like
        pts = []
        a = 0.5
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    pts.append([(i-1)*a, (j-1)*a, (k-1)*a])
                    # Face centers
                    pts.append([(i-1)*a, (j-0.5)*a, (k-0.5)*a])
                    pts.append([(i-0.5)*a, (j-1)*a, (k-0.5)*a])
                    pts.append([(i-0.5)*a, (j-0.5)*a, (k-1)*a])
        pts = np.array(pts)
        indices = np.random.choice(len(pts), N, replace=False)
        points = pts[indices]
        points += np.random.randn(N, 3) * 0.015
    
    elif type_id % 5 == 3:
        # Random with repulsion
        points = np.random.uniform(-0.4, 0.4, (N, 3))
        # Simple repulsion pass
        for _ in range(50):
            for i in range(N):
                for j in range(i+1, N):
                    d = points[i] - points[j]
                    dist = np.linalg.norm(d)
                    if dist < 0.15 and dist > 1e-10:
                        push = d / dist * 0.005
                        points[i] += push
                        points[j] -= push
        points = np.clip(points, -0.45, 0.45)
    
    else:
        # Hexagonal layers
        pts = []
        layer_spacing = 0.3
        for layer in range(5):
            z = -0.4 + layer * layer_spacing
            for i in range(4):
                for j in range(4):
                    x = -0.35 + i * 0.24
                    y = -0.35 + j * 0.24 + (0.12 if i % 2 else 0)
                    pts.append([x, y, z])
        pts = np.array(pts)
        indices = np.random.choice(len(pts), N, replace=False)
        points = pts[indices]
        points += np.random.randn(N, 3) * 0.01
    
    return np.clip(points, -0.48, 0.48)

def _polish_local(points, max_time):
    """Coordinate descent polish on actual min-distance objective."""
    start = time.time()
    pts = points.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    lr = 0.01
    decay = 0.99
    
    while time.time() - start < max_time and lr > 1e-6:
        # Find bottleneck
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists) / 2.0
        
        wall_dists = 0.5 - np.abs(pts)
        min_wall_val = np.min(wall_dists)
        
        improved = False
        
        if min_pair_val < min_wall_val:
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                d /= dist
                pts[i] += lr * d
                pts[j] -= lr * d
                improved = True
        else:
            flat_idx = np.argmin(wall_dists)
            i, k = flat_idx // 3, flat_idx % 3
            s = np.sign(pts[i, k])
            if s == 0:
                s = 1.0
            pts[i, k] += lr * s
            improved = True
        
        pts = np.clip(pts, -0.5, 0.5)
        
        if improved:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        
        lr *= decay
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 0.0
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Multi-restart with L-BFGS-B
    num_restarts = 0
    while time.time() - start_time < DEADLINE - 20:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 15:
            break
        
        init_type = num_restarts
        points = _generate_init(init_type)
        x0 = points.ravel()
        
        # Adaptive beta: start coarse, refine
        betas = [5.0, 10.0, 20.0, 50.0]
        x_current = x0
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            try:
                res = minimize(
                    _soft_min_obj,
                    x_current,
                    jac=_soft_min_grad,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
                )
                x_current = res.x
            except Exception:
                break
        
        # Final polish
        pts = x_current.reshape(N, 3)
        pts = np.clip(pts, -0.5, 0.5)
        
        polish_time = min(5.0, max(1.0, (DEADLINE - 5) - (time.time() - start_time)))
        if polish_time > 0.5:
            pts, sc = _polish_local(pts, polish_time)
        else:
            sc = _score(pts)
        
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        
        num_restarts += 1
    
    if best_points is None:
        best_points = _generate_init(0)
    
    return best_points

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
