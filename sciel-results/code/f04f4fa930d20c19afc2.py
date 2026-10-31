import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

def _min_area_bary_fast(points_bary):
    """Vectorized minimum triangle area in barycentric coordinates."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    n = len(u)
    # Compute all determinants in a vectorized manner
    # For each triple (i,j,k): det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
    
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k])
                     - v[i] * (u[j] * w[k] - w[j] * u[k])
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                a = abs(det)
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_area(points_bary, alpha=50.0):
    """Soft minimum using log-sum-exp for smoother gradients."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    n = len(u)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k])
                     - v[i] * (u[j] * w[k] - w[j] * u[k])
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                areas.append(abs(det))
    
    areas = np.array(areas)
    # Soft min: -alpha * log(sum(exp(-areas/alpha)))
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / alpha)))
    soft_min = min_val - alpha * log_sum
    return soft_min

def _objective(params, use_soft=False, alpha=50.0):
    """Objective function: negative of min area (or soft min)."""
    points_bary = params.reshape(N, 2)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    # Check validity
    if np.any(u < -1e-10) or np.any(v < -1e-10) or np.any(w < -1e-10):
        return 1e10
    
    # Clamp small negatives
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    w = np.maximum(w, 0)
    
    s = u + v + w
    if np.any(s < 1e-10):
        return 1e10
    u /= s
    v /= s
    w /= s
    
    points_bary = np.column_stack([u, v])
    
    if use_soft:
        min_a = _soft_min_area(points_bary, alpha)
    else:
        min_a = _min_area_bary_fast(points_bary)
    
    return -min_a

def _bary_to_cart(points_bary):
    """Convert barycentric (u,v) with w=1-u-v to Cartesian in equilateral triangle."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    h = np.sqrt(3) / 2
    x = v * 1.0 + w * 0.5
    y = w * h
    return np.column_stack([x, y])

def _clamp_to_triangle(points_cart):
    """Ensure points are inside the equilateral triangle."""
    eps = 1e-9
    for i in range(len(points_cart)):
        x, y = points_cart[i]
        y = max(eps, y)
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x - eps
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x) - eps
        x = max(eps, min(1 - eps, x))
        points_cart[i] = [x, y]
    return points_cart

def _polish(points_bary, deadline, max_iter=2000):
    """Final high-precision polish using Nelder-Mead with tight tolerances."""
    if time.time() > deadline - 5:
        return points_bary
    
    x0 = points_bary.flatten()
    
    def obj(params):
        pb = params.reshape(N, 2)
        u = pb[:, 0]
        v = pb[:, 1]
        w = 1.0 - u - v
        if np.any(u < -1e-8) or np.any(v < -1e-8) or np.any(w < -1e-8):
            return 1e10
        u = np.maximum(u, 0)
        v = np.maximum(v, 0)
        w = np.maximum(w, 0)
        s = u + v + w
        if np.any(s < 1e-10):
            return 1e10
        u /= s
        v /= s
        w /= s
        pb = np.column_stack([u, v])
        return -_min_area_bary_fast(pb)
    
    try:
        res = minimize(
            obj, x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.fun < -1e-15:
            pb = res.x.reshape(N, 2)
            u = np.maximum(pb[:, 0], 0)
            v = np.maximum(pb[:, 1], 0)
            w = 1.0 - u - v
            if np.all(w >= -1e-10):
                u = np.maximum(u, 0)
                v = np.maximum(v, 0)
                w = np.maximum(w, 0)
                s = u + v + w
                u /= s
                v /= s
                return np.column_stack([u, v])
    except Exception:
        pass
    
    return points_bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_score = -float('inf')
    best_points_bary = None
    
    # Phase 1: Differential Evolution for global search
    bounds = [(0.0, 1.0)] * (2 * N)
    
    if time.time() < deadline - 70:
        try:
            de_result = differential_evolution(
                lambda p: _objective(p, use_soft=False),
                bounds,
                seed=42,
                maxiter=150,
                popsize=12,
                tol=1e-8,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,
                update_interval=5,
                workers=1
            )
            
            score = -de_result.fun
            if score > best_score and score > 0:
                best_score = score
                pb = de_result.x.reshape(N, 2)
                u = np.maximum(pb[:, 0], 0)
                v = np.maximum(pb[:, 1], 0)
                w = 1.0 - u - v
                if np.all(w >= -1e-10):
                    u = np.maximum(u, 0)
                    v = np.maximum(v, 0)
                    w = np.maximum(w, 0)
                    s = u + v + w
                    u /= s
                    v /= s
                    best_points_bary = np.column_stack([u, v])
        except Exception:
            pass
    
    # Phase 2: Multiple L-BFGS-B restarts from diverse initializations
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # Grid-based initialization
    grid_10 = [
        (1.0, 0.0), (2/3, 1/3), (1/3, 2/3), (0.0, 1.0),
        (2/3, 0.0), (1/3, 1/3), (0.0, 2/3),
        (1/3, 0.0), (0.0, 1/3),
        (0.0, 0.0)
    ]
    inits.append(np.array(grid_10 + [(0.4, 0.4)]))
    
    # Perturbed grid
    for _ in range(8):
        base = np.array(grid_10 + [(0.4, 0.4)])
        noise = rng.uniform(-0.08, 0.08, size=base.shape)
        perturbed = base + noise
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
        sum_uv = perturbed[:, 0] + perturbed[:, 1]
        mask = sum_uv > 1
        if np.any(mask):
            factor = 1.0 / sum_uv[mask]
            perturbed[mask, 0] *= factor
            perturbed[mask, 1] *= factor
        inits.append(perturbed)
    
    # Random Dirichlet initializations
    for _ in range(8):
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary[:, :2])
    
    # If we have a good DE result, add perturbed versions of it
    if best_points_bary is not None:
        for _ in range(4):
            noise = rng.uniform(-0.05, 0.05, size=best_points_bary.shape)
            perturbed = best_points_bary + noise
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
            sum_uv = perturbed[:, 0] + perturbed[:, 1]
            mask = sum_uv > 1
            if np.any(mask):
                factor = 1.0 / sum_uv[mask]
                perturbed[mask, 0] *= factor
                perturbed[mask, 1] *= factor
            inits.append(perturbed)
    
    # L-BFGS-B optimization from each init
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        def objective(params):
            return _objective(params, use_soft=False)
        
        try:
            res = minimize(
                objective,
                init.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            uv_opt = res.x.reshape(N, 2)
            u_opt = np.maximum(uv_opt[:, 0], 0)
            v_opt = np.maximum(uv_opt[:, 1], 0)
            w_opt = 1.0 - u_opt - v_opt
            
            if np.any(w_opt < -1e-8):
                continue
            
            u_opt = np.maximum(u_opt, 0)
            v_opt = np.maximum(v_opt, 0)
            w_opt = np.maximum(w_opt, 0)
            s = u_opt + v_opt + w_opt
            u_opt /= s
            v_opt /= s
            w_opt /= s
            
            pb_opt = np.column_stack([u_opt, v_opt])
            min_a_opt = _min_area_bary_fast(pb_opt)
            
            if min_a_opt > best_score:
                best_score = min_a_opt
                best_points_bary = pb_opt.copy()
                
        except Exception:
            pass
    
    # Phase 3: Soft-min optimization from best point (smoother landscape)
    if best_points_bary is not None and time.time() < deadline - 15:
        try:
            def soft_obj(params):
                return _objective(params, use_soft=True, alpha=30.0)
            
            res_soft = minimize(
                soft_obj,
                best_points_bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            uv_soft = res_soft.x.reshape(N, 2)
            u_soft = np.maximum(uv_soft[:, 0], 0)
            v_soft = np.maximum(uv_soft[:, 1], 0)
            w_soft = 1.0 - u_soft - v_soft
            
            if np.all(w_soft >= -1e-8):
                u_soft = np.maximum(u_soft, 0)
                v_soft = np.maximum(v_soft, 0)
                w_soft = np.maximum(w_soft, 0)
                s = u_soft + v_soft + w_soft
                u_soft /= s
                v_soft /= s
                w_soft /= s
                
                pb_soft = np.column_stack([u_soft, v_soft])
                min_a_soft = _min_area_bary_fast(pb_soft)
                
                if min_a_soft > best_score:
                    best_score = min_a_soft
                    best_points_bary = pb_soft.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if best_points_bary is not None and time.time() < deadline - 5:
        best_points_bary = _polish(best_points_bary, deadline, max_iter=3000)
    
    # Fallback if nothing worked
    if best_points_bary is None:
        grid_points = []
        for i in range(4):
            for j in range(4 - i):
                u = j / 3.0
                v = i / 3.0
                w = 1 - u - v
                if w >= -1e-10:
                    grid_points.append([u, v])
        if len(grid_points) < N:
            grid_points.append([1/3, 1/3])
        best_points_bary = np.array(grid_points[:N])
    
    # Convert to Cartesian
    points_cart = _bary_to_cart(best_points_bary)
    points_cart = _clamp_to_triangle(points_cart)
    
    return points_cart
