import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-12):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def soft_min_grad(bary_flat, beta=100.0):
    """
    Gradient of the soft-min objective.
    Returns (value, gradient)
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10, np.zeros_like(bary_flat)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Fall back to hard min gradient (subgradient)
        min_idx = np.argmin(areas)
        # This is complex, so just return a dummy gradient
        return -np.min(areas), np.zeros_like(bary_flat)
        
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp  # softmax weights
    
    # The gradient of soft_min w.r.t. areas is -weights
    # d(soft_min)/d(bary) = -weights @ d(areas)/d(bary)
    # But we want to MINIMIZE soft_min, so we want -d(soft_min)/d(bary) = weights @ d(areas)/d(bary)
    # Wait, let's be careful.
    # f = -1/beta * log(sum(exp(-beta * A)))
    # df/dA_j = -1/beta * (1/sum(exp)) * exp(-beta*A_j) * (-beta) = exp(-beta*A_j)/sum(exp) = weights[j]
    # So grad_f = sum_j weights[j] * grad_A_j
    # We want to minimize f, so we use -grad_f as the direction of descent? No, minimize uses the gradient.
    # scipy.minimize minimizes f. It computes grad = df/dx.
    # So we need df/dx.
    # df/dx = sum_j weights[j] * dA_j/dx
    
    # Compute dA_j/dx for each triangle j
    # A_j = |det|
    # This is getting complicated. Let's just use numerical gradient or a simpler approach.
    # Actually, for L-BFGS-B, providing an analytical gradient is much faster.
    # Let's compute the gradient of the log-sum-exp term.
    
    # Let L = log(sum(exp(-beta * A)))
    # dL/dA_j = -beta * exp(-beta * A_j) / sum(exp(-beta * A_k)) = -beta * weights[j]
    # f = -L / beta
    # df/dA_j = -1/beta * dL/dA_j = weights[j]
    
    # So we need grad_x = sum_j weights[j] * grad_x(A_j)
    
    # A_j = |det_j|
    # Let det_j be the signed area.
    # d(det_j)/d(u_i) etc.
    
    # Let's compute the Jacobian of areas w.r.t. barycentric coords.
    # J[j, k, m] = dA_j / d(bary[k, m])
    
    # This is 165 x 11 x 2.
    # We can compute this efficiently.
    
    # For a triangle (i, j, k), the area is 0.5 * |det|
    # det = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j)
    
    # This is doable but verbose. Given time constraints, I'll stick to numerical gradients
    # or just use L-BFGS-B without gradient (it will use finite differences).
    # However, finite differences on 22 variables x 165 areas is expensive.
    
    # Let's try to implement the analytical gradient.
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # We need d(det_j)/d(u_p), d(det_j)/d(v_p) for all j, p.
    # det_j depends only on u_i, v_i, u_j, v_j, u_k, v_k.
    
    # Let's precompute the partial derivatives of det_j w.r.t. u and v for each point involved.
    
    # For triangle (a, b, c):
    # det = u_a(v_b w_c - v_c w_b) - v_a(u_b w_c - u_c w_b) + w_a(u_b v_c - u_c v_b)
    
    # d(det)/d(u_a) = (v_b w_c - v_c w_b)
    # d(det)/d(v_a) = -(u_b w_c - u_c w_b)
    # d(det)/d(w_a) = (u_b v_c - u_c v_b)
    # Note: w_a = 1 - u_a - v_a, so d(w_a)/d(u_a) = -1, d(w_a)/d(v_a) = -1.
    
    # So:
    # d(det)/d(u_a) = (v_b w_c - v_c w_b) - (u_b v_c - u_c v_b)
    # d(det)/d(v_a) = -(u_b w_c - u_c w_b) - (u_b v_c - u_c v_b)
    
    # Similarly for b and c.
    
    # Let's compute these for all triangles.
    
    a_idx = tri_idx[:, 0]
    b_idx = tri_idx[:, 1]
    c_idx = tri_idx[:, 2]
    
    ua = u[a_idx]; va = v[a_idx]; wa = w[a_idx]
    ub = u[b_idx]; vb = v[b_idx]; wb = w[b_idx]
    uc = u[c_idx]; vc = v[c_idx]; wc = w[c_idx]
    
    # Terms
    term1 = vb * wc - vc * wb  # d(det)/d(u_a) part 1
    term2 = -(ub * wc - uc * wb) # d(det)/d(v_a) part 1
    term3 = ub * vc - uc * vb  # d(det)/d(w_a) part 1
    
    # Full derivatives w.r.t. u_a, v_a
    d_det_d_ua = term1 - term3
    d_det_d_va = term2 - term3
    
    # For b:
    # d(det)/d(u_b) = (v_c w_a - v_a w_c) - (u_c v_a - u_a v_c)
    # d(det)/d(v_b) = -(u_c w_a - u_a w_c) - (u_c v_a - u_a v_c)
    
    term4 = vc * wa - va * wc
    term5 = -(uc * wa - ua * wc)
    term6 = uc * va - ua * vc
    
    d_det_d_ub = term4 - term6
    d_det_d_vb = term5 - term6
    
    # For c:
    # d(det)/d(u_c) = (v_a w_b - v_b w_a) - (u_a v_b - u_b v_a)
    # d(det)/d(v_c) = -(u_a w_b - u_b w_a) - (u_a v_b - u_b v_a)
    
    term7 = va * wb - vb * wa
    term8 = -(ua * wb - ub * wa)
    term9 = ua * vb - ub * va
    
    d_det_d_uc = term7 - term9
    d_det_d_vc = term8 - term9
    
    # Now, A_j = |det_j|
    # dA_j/dx = sign(det_j) * d(det_j)/dx
    
    signs = np.sign(np.column_stack([
        ua * (vb * wc - vc * wb) - va * (ub * wc - uc * wb) + wa * (ub * vc - uc * vb)
    ]))
    # Actually, we already have det in calc_areas_bary. Let's recompute it here for consistency.
    dets = ua * (vb * wc - vc * wb) - va * (ub * wc - uc * wb) + wa * (ub * vc - uc * vb)
    signs = np.sign(dets)
    
    # Now we have the gradient of each area w.r.t. the coordinates of the 3 vertices.
    # We need to accumulate these into a single gradient vector of shape (N, 2).
    
    grad_areas = np.zeros((len(dets), N, 2))
    
    grad_areas[:, a_idx, 0] = signs * d_det_d_ua
    grad_areas[:, a_idx, 1] = signs * d_det_d_va
    
    grad_areas[:, b_idx, 0] = signs * d_det_d_ub
    grad_areas[:, b_idx, 1] = signs * d_det_d_vb
    
    grad_areas[:, c_idx, 0] = signs * d_det_d_uc
    grad_areas[:, c_idx, 1] = signs * d_det_d_vc
    
    # grad_f = sum_j weights[j] * grad_A_j
    # weights has shape (165,)
    # grad_areas has shape (165, 11, 2)
    
    weighted_grad = np.sum(weights[:, np.newaxis, np.newaxis] * grad_areas, axis=0)
    
    # weighted_grad has shape (11, 2)
    # Flatten it
    grad_flat = weighted_grad.flatten()
    
    # The objective is -soft_min.
    # soft_min = -log_sum / beta
    # f = -soft_min = log_sum / beta
    # Wait, my soft_min_obj returns -soft_min.
    # Let's check:
    # log_sum = max_scaled + log(sum(exp(scaled - max_scaled)))
    # soft_min = -log_sum / beta
    # return -soft_min = log_sum / beta
    
    # So f = log_sum / beta
    # df/dx = 1/beta * d(log_sum)/dx
    # d(log_sum)/dx = d(log(sum(exp(-beta * A))))/dx
    # = (1/sum(exp)) * sum(exp(-beta * A_j) * (-beta * dA_j/dx))
    # = -beta * sum(weights[j] * dA_j/dx)
    
    # So df/dx = 1/beta * (-beta * sum(weights[j] * dA_j/dx)) = -sum(weights[j] * dA_j/dx)
    
    # So the gradient of the objective f is -weighted_grad.
    
    return f, -grad_flat

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                bary = clamp_bary(bary)
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    1. L-BFGS-B on soft-min (smooths the landscape)
    2. Nelder-Mead on hard-min (handles non-differentiability)
    3. SLSQP on hard-min (polish with bounds)
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with analytical gradient
    try:
        def obj_and_grad(x):
            val, grad = soft_min_grad(x, beta=100.0)
            return val, grad
        
        result1 = minimize(
            obj_and_grad,
            bary_current.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            jac=True,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Hexagonal Init + Refine
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Basin Hopping / Multi-start Loop
    rng = np.random.default_rng(123)
    restart_count = 0
    base_pert_scale = 0.05
    
    while time.time() < deadline - 12:
        restart_count += 1
        
        if best_bary is not None and restart_count % 3 != 0:
            sigma = base_pert_scale * (1.0 + 0.1 * (restart_count % 10))
            noise = rng.normal(0, sigma, best_bary.shape)
            init_bary = clamp_bary(best_bary + noise)
        else:
            init_bary = generate_random_init(seed=1000 + restart_count)
            
        remaining_time = deadline - time.time() - 10
        if remaining_time < 3:
            break
            
        refine_time = min(5.0, remaining_time)
        
        try:
            refined = refine_local(init_bary, time_limit=refine_time)
            update_best(refined)
        except Exception:
            pass

    # 4. Final High-Precision Polish
    if best_bary is not None and time.time() < deadline - 2:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    cart = bary_to_cart(best_bary)
    
    return cart

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Time taken: {time.time() - start:.2f}s")
    print(f"Result shape: {res.shape}")
    print("Done.")
