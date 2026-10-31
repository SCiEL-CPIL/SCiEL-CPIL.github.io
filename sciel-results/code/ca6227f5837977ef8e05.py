import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math
from itertools import combinations

N = 11
TIME_LIMIT = 140.0

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

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

def clamp_bary(bary, eps=1e-9):
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

def hard_min_obj(x):
    """Objective: negative of the minimum area (for minimization)."""
    bary = x[:N*2].reshape(N, 2)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(x, alpha=10.0):
    """Smooth surrogate: -log-sum-exp of negative areas scaled by alpha."""
    bary = x[:N*2].reshape(N, 2)
    areas = calc_areas_bary(bary)
    # Use log-sum-exp trick for stability
    max_neg = -np.min(areas)  # This is -min(areas)
    # We want to approximate min(areas) = -max(-areas)
    # log-sum-exp(-alpha * areas) approximates -alpha * min(areas) + log(165)
    # So -1/alpha * log-sum-exp(-alpha * areas) approximates min(areas)
    
    neg_alpha_areas = -alpha * areas
    max_neg_alpha = np.max(neg_alpha_areas)
    if max_neg_alpha > 700:
        # Overflow protection
        return -np.min(areas)
    
    lse = max_neg_alpha + np.log(np.sum(np.exp(neg_alpha_areas - max_neg_alpha)))
    return -lse / alpha

def soft_min_obj_grad(x, alpha=10.0):
    """Gradient of soft_min_obj."""
    bary = x[:N*2].reshape(N, 2)
    areas = calc_areas_bary(bary)
    
    # Gradient of -1/alpha * log-sum-exp(-alpha * areas) w.r.t. x
    # = 1/alpha * sum_i (exp(-alpha * areas_i) / sum_j exp(-alpha * areas_j)) * (-alpha * d(areas_i)/dx)
    # = -sum_i (p_i * d(areas_i)/dx) where p_i = exp(-alpha * areas_i) / sum(exp(-alpha * areas_j))
    
    neg_alpha_areas = -alpha * areas
    max_neg_alpha = np.max(neg_alpha_areas)
    if max_neg_alpha > 700:
        # Fall back to hard gradient (subgradient of min)
        min_idx = np.argmin(areas)
        # We need gradient of areas[min_idx]
        # This is complex to compute analytically. For simplicity, return zeros if overflow.
        return np.zeros_like(x)
    
    exp_vals = np.exp(neg_alpha_areas - max_neg_alpha)
    probs = exp_vals / np.sum(exp_vals)
    
    # Now we need d(areas_i)/dx for all i, weighted by probs
    # areas_i = |det_i|. The sign of det matters for the gradient.
    # Let's compute the signed determinants first.
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
    
    # d(det_i)/dx is a sparse vector.
    # For a specific triangle i with points p, q, r:
    # det = u_p(v_q w_r - v_r w_q) - v_p(u_q w_r - u_r w_q) + w_p(u_q v_r - u_r v_q)
    # This is linear in u_p, v_p, w_p, etc.
    # Since w = 1 - u - v, dw/du = -1, dw/dv = -1.
    
    # Instead of deriving the full analytical gradient which is error-prone and complex,
    # and given the time constraint, I will rely on the fact that `minimize` with BFGS
    # or L-BFGS-B can use numerical gradients if `jac` is not provided, but that's slow.
    
    # However, implementing the analytical gradient for the log-sum-exp of determinants
    # is the key to speed.
    
    # Let's compute the gradient of the signed determinant for each triangle.
    # d(det_i)/dx_k where x_k is a coordinate of point j.
    
    # This is getting too complex to implement correctly in this short time without testing.
    # I will use `minimize` with method='L-BFGS-B' and let it use numerical gradients,
    # but limit the iterations. Or, I can use a simpler approach:
    # Use the hard objective with a subgradient? No, SLSQP needs smoothness.
    
    # Alternative: Use the soft objective but provide a simplified gradient
    # or just let scipy compute it numerically. For N=11, 22 variables, numerical
    # gradient is 23 function evaluations. If each is ~1ms, that's 23ms/step.
    # 100 steps = 2.3s. This is feasible.
    
    return None  # Signal to use numerical gradient

def generate_hexagonal_init(seed=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    
    # Center
    bary[0] = [1/3, 1/3]
    
    # First ring (6 points)
    # In barycentric, a hexagon around the center can be approximated.
    # Let's place them at distance r from center.
    # Center is (1/3, 1/3).
    # Directions in Cartesian: 0, 60, 120, 180, 240, 300 degrees.
    # But we need to stay in the triangle.
    
    # Simpler: Use a fixed set of "good" points.
    # Known good configurations often have points near:
    # Vertices: (0,0), (0,1), (1,0) -> bary (0,0), (0,1), (1,0)
    # Midpoints: (0,0.5), (0.5,0), (0.5,0.5)
    # Center: (1/3, 1/3)
    # Inner ring: (1/6, 1/6), (1/6, 5/6), (5/6, 1/6) ... etc.
    
    # Let's try a specific pattern:
    # 1 center
    # 3 near vertices
    # 3 near midpoints
    # 3 near "inner" lattice points
    # 1 extra?
    
    # Let's use a perturbed grid.
    # Grid 4x4 in barycentric?
    # u, v in {0, 0.25, 0.5, 0.75, 1} such that u+v <= 1.
    # (0,0), (0,0.25), (0,0.5), (0,0.75), (0,1) -> 5
    # (0.25,0), (0.25,0.25), (0.25,0.5), (0.25,0.75) -> 4 (0.25,1) invalid
    # (0.5,0), (0.5,0.25), (0.5,0.5) -> 3
    # (0.75,0), (0.75,0.25) -> 2
    # (1,0) -> 1
    # Total 15 points. We need 11.
    
    # Select 11 from this grid.
    grid_pts = []
    for i in range(5):
        for j in range(5):
            u = i * 0.25
            v = j * 0.25
            if u + v <= 1.0 + 1e-9:
                grid_pts.append([u, v])
    
    # grid_pts has 15 entries.
    # We want to pick 11.
    # Maybe pick the ones that are most "spread out".
    # Or just pick the first 11?
    # Let's pick a subset that includes the center and corners.
    
    # Indices of grid_pts:
    # 0: (0,0)
    # 1: (0,0.25)
    # 2: (0,0.5)
    # 3: (0,0.75)
    # 4: (0,1)
    # 5: (0.25,0)
    # 6: (0.25,0.25)
    # 7: (0.25,0.5)
    # 8: (0.25,0.75)
    # 9: (0.5,0)
    # 10: (0.5,0.25)
    # 11: (0.5,0.5)
    # 12: (0.75,0)
    # 13: (0.75,0.25)
    # 14: (1,0)
    
    # Let's pick: 0, 4, 14 (vertices)
    # 2, 9, 11 (midpoints)
    # 6, 7, 10 (inner)
    # 3, 8, 13 (outer mid)
    # Total 11.
    
    selected_indices = [0, 4, 14, 2, 9, 11, 6, 7, 10, 3, 8]
    bary = np.array([grid_pts[i] for i in selected_indices])
    
    # Add small perturbation
    noise = rng.normal(0, 0.01, bary.shape)
    bary += noise
    bary = clamp_bary(bary)
    
    return bary

def generate_random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        val = np.min(areas)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Strategy:
    # 1. Start with a good heuristic init.
    # 2. Use Differential Evolution for global exploration (fast, parallelizable).
    # 3. Use L-BFGS-B with soft objective for local refinement.
    
    # Phase 1: Differential Evolution
    # Bounds for barycentric coords: u in [0,1], v in [0,1], but we must enforce u+v <= 1.
    # DE doesn't handle complex constraints well.
    # We can use a penalty function in the objective.
    
    def de_obj(x):
        bary = x[:N*2].reshape(N, 2)
        # Check constraints
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1 - u - v
        
        # Penalty if out of bounds
        penalty = 0
        if np.any(u < 0) or np.any(v < 0) or np.any(w < 0):
            penalty += 1000 * (np.sum(np.clip(u, 0, None)**2) + np.sum(np.clip(v, 0, None)**2) + np.sum(np.clip(w, 0, None)**2))
        
        # Also clamp for area calculation to avoid issues
        bary_clamped = clamp_bary(bary)
        areas = calc_areas_bary(bary_clamped)
        min_area = np.min(areas)
        
        return -min_area + penalty

    # DE bounds
    bounds_de = [(0, 1)] * (N * 2)
    
    if time.time() < deadline - 20:
        try:
            res_de = differential_evolution(
                de_obj,
                bounds_de,
                maxiter=20,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False  # We will polish manually
            )
            if res_de.success:
                update_best(res_de.x[:N*2].reshape(N, 2))
        except Exception:
            pass
    
    # Phase 2: Local Refinement with L-BFGS-B
    # Use multiple starts: DE result, Heuristic init, Random inits.
    
    inits = []
    if best_bary is not None:
        inits.append(best_bary)
    inits.append(generate_hexagonal_init(0))
    for i in range(3):
        inits.append(generate_random_init(100 + i))
    
    for init_bary in inits:
        if time.time() > deadline - 5:
            break
            
        x0 = init_bary.flatten()
        
        # Use soft objective for smoothness
        # L-BFGS-B can handle bounds.
        # Bounds: u, v in [0, 1]. The constraint u+v<=1 is not directly supported by L-BFGS-B.
        # We must rely on the soft objective to penalize violations or clamp.
        # Since L-BFGS-B doesn't support inequality constraints other than bounds,
        # we will use a penalty in the soft objective if out of bounds.
        
        def soft_obj_with_penalty(x, alpha=10.0):
            bary = x[:N*2].reshape(N, 2)
            u = bary[:, 0]
            v = bary[:, 1]
            w = 1 - u - v
            
            # Penalty for leaving the triangle
            penalty = 0
            if np.any(u < 0) or np.any(v < 0) or np.any(w < 0):
                penalty += 100 * (np.sum(np.clip(u, 0, None)**2) + np.sum(np.clip(v, 0, None)**2) + np.sum(np.clip(w, 0, None)**2))
            
            bary_clamped = clamp_bary(bary)
            val = soft_min_obj(bary_clamped, alpha)
            return val + penalty
        
        # We don't have an analytical gradient for the penalty part easily.
        # Let's use numerical gradients.
        
        try:
            res_lbfgs = minimize(
                soft_obj_with_penalty,
                x0,
                method='L-BFGS-B',
                bounds=bounds_de,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res_lbfgs.success or res_lbfgs.fun < 0:
                update_best(res_lbfgs.x[:N*2].reshape(N, 2))
        except Exception:
            pass
    
    # Phase 3: Final SLSQP polish on the best found
    if best_bary is not None and time.time() < deadline - 2:
        x0 = np.concatenate([best_bary.flatten(), [best_val]])
        
        def epigraph_obj(x):
            bary = x[:N*2].reshape(N, 2)
            t = x[N*2]
            return -t
        
        def epigraph_grad(x):
            grad = np.zeros_like(x)
            grad[N*2] = -1.0
            return grad
        
        def epigraph_con_fun(x):
            bary = x[:N*2].reshape(N, 2)
            t = x[N*2]
            areas = calc_areas_bary(bary)
            cons = areas - t
            bary_cons = np.zeros(3*N)
            bary_cons[0::3] = bary[:, 0]
            bary_cons[1::3] = bary[:, 1]
            bary_cons[2::3] = 1.0 - bary[:, 0] - bary[:, 1]
            return np.concatenate([cons, bary_cons])
        
        constraints = {'type': 'ineq', 'fun': epigraph_con_fun}
        bounds_slsqp = bounds_de + [(0, 1)]
        
        try:
            res_slsqp = minimize(
                epigraph_obj,
                x0,
                method='SLSQP',
                jac=epigraph_grad,
                bounds=bounds_slsqp,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if res_slsqp.success or res_slsqp.fun < 0:
                update_best(res_slsqp.x[:N*2].reshape(N, 2))
        except Exception:
            pass
    
    # Fallback
    if best_bary is None:
        best_bary = generate_random_init(0)
        
    cart = bary_to_cart(best_bary)
    return cart
