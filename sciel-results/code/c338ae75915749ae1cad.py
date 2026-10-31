import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_TOTAL = np.sqrt(3) / 4.0

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(x, y):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v."""
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - w - v
    return np.column_stack([u, v])

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
    """Clamp barycentric coordinates to the valid triangle region."""
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

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def compute_jacobian(bary):
    """
    Compute the Jacobian of the areas with respect to the barycentric coordinates.
    J[i, 2*k] = d(area_i)/du_k
    J[i, 2*k+1] = d(area_i)/dv_k
    
    Area_i = |det(P_i, P_j, P_k)|
    In barycentric coords, the determinant is linear in each point's coordinates.
    Let P_i = (u_i, v_i, w_i).
    det = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j)
    
    Since w = 1 - u - v:
    det = u_i * A_i - v_i * B_i + (1-u_i-v_i) * C_i
        = u_i(A_i - C_i) - v_i(B_i + C_i) + C_i
    
    Where:
    A_i = v_j w_k - v_k w_j
    B_i = u_j w_k - u_k w_j
    C_i = u_j v_k - u_k v_j
    
    So:
    d(det)/du_i = A_i - C_i
    d(det)/dv_i = -(B_i + C_i)
    
    For points j and k, we need to differentiate A, B, C with respect to their coords.
    Note that the area is |det|. If det > 0, grad = d(det)/dx. If det < 0, grad = -d(det)/dx.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    J = np.zeros((len(tri_idx), N * 2))
    
    # Vectorized calculation of coefficients
    # For each triangle (i, j, k):
    i_idx = tri_idx[:, 0]
    j_idx = tri_idx[:, 1]
    k_idx = tri_idx[:, 2]
    
    ui, vi, wi = u[i_idx], v[i_idx], w[i_idx]
    uj, vj, wj = u[j_idx], v[j_idx], w[j_idx]
    uk, vk, wk = u[k_idx], v[k_idx], w[k_idx]
    
    # Coefficients for point i
    A_i = vj * wk - vk * wj
    B_i = uj * wk - uk * wj
    C_i = uj * vk - uk * vj
    
    # Coefficients for point j
    # det = ui(vj wk - vk wj) - vi(uj wk - uk wj) + wi(uj vk - uk vj)
    # d/dvj: ui * wk - wi * uk
    # d/dwj: ui * vj - wi * uj
    # d/dvj for v_j term: -vi * wk (wait, B_i depends on v_j? No, B_i = uj wk - uk wj. v_j is not in B_i.
    # Let's re-derive carefully.
    # det = ui(vj wk - vk wj) - vi(uj wk - uk wj) + wi(uj vk - uk vj)
    # Expand:
    # = ui vj wk - ui vk wj - vi uj wk + vi uk wj + wi uj vk - wi uk vj
    
    # Terms with vj:
    # ui vj wk - wi uk vj = vj (ui wk - wi uk)
    # So d(det)/dvj = ui wk - wi uk
    
    # Terms with wj:
    # - ui vk wj + vi uk wj = wj (vi uk - ui vk)
    # Note wj = 1 - uj - vj. So wj is not an independent variable in (u,v) space?
    # We are differentiating w.r.t (uj, vj).
    # wj = 1 - uj - vj.
    # d(wj)/duj = -1, d(wj)/dvj = -1.
    
    # Let's use the chain rule or explicit substitution.
    # det(uj, vj) = ui(vj wk - vk wj) - vi(uj wk - uk wj) + wi(uj vk - uk vj)
    # Substitute wj = 1 - uj - vj, wk = 1 - uk - vk (constant w.r.t j), wi, ui, vi, vk, uk, vj (variable), uj (variable).
    
    # This is getting messy. Let's use the property that the area of a triangle with vertices in barycentric coords
    # is 0.5 * |det|. The determinant is multilinear.
    # Actually, there is a simpler way. The area is a linear function of each vertex's coordinates if the other two are fixed.
    # Area_i = 0.5 * |det(M)|.
    # Let's just compute the gradient numerically for the active set? No, we need it for the Newton step.
    # Let's compute the Jacobian for ALL triangles once, then select active ones.
    
    # d(det)/du_i = A_i - C_i
    # d(det)/dv_i = -(B_i + C_i)
    
    # For point j:
    # det = ... + vj (ui wk - wi uk) ...
    # Wait, let's look at the term linear in vj.
    # From expansion: ui vj wk - wi uk vj = vj (ui wk - wi uk).
    # So coeff of vj is (ui wk - wi uk).
    # What about uj?
    # Terms with uj: - vi uj wk + wi uj vk = uj (wi vk - vi wk).
    # So coeff of uj is (wi vk - vi wk).
    # But we must account for the fact that wj changes when uj/vj changes.
    # The formula d(det)/dx is valid if we treat u,v,w as independent with sum=1 constraint?
    # No, in the (u,v) parameterization, w is dependent.
    
    # Let's stick to the (u,v) parameterization explicitly.
    # det(uj, vj) = ui(vj wk - vk(1-uj-vj)) - vi(uj wk - uk(1-uj-vj)) + wi(uj vk - uk vj)
    # = ui vj wk - ui vk + ui vk uj + ui vk vj - vi uj wk + vi uk - vi uk uj - vi uk vj + wi uj vk - wi uk vj
    
    # Group by uj:
    # uj (ui vk - vi wk - vi uk + wi vk)
    # Group by vj:
    # vj (ui wk + ui vk - vi uk - wi uk)
    # Constant:
    # - ui vk + vi uk
    
    # So:
    # d(det)/duj = ui vk - vi wk - vi uk + wi vk
    # d(det)/dvj = ui wk + ui vk - vi uk - wi uk
    
    # Let's verify for point i.
    # det(ui, vi) = ui(vj wk - vk wj) - vi(uj wk - uk wj) + (1-ui-vi)(uj vk - uk vj)
    # = ui vj wk - ui vk wj - vi uj wk + vi uk wj + uj vk - ui uj vk - vi uj vk - uk vj + ui ui uk vj + vi vi uk vj ... this is getting complicated.
    
    # Let's use the vectorized formulas derived from the linear dependence.
    # For point i:
    # dA_i/du_i = A_i - C_i
    # dA_i/dv_i = -(B_i + C_i)
    
    # For point j:
    # dA_i/du_j = W_i * V_k - V_i * W_k - V_i * U_k + W_i * V_k  (Wait, check signs)
    # Let's re-calculate d(det)/du_j from: uj (wi vk - vi wk - vi uk + wi vk)
    # = wi vk - vi wk - vi uk + wi vk = 2 wi vk - vi wk - vi uk.
    # Is this correct?
    # Let's check with a simple case.
    
    # Actually, there is a known result. The gradient of the determinant with respect to the i-th row is the cross product of the other two rows (in 3D barycentric space).
    # In 2D barycentric (u,v), the mapping to 3D (u,v,w) is linear.
    # Let's just use finite differences for the Jacobian if it's too slow to derive, but 165*22 = 3630 ops per point? No, 165 triangles * 22 vars.
    # We can compute the Jacobian for the active set only.
    
    # Let's implement the analytic gradient for the active set.
    
    # Coeffs for i:
    d_det_d_ui = A_i - C_i
    d_det_d_vi = -(B_i + C_i)
    
    # Coeffs for j:
    # From derivation:
    # d(det)/du_j = 2*wi*vk - vi*wk - vi*uk  <-- This looks suspicious. Let's re-verify.
    # Let's use the property: det(P, Q, R) is linear in P.
    # grad_P det = Q x R (in 3D barycentric).
    # Q x R = (vj wk - vk wj, vk uj - vj uk, vj uk - vk uj) ... wait, cross product in 3D.
    # The 3D vector for a point is (u, v, w).
    # grad_u (component 1) = vj wk - vk wj = A_i.
    # grad_v (component 2) = vk uj - vj uk.
    # grad_w (component 3) = vj uk - vk uj.
    
    # The objective is f(u,v) = det(u, v, 1-u-v).
    # df/du = grad_u * 1 + grad_v * 0 + grad_w * (-1) = grad_u - grad_w.
    # df/dv = grad_u * 0 + grad_v * 1 + grad_w * (-1) = grad_v - grad_w.
    
    # So for point i:
    # grad_u = A_i
    # grad_v = vk * uj - vj * uk  (Wait, this is for P_i, so Q=P_j, R=P_k)
    # grad_w = vj * uk - vk * uj
    # df/du_i = A_i - (vj uk - vk uj) = A_i - C_i. (Matches)
    # df/dv_i = (vk uj - vj uk) - (vj uk - vk uj) = 2(vk uj - vj uk).
    # Wait, my previous derivation was -(B_i + C_i).
    # B_i = uj wk - uk wj. C_i = uj vk - uk vj.
    # Let's check if 2(vk uj - vj uk) == -(B_i + C_i).
    # B_i + C_i = uj wk - uk wj + uj vk - uk vj
    # = uj (wk + vk) - uk (wj + vj)
    # = uj (1 - uj) - uk (1 - uk)  ... since wk+vk = 1-uj? No, wk = 1-uk-vk.
    # wk + vk = 1 - uk.
    # wj + vj = 1 - uj.
    # So B_i + C_i = uj(1-uk) - uk(1-uj) = uj - uj uk - uk + uj uk = uj - uk.
    # So -(B_i + C_i) = uk - uj.
    # My new formula: 2(vk uj - vj uk).
    # Are they equal? Generally no.
    
    # Let's re-evaluate grad_v for point i.
    # grad_v (component 2 of cross product Q x R) = vk * uj - vj * uk.
    # grad_w (component 3) = vj * uk - vk * uj.
    # Note grad_w = -grad_v.
    # df/dv_i = grad_v - grad_w = grad_v - (-grad_v) = 2 * grad_v = 2(vk uj - vj uk).
    
    # So:
    # d(det)/du_i = A_i - C_i
    # d(det)/dv_i = 2 * (vk * uj - vj * uk)
    
    # Similarly for j:
    # Q = P_i, R = P_k.
    # grad_u = vi wk - vk wi
    # grad_v = vk ui - vi uk
    # grad_w = vi uk - vk ui = -grad_v
    # df/du_j = grad_u - grad_w = (vi wk - vk wi) + (vk ui - vi uk)
    # df/dv_j = 2 * grad_v = 2 * (vk ui - vi uk)
    
    # For k:
    # Q = P_i, R = P_j.
    # grad_u = vi wj - vj wi
    # grad_v = vj ui - vi uj
    # grad_w = vi uj - vj ui = -grad_v
    # df/du_k = (vi wj - vj wi) + (vj ui - vi uj)
    # df/dv_k = 2 * (vj ui - vi uj)
    
    # Let's implement this.
    
    # Precompute terms for all triangles
    # For i:
    d_det_d_ui = A_i - C_i
    d_det_d_vi = 2.0 * (vk * uj - vj * uk)
    
    # For j:
    d_det_d_uj = (vi * wk - vk * wi) + (vk * ui - vi * uk)
    d_det_d_vj = 2.0 * (vk * ui - vi * uk)
    
    # For k:
    d_det_d_uk = (vi * wj - vj * wi) + (vj * ui - vi * uj)
    d_det_d_vk = 2.0 * (vj * ui - vi * uj)
    
    # Areas are |det|.
    # Sign of det:
    det_vals = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    signs = np.sign(det_vals)
    # If det < 0, gradient is negated.
    signs = np.where(signs == 0, 1.0, signs)
    
    # Scatter into J
    # J[row, 2*i_idx] += signs * d_det_d_ui
    J[np.arange(len(tri_idx)), 2 * i_idx] += signs * d_det_d_ui
    J[np.arange(len(tri_idx)), 2 * i_idx + 1] += signs * d_det_d_vi
    
    J[np.arange(len(tri_idx)), 2 * j_idx] += signs * d_det_d_uj
    J[np.arange(len(tri_idx)), 2 * j_idx + 1] += signs * d_det_d_vj
    
    J[np.arange(len(tri_idx)), 2 * k_idx] += signs * d_det_d_uk
    J[np.arange(len(tri_idx)), 2 * k_idx + 1] += signs * d_det_d_vk
    
    return J, det_vals, signs

def active_set_newton_step(bary, t, active_indices, max_iter=20, tol=1e-8):
    """
    Perform Newton steps to maximize t subject to area_i >= t for active i.
    We solve the KKT system for the equality constrained problem:
    Maximize t
    Subject to: area_i(x) = t for i in active
                x in Triangle
    
    Lagrangian: L = -t + sum lambda_i (t - area_i(x))
    Grad_x L = - sum lambda_i grad(area_i) = 0
    Grad_t L = -1 + sum lambda_i = 0  => sum lambda_i = 1
    
    This is a system of equations.
    Let G be the matrix of gradients of active areas w.r.t x (size m x 22).
    We want to find dx and dt such that:
    G dx - dt * 1 = 0  (linearized area change equals dt)
    And we want to move in the direction that increases t.
    
    Actually, simpler:
    We want to find a direction (dx, dt) such that:
    d(area_i)/dx * dx = dt  for all active i.
    This is an underdetermined system (22 vars, m eqs).
    We want to maximize dt.
    This is equivalent to finding the component of the gradient of t in the null space of the constraints?
    
    Alternative: Use SLSQP on the active set with high precision.
    """
    x = bary.flatten()
    t_val = t
    for _ in range(max_iter):
        J, det_vals, signs = compute_jacobian(bary)
        J_active = J[active_indices, :]
        
        # We want to solve: J_active @ dx = dt
        # And maximize dt.
        # This is a linear programming problem? No, dx is bounded by the triangle constraints.
        
        # Let's use a least squares approach with a regularization to find a feasible direction.
        # Or just use scipy minimize on the active set constraints.
        
        # Define objective: -t
        # Constraints: area_i - t >= 0 for active i.
        
        def obj(z):
            b = z[:N*2].reshape(N, 2)
            b = clamp_bary(b)
            tv = z[N*2]
            areas = calc_areas_bary(b)
            # Penalty for violating active constraints
            viol = np.max(0, tv - areas[active_indices])
            return -tv + 1e6 * viol**2
        
        def cons(z):
            b = z[:N*2].reshape(N, 2)
            b = clamp_bary(b)
            tv = z[N*2]
            areas = calc_areas_bary(b)
            return areas[active_indices] - tv
        
        z0 = np.concatenate([bary.flatten(), [t_val]])
        
        try:
            res = minimize(
                obj, z0, method='SLSQP', bounds=[(0,1)]*22 + [(0, 0.1)],
                constraints=[{'type': 'ineq', 'fun': cons}],
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            if res.success:
                new_bary = clamp_bary(res.x[:N*2].reshape(N, 2))
                new_t = res.x[N*2]
                if new_t > t_val + 1e-10:
                    bary = new_bary
                    t_val = new_t
                    # Update active set
                    areas = calc_areas_bary(bary)
                    min_area = np.min(areas)
                    active_indices = np.where(areas <= min_area + 1e-6)[0]
                else:
                    break
            else:
                break
        except Exception:
            break
            
    return bary, t_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Initial guess: Hexagonal packing
    # 3 corners, 3 edge midpoints, 3 inner ring, 1 center?
    # 11 points.
    # Let's use a structured init.
    inits = []
    
    # Init 1: Grid
    grid = []
    for i in range(4):
        for j in range(3):
            if i + j < 4:
                grid.append([i/3.0, j/3.0])
    # This gives 6 points. Need 11.
    
    # Let's use the hexagonal init from inspiration 1
    hex_init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.0, 0.33],
        [0.0, 0.67],
        [0.17, 0.5],
        [0.5, 0.17],
        [0.33, 0.33],
        [0.5, 0.33],
    ])
    inits.append(clamp_bary(hex_init))
    
    # Init 2: Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N)
        # Ensure u+v < 1
        s = u + v
        mask = s > 1
        if np.any(mask):
            scale = 1.0 / s[mask]
            u[mask] *= scale
            v[mask] *= scale
        inits.append(clamp_bary(np.column_stack([u, v])))
        
    # Run Active Set Newton for each init
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        current_bary = init.copy()
        current_min = get_min_area(current_bary)
        
        # Iterative Active Set Refinement
        for iter_num in range(50):
            if time.time() > deadline - 10:
                break
                
            areas = calc_areas_bary(current_bary)
            min_area = np.min(areas)
            # Identify active triangles (those within 1e-5 of min)
            active_indices = np.where(areas <= min_area + 1e-5)[0]
            
            if len(active_indices) == 0:
                break
                
            # Try to improve using Active Set Newton
            new_bary, new_min = active_set_newton_step(current_bary, min_area, active_indices)
            
            if new_min > current_min + 1e-10:
                current_bary = new_bary
                current_min = new_min
            else:
                # If no improvement, try a small perturbation to escape local minimum
                # and re-evaluate
                if iter_num % 5 == 0:
                    perturbed = current_bary + rng.normal(0, 0.05, size=current_bary.shape)
                    perturbed = clamp_bary(perturbed)
                    p_min = get_min_area(perturbed)
                    if p_min > current_min:
                        current_bary = perturbed
                        current_min = p_min
                    else:
                        # Try to drop a constraint? Or just break if stuck for too long
                        if iter_num > 20:
                            break
        
        update_best(current_bary)
        
    # Final polish with generic L-BFGS-B if time permits
    if best_bary is not None and time.time() < deadline - 5:
        def soft_obj(x):
            b = x.reshape(N, 2)
            b = clamp_bary(b)
            areas = calc_areas_bary(b)
            if np.any(areas < 1e-15):
                return 1e10
            beta = 100.0
            scaled = -beta * areas
            max_scaled = np.max(scaled)
            if max_scaled > 700:
                return -np.min(areas)
            log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            soft_min = -log_sum / beta
            return -soft_min
        
        try:
            res = minimize(
                soft_obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0,1)]*(N*2),
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            new_bary = clamp_bary(res.x.reshape(N, 2))
            update_best(new_bary)
        except Exception:
            pass

    if best_bary is None:
        best_bary = inits[0]

    cart = bary_to_cart(best_bary)
    return cart

if __name__ == "__main__":
    res = construct()
    print(res)
