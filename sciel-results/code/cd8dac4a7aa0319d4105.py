import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 145.0

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
M = len(tri_indices)
tri_idx = np.array(tri_indices)  # (M, 3)

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

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

def generate_hexagonal_init(seed=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    
    # Grid 4x4 in barycentric?
    grid_pts = []
    for i in range(5):
        for j in range(5):
            u = i * 0.25
            v = j * 0.25
            if u + v <= 1.0 + 1e-9:
                grid_pts.append([u, v])
            
    selected_indices = [0, 4, 14, 2, 9, 11, 6, 7, 10, 3, 8]
    bary = np.array([grid_pts[i] for i in selected_indices])
    
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

    # Phase 1: Initial setup
    inits = []
    inits.append(generate_hexagonal_init(0))
    for i in range(2):
        inits.append(generate_random_init(100 + i))
        
    # Pre-compute indices for Jacobian
    # tri_idx is (M, 3). 
    # We need to build the constraint Jacobian J of shape (M, N*2 + 1)
    # The constraint is: Area_i - t >= 0
    # d(Area_i)/dx is non-zero only for the 3 points involved in triangle i.
    
    # Let's define the objective and constraints for SLSQP
    # x = [u0, v0, ..., u10, v10, t]
    
    def objective(x):
        return -x[N*2] # Minimize -t, i.e., Maximize t

    def objective_grad(x):
        grad = np.zeros(N*2 + 1)
        grad[N*2] = -1.0
        return grad

    def constraints(x):
        bary = x[:N*2].reshape(N, 2)
        t = x[N*2]
        
        # Calculate signed determinants to handle gradients correctly
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
        
        # Signed determinant
        det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
        
        # Area is |det|. 
        # Constraint: |det| - t >= 0
        # This is equivalent to (det - t >= 0) AND (-det - t >= 0) ? No.
        # |det| >= t is equivalent to det^2 - t^2 >= 0? No, that's non-linear in t.
        # Actually, |det| >= t  <=>  det >= t  AND  -det >= t  is WRONG.
        # |det| >= t  <=>  (det >= t) OR (-det >= t).
        # Standard SLSQP handles "f(x) >= 0".
        # If we use det - t >= 0, we force orientation.
        # If we use -det - t >= 0, we force opposite orientation.
        # We don't know the orientation.
        
        # Alternative: Use det^2 - t^2 >= 0? 
        # If det^2 >= t^2 and t>0, then |det| >= t.
        # This is smooth!
        # Let's check gradient of det^2 - t^2.
        # d/dx (det^2) = 2 * det * d(det)/dx
        # d/dt (-t^2) = -2t
        
        # However, SLSQP with squared constraints can be tricky if we cross zero.
        # Let's stick to the standard approach: 
        # We can define the constraint as:
        # min(det, -det) - t >= 0 ? No, min is non-differentiable.
        
        # Let's go back to the idea:
        # We can split the problem? No.
        
        # Let's use the property that Area is the absolute value.
        # We can define the constraint function to return the *signed* determinant 
        # if we know the sign, but we don't.
        
        # Wait, for a fixed set of points, the sign of the determinant is fixed 
        # unless the points become collinear. 
        # We can compute the sign once at the start of the constraint evaluation 
        # and use the appropriate linear form?
        # No, the sign can flip if the optimizer moves points across a line.
        
        # Let's use the smooth approximation: 
        # Constraint: det^2 - t^2 >= 0
        # This ensures |det| >= t.
        
        # Let's calculate the Jacobian for J_det_sq = 2 * det * J_det
        # And J_t_sq = -2 * t
        
        # This is a valid structural change: using squared constraints to handle absolute value smoothly.
        
        # Compute Jacobian of det w.r.t x
        # det = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j)
        # Note w = 1 - u - v.
        
        # Let's pre-compute the structure.
        # For each triangle m (row in tri_idx):
        # p, q, r = tri_idx[m]
        
        # We will build the Jacobian matrix J of shape (M, N*2+1)
        # J[m, :] is the gradient of (det_m^2 - t^2) w.r.t x
        
        # It is more efficient to compute the gradient of det first, then multiply by 2*det.
        
        # Gradient of det_m w.r.t u_p, v_p, u_q, v_q, u_r, v_r
        # Let's derive d(det)/du_p, etc.
        
        # det = u_p (v_q w_r - v_r w_q) - v_p (u_q w_r - u_r w_q) + w_p (u_q v_r - u_r v_q)
        
        # Terms involving u_p:
        # d/du_p [u_p (v_q w_r - v_r w_q)] = (v_q w_r - v_r w_q)
        # d/du_p [-v_p (u_q w_r - u_r w_q)] = 0
        # d/du_p [w_p (u_q v_r - u_r v_q)] = dw_p/du_p * (u_q v_r - u_r v_q) = -1 * (u_q v_r - u_r v_q)
        
        # So d(det)/du_p = (v_q w_r - v_r w_q) - (u_q v_r - u_r v_q)
        
        # Similarly for v_p:
        # d/dv_p [u_p (v_q w_r - v_r w_q)] = 0
        # d/dv_p [-v_p (u_q w_r - u_r w_q)] = -(u_q w_r - u_r w_q)
        # d/dv_p [w_p (u_q v_r - u_r v_q)] = dw_p/dv_p * (u_q v_r - u_r v_q) = -1 * (u_q v_r - u_r v_q)
        
        # So d(det)/dv_p = -(u_q w_r - u_r w_q) - (u_q v_r - u_r v_q)
        
        # Let's verify this logic.
        # It seems correct.
        
        # We need to fill J[m, 2*p] with 2*det_m * d(det_m)/du_p
        # And J[m, 2*p+1] with 2*det_m * d(det_m)/dv_p
        
        # And J[m, N*2] with -2*t
        
        # Let's implement this efficiently using vectorized operations if possible,
        # or a loop over M (165) which is small enough.
        
        J = np.zeros((M, N*2 + 1))
        
        # Pre-compute components for all triangles
        # We need u_p, v_p, w_p, etc. for p, q, r
        
        p_idx = tri_idx[:, 0]
        q_idx = tri_idx[:, 1]
        r_idx = tri_idx[:, 2]
        
        up = u[p_idx]
        vp = v[p_idx]
        wp = w[p_idx]
        
        uq = u[q_idx]
        vq = v[q_idx]
        wq = w[q_idx]
        
        ur = u[r_idx]
        vr = v[r_idx]
        wr = w[r_idx]
        
        # Terms for gradient
        # A = v_q w_r - v_r w_q
        A = vq * wr - vr * wq
        # B = u_q w_r - u_r w_q
        B = uq * wr - ur * wq
        # C = u_q v_r - u_r v_q
        C = uq * vr - ur * vq
        
        # d(det)/du_p = A - C
        d_det_dup = A - C
        # d(det)/dv_p = -B - C
        d_det_dvp = -B - C
        
        # By symmetry (cyclic permutation p->q->r->p)
        # d(det)/du_q = B - A  (Wait, let's re-derive or check symmetry)
        # det is antisymmetric.
        # Let's just calculate for q and r explicitly to be safe, or use symmetry.
        # d(det)/du_q:
        # Term 1: u_p (v_q w_r - v_r w_q) -> d/du_q is 0? No, u_q is not in term 1? 
        # Wait, term 1 is u_p * (v_q w_r - v_r w_q). u_q is not there.
        # Term 2: -v_p (u_q w_r - u_r w_q). d/du_q = -v_p * w_r
        # Term 3: w_p (u_q v_r - u_r v_q). d/du_q = w_p * v_r
        # So d(det)/du_q = -v_p w_r + w_p v_r = w_p v_r - v_p w_r
        
        # Let's check if this matches "B - A" or similar.
        # B = u_q w_r - u_r w_q. No.
        
        # Let's just compute all 9 partials for each triangle? 
        # M=165, 9*165 = 1485 operations. Very fast.
        
        # Let's compute d_det_du_q, d_det_dv_q, d_det_du_r, d_det_dv_r
        
        # d(det)/du_q = -vp * wr + wp * vr
        d_det_duq = -vp * wr + wp * vr
        # d(det)/dv_q = -vp * wr? No.
        # Term 1: u_p (v_q w_r - v_r w_q). d/dv_q = u_p * w_r
        # Term 2: -v_p (u_q w_r - u_r w_q). d/dv_q = 0
        # Term 3: w_p (u_q v_r - u_r v_q). d/dv_q = -w_p * u_r
        # So d(det)/dv_q = u_p * w_r - w_p * u_r
        
        d_det_dvq = up * wr - wp * ur
        
        # d(det)/du_r = ?
        # Term 1: u_p (v_q w_r - v_r w_q). d/du_r = 0
        # Term 2: -v_p (u_q w_r - u_r w_q). d/du_r = -v_p * (-w_q) = v_p w_q
        # Term 3: w_p (u_q v_r - u_r v_q). d/du_r = -w_p * v_q
        # So d(det)/du_r = v_p w_q - w_p v_q
        
        d_det_dur = vp * wq - wp * vq
        
        # d(det)/dv_r = ?
        # Term 1: u_p (v_q w_r - v_r w_q). d/dv_r = u_p * (-w_q) = -u_p w_q
        # Term 2: -v_p (u_q w_r - u_r w_q). d/dv_r = 0
        # Term 3: w_p (u_q v_r - u_r v_q). d/dv_r = w_p * u_q
        # So d(det)/dv_r = -u_p w_q + w_p u_q
        
        d_det_dvr = -up * wq + wp * uq
        
        # Now we have the gradients of det for all points involved.
        # We need to scatter these into the Jacobian J.
        
        # J[m, 2*p] += 2 * det[m] * d_det_dup[m]
        # J[m, 2*p+1] += 2 * det[m] * d_det_dvp[m]
        # ... etc
        
        # Use np.add.at for accumulation if indices are not unique? 
        # Here indices are unique per row m, so we can just assign.
        
        # But p, q, r are arrays of indices.
        # We need to map these to columns in J.
        
        # col_p = 2 * p_idx
        # col_p_v = 2 * p_idx + 1
        
        J[:, 2 * p_idx] = 2 * det * d_det_dup
        J[:, 2 * p_idx + 1] = 2 * det * d_det_dvp
        
        J[:, 2 * q_idx] = 2 * det * d_det_duq
        J[:, 2 * q_idx + 1] = 2 * det * d_det_dvq
        
        J[:, 2 * r_idx] = 2 * det * d_det_dur
        J[:, 2 * r_idx + 1] = 2 * det * d_det_dvr
        
        # Gradient w.r.t t
        J[:, N*2] = -2 * t
        
        # The constraint function value is det^2 - t^2
        # We return the constraint values and the Jacobian
        return det**2 - t**2, J

    # SLSQP requires constraints as a list of dicts
    # Or a single dict if it's a vector constraint
    # We have M constraints.
    
    constraints = {'type': 'ineq', 'fun': constraints}
    
    bounds = [(0, 1)] * (N*2) + [(0, 0.1)] # t starts small, max area is ~0.25/11? No, max min area is small. 0.1 is safe upper bound.
    
    best_result = None
    best_fun = np.inf
    
    # Run SLSQP from multiple starts
    for init_bary in inits:
        if time.time() > deadline - 10:
            break
            
        x0 = np.concatenate([init_bary.flatten(), [0.01]]) # Start with small t
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            # Check if result is valid
            if res.success:
                t_val = res.x[N*2]
                bary_res = res.x[:N*2].reshape(N, 2)
                bary_res = clamp_bary(bary_res)
                
                # Verify feasibility
                areas = calc_areas_bary(bary_res)
                min_area = np.min(areas)
                
                if min_area >= t_val - 1e-6: # Feasible
                    if t_val > best_fun: # We are minimizing -t, so lower fun is better? 
                        # objective is -t. So we want min fun.
                        # But we track best_val (max area).
                        if t_val > best_val:
                            best_val = t_val
                            best_bary = bary_res
                            best_fun = res.fun
                            best_result = res
            else:
                # Even if not "success", check the point
                bary_res = res.x[:N*2].reshape(N, 2)
                bary_res = clamp_bary(bary_res)
                areas = calc_areas_bary(bary_res)
                min_area = np.min(areas)
                if min_area > best_val:
                    best_val = min_area
                    best_bary = bary_res
                    
        except Exception as e:
            pass

    # Fallback
    if best_bary is None:
        best_bary = generate_random_init(0)
        
    cart = bary_to_cart(best_bary)
    return cart
