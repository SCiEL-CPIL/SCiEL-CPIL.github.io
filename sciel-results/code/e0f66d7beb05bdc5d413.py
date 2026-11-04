"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area.

Strategy: Use a smooth log-sum-exp surrogate for the minimum area, optimized via L-BFGS-B with
analytical gradients, combined with multi-start, simulated annealing for global exploration,
and a final Nelder-Mead polish on the exact non-smooth objective.
"""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
S3 = np.sqrt(3.0)
H = S3 / 2.0
TRI_AREA = 0.5 * H

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all C(11,3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]
M = len(_IDX_TRIPLES)  # 165

def _bary_to_cart(bary):
    """Convert barycentric coords (l0, l1, l2) shape (N,3) to Cartesian (N,2)."""
    l1 = bary[:, 1:2]
    l2 = bary[:, 2:3]
    x = l1 * V1[0] + l2 * V2[0]
    y = l1 * V1[1] + l2 * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian (N,2) to barycentric (N,3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    l2 = 2.0 * y / S3
    l1 = x - 0.5 * l2
    l0 = 1.0 - l1 - l2
    return np.column_stack([l0, l1, l2])

def _min_area_from_bary(bary):
    """Compute min triangle area from barycentric coordinates."""
    pts = _bary_to_cart(bary)
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _min_triangle_area(pts):
    """Compute the smallest triangle area among all triples."""
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_to_simplex(bary):
    """Project barycentric coordinates onto the valid simplex (l_i >= 0, sum l_i = 1)."""
    # Clip negative values and renormalize
    bary = bary.copy()
    bary = np.maximum(bary, 0.0)
    row_sums = bary.sum(axis=1, keepdims=True)
    # Handle rows that sum to zero
    zero_mask = (row_sums[:, 0] < 1e-15)
    row_sums[zero_mask] = 1.0
    bary = bary / row_sums
    # Ensure they sum to 1
    bary[:, 0] = 1.0 - bary[:, 1] - bary[:, 2]
    return bary

def _objective_smooth(bary, beta=50.0):
    """
    Smooth surrogate: -softmin(areas) where softmin uses log-sum-exp.
    bary: (N, 3) barycentric coordinates.
    Returns scalar (negative of soft-min area).
    """
    pts = _bary_to_cart(bary)
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    # For numerical stability
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_a - log_sum
    return -soft_min

def _objective_smooth_grad(bary, beta=50.0):
    """
    Gradient of -softmin(areas) w.r.t. barycentric coordinates.
    Returns (N, 3) gradient.
    """
    pts = _bary_to_cart(bary)
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    
    dx_i = pj[:, 0] - pi[:, 0]
    dy_i = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    
    cross = dx_i * dy_k - dy_i * dx_k
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    
    # d(soft_min)/d(area_m) = -exp_term[m] / sum_exp
    d_softmin_d_area = -exp_term / sum_exp
    
    sign = np.sign(cross)
    d_area_d_cross = 0.5 * sign
    
    # Gradient of cross w.r.t. each point
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d(cross)/d(x_i) = -(y_k - y_i) = -dy_k
    # d(cross)/d(y_i) = (x_k - x_i) = dx_k
    # d(cross)/d(x_j) = (y_k - y_i) = dy_k
    # d(cross)/d(y_j) = -(x_k - x_i) = -dx_k
    # d(cross)/d(x_k) = -(y_j - y_i) = -dy_i
    # d(cross)/d(y_k) = (x_j - x_i) = dx_i
    
    # Gradient w.r.t. Cartesian points (N, 2)
    grad_pts = np.zeros((N, 2))
    
    # Contribution from point i (pi)
    contrib_i = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * np.column_stack([-dy_k, dx_k])
    np.add.at(grad_pts, _IDX_I, contrib_i)
    
    # Contribution from point j (pj)
    contrib_j = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * np.column_stack([dy_k, -dx_k])
    np.add.at(grad_pts, _IDX_J, contrib_j)
    
    # Contribution from point k (pk)
    contrib_k = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * np.column_stack([-dy_i, dx_i])
    np.add.at(grad_pts, _IDX_K, contrib_k)
    
    # Map Cartesian gradient to barycentric gradient
    # x = l1 * 1 + l2 * 0.5, y = l1 * 0 + l2 * H
    # dx/dl1 = 1, dx/dl2 = 0.5
    # dy/dl1 = 0, dy/dl2 = H
    # dl0 = -dl1 - dl2, so d/dl0 = -(d/dl1 + d/dl2)
    
    grad_l1 = grad_pts[:, 0] * 1.0 + grad_pts[:, 1] * 0.0
    grad_l2 = grad_pts[:, 0] * 0.5 + grad_pts[:, 1] * H
    grad_l0 = -(grad_l1 + grad_l2)
    
    grad_bary = np.column_stack([grad_l0, grad_l1, grad_l2])
    
    # Objective is -soft_min, so gradient is -grad_softmin
    return -grad_bary

def _objective_exact(bary):
    """Exact objective: negative of min triangle area."""
    return -_min_area_from_bary(bary)

def _generate_fps_initialization(rng, n_points=N, n_candidates=300):
    """Generate initial configuration using Farthest Point Sampling in barycentric space."""
    # Start with centroid
    bary = np.zeros((n_points, 3))
    bary[0] = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    
    for k in range(1, n_points):
        # Generate random candidates in the triangle
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        cand_bary = np.column_stack([1 - r1 - r2, r1, r2])
        
        # Compute distances to existing points (in Cartesian space for geometric meaning)
        cand_pts = _bary_to_cart(cand_bary)
        existing_pts = _bary_to_cart(bary[:k])
        
        diffs = cand_pts[:, np.newaxis, :] - existing_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        bary[k] = cand_bary[best_idx]
    
    return bary

def _structured_init(rng):
    """Generate a structured/symmetric initial configuration."""
    bary = np.zeros((N, 3))
    
    # Try to create a roughly grid-like pattern
    # 11 points: maybe 3-4-3-1 or some arrangement
    # Use a few different structured patterns
    
    # Pattern: 4 on each edge midpoint region + 3 interior
    # Simpler: just use a jittered grid
    positions = []
    # 3x3 grid minus corners = 5, plus 6 more
    # Let's just do a random structured approach
    
    # 11 points in a triangular arrangement
    # Row 0: 1 point (top)
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 3 points
    # Row 4: 2 points
    # Total: 11
    
    rows = [1, 2, 3, 3, 2]
    idx = 0
    for row_i, count in enumerate(rows):
        # Barycentric: l2 increases with row (top to bottom)
        l2 = (row_i + 0.5) / 5.0  # Spread over the triangle height
        for j in range(count):
            # l1 varies along the row
            if count == 1:
                l1 = 0.5
            else:
                l1 = (j + 0.5) / count
            # Adjust for the triangle shape: at height l2, the width is (1-l2)
            # l1 should be scaled: actual l1 = l1 * (1 - l2) + l2 * 0.5
            # Wait, in barycentric: x = l1 + 0.5*l2, y = H*l2
            # At height l2, x ranges from 0.5*l2 to 1 - 0.5*l2
            # So l1 ranges from 0 to 1 - l2
            l1_actual = l1 * (1.0 - l2)
            l0 = 1.0 - l1_actual - l2
            
            # Add small jitter
            jitter = 0.03
            l0 += rng.uniform(-jitter, jitter)
            l1_actual += rng.uniform(-jitter, jitter)
            l2 += rng.uniform(-jitter, jitter)
            
            # Normalize
            s = l0 + l1_actual + l2
            l0 /= s
            l1_actual /= s
            l2 /= s
            
            # Clamp
            l0 = max(0, l0)
            l1_actual = max(0, l1_actual)
            l2 = max(0, l2)
            s = l0 + l1_actual + l2
            l0 /= s
            l1_actual /= s
            l2 /= s
            
            bary[idx] = [l0, l1_actual, l2]
            idx += 1
    
    return bary

def _local_optimize_lbfgs(bary, deadline, beta=50.0):
    """Local optimization using L-BFGS-B on the smooth objective."""
    # Flatten barycentric coordinates: 3*N variables
    x0 = bary.flatten()
    
    def obj_flat(x):
        b = x.reshape(N, 3)
        b = _project_to_simplex(b)
        return _objective_smooth(b, beta)
    
    def grad_flat(x):
        b = x.reshape(N, 3)
        b = _project_to_simplex(b)
        g = _objective_smooth_grad(b, beta)
        return g.flatten()
    
    bounds = [(0.0, 1.0)] * (3 * N)
    
    try:
        res = minimize(
            obj_flat, x0, jac=grad_flat, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        bary_res = _project_to_simplex(res.x.reshape(N, 3))
        score = _min_area_from_bary(bary_res)
        return bary_res, score
    except Exception:
        return bary, _min_area_from_bary(bary)

def _local_optimize_nm(bary, deadline):
    """Local optimization using Nelder-Mead on the exact objective."""
    x0 = bary.flatten()
    
    def obj_flat(x):
        b = x.reshape(N, 3)
        b = _project_to_simplex(b)
        return _objective_exact(b)
    
    try:
        res = minimize(
            obj_flat, x0, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_res = _project_to_simplex(res.x.reshape(N, 3))
        score = _min_area_from_bary(bary_res)
        return bary_res, score
    except Exception:
        return bary, _min_area_from_bary(bary)

def _simulated_annealing(bary, rng, deadline, n_iterations=10000):
    """Simulated annealing to escape local minima."""
    best_bary = bary.copy()
    best_score = _min_area_from_bary(bary)
    current_bary = bary.copy()
    current_score = _min_area_from_bary(bary)
    
    T = 0.02
    cooling_rate = 0.9998
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Perturb in barycentric space
        noise = rng.normal(0, T, size=(N, 3))
        new_bary = current_bary + noise
        new_bary = _project_to_simplex(new_bary)
        
        new_score = _min_area_from_bary(new_bary)
        
        # Accept if better or with probability exp((new - current) / T)
        if new_score > current_score:
            current_bary = new_bary
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bary = new_bary.copy()
        else:
            delta = (new_score - current_score) / T
            if delta > -50:  # Avoid underflow
                if rng.random() < np.exp(delta):
                    current_bary = new_bary
                    current_score = new_score
        
        T *= cooling_rate
    
    return best_bary, best_score

def _active_set_lp(bary, deadline, k=20):
    """
    Active-set LP approach: identify the k smallest triangles, then solve an LP
    that maximizes t subject to those k triangles having area >= t, plus simplex constraints.
    This exploits the piecewise-linear structure.
    """
    pts = _bary_to_cart(bary)
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Get the k smallest
    k_actual = min(k, M)
    smallest_idx = np.argsort(areas)[:k_actual]
    
    # Determine orientation for each of these triangles
    # cross > 0 means CCW, cross < 0 means CW
    signs = np.sign(cross[smallest_idx])
    # If cross == 0, assign +1
    signs[signs == 0] = 1.0
    
    # LP variables: x_i, y_i for i in 0..N-1, and t
    # Total: 2*N + 1 variables
    n_vars = 2 * N + 1
    
    # Objective: maximize t => minimize -t
    c = np.zeros(n_vars)
    c[-1] = -1.0
    
    A_ub = []
    b_ub = []
    
    # Triangle containment constraints (in Cartesian)
    # y >= 0 => -y <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i + 1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # y <= sqrt(3)*x => y - sqrt(3)*x <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = -S3
        row[2*i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # y <= sqrt(3)*(1-x) => y + sqrt(3)*x <= sqrt(3)
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = S3
        row[2*i + 1] = 1.0
        A_ub.append(row)
        b_ub.append(S3)
    
    # x >= 0 => -x <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # x <= 1
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
    
    # Area constraints for the k smallest triangles
    # For triangle (i, j, k) with sign s:
    # s * cross >= 2*t  =>  -s*cross + 2*t <= 0
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # This is bilinear, NOT linear. So we can't directly use LP.
    # 
    # Alternative: use barycentric coordinates.
    # In barycentric, the area of triangle (i,j,k) is:
    # area = 0.5 * |det of the 3x3 matrix of barycentric coords| * TRI_AREA
    # Actually, the area ratio is |det([bary_i; bary_j; bary_k])|
    # where bary_i = (l0_i, l1_i, l2_i)
    #
    # The determinant of the 3x3 matrix with rows (l0_i, l1_i, l2_i), (l0_j, l1_j, l2_j), (l0_k, l1_k, l2_k)
    # is a trilinear function, not linear.
    #
    # So LP doesn't directly work. Let's use a different approach.
    # 
    # Instead, let's use the fact that if we fix the orientation, the area is:
    # area = 0.5 * s * cross
    # and we want area >= t, i.e., s * cross >= 2t
    # But cross is bilinear in the coordinates.
    #
    # Let's just use SLSQP with the active set constraints.
    
    # Use SLSQP with the k active constraints
    x0 = bary.flatten()
    
    # Convert to Cartesian for constraint evaluation
    def _area_constraint(m_idx):
        i, j, k = _IDX_TRIPLES[m_idx]
        sign = signs[np.where(smallest_idx == m_idx)[0][0]] if m_idx in smallest_idx else 1.0
        
        def constraint(x):
            b = x.reshape(N, 3)
            b = _project_to_simplex(b)
            pts = _bary_to_cart(b)
            pi = pts[i]
            pj = pts[j]
            pk = pts[k]
            cross_val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
            area_val = 0.5 * sign * cross_val
            return area_val  # Should be >= 0 (we'll handle the t separately)
        
        return constraint
    
    # Actually, let's just use a simpler approach: Nelder-Mead on the exact objective
    # from the current best, with many restarts.
    
    return None

def _ga_population_init(rng, pop_size=20):
    """Generate a population of initial configurations."""
    population = []
    
    # FPS initializations
    for _ in range(10):
        bary = _generate_fps_initialization(rng)
        population.append(bary)
    
    # Structured
    for _ in range(3):
        bary = _structured_init(rng)
        population.append(bary)
    
    # Random
    for _ in range(pop_size - 13):
        bary = np.zeros((N, 3))
        for i in range(N):
            r1 = rng.uniform(0, 1)
            r2 = rng.uniform(0, 1)
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            bary[i] = [1 - r1 - r2, r1, r2]
        population.append(bary)
    
    return population

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_bary = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Multi-start with L-BFGS-B on smooth objective
    population = _ga_population_init(rng, pop_size=25)
    
    for bary_init in population:
        if time_left() < 60.0:
            break
        
        bary_opt, score = _local_optimize_lbfgs(bary_init, deadline)
        if score > best_score:
            best_score = score
            best_bary = bary_opt.copy()
    
    # Phase 2: Nelder-Mead polish on the best
    if best_bary is not None and time_left() > 40.0:
        bary_nm, score_nm = _local_optimize_nm(best_bary, deadline)
        if score_nm > best_score:
            best_score = score_nm
            best_bary = bary_nm.copy()
    
    # Phase 3: Simulated annealing to escape local minima
    if best_bary is not None and time_left() > 30.0:
        bary_sa, score_sa = _simulated_annealing(best_bary, rng, deadline, n_iterations=15000)
        if score_sa > best_score:
            best_score = score_sa
            best_bary = bary_sa.copy()
    
    # Phase 4: L-BFGS-B polish after SA
    if best_bary is not None and time_left() > 20.0:
        bary_opt, score = _local_optimize_lbfgs(best_bary, deadline, beta=100.0)
        if score > best_score:
            best_score = score
            best_bary = bary_opt.copy()
    
    # Phase 5: Nelder-Mead final polish
    if best_bary is not None and time_left() > 10.0:
        bary_nm, score_nm = _local_optimize_nm(best_bary, deadline)
        if score_nm > best_score:
            best_score = score_nm
            best_bary = bary_nm.copy()
    
    # Phase 6: More SA restarts from perturbed best
    if best_bary is not None and time_left() > 20.0:
        for i in range(5):
            if time_left() < 10.0:
                break
            noise = rng.normal(0, 0.05, size=(N, 3))
            perturbed = _project_to_simplex(best_bary + noise)
            bary_sa, score_sa = _simulated_annealing(perturbed, rng, deadline, n_iterations=5000)
            if score_sa > best_score:
                best_score = score_sa
                best_bary = bary_sa.copy()
            # Quick L-BFGS polish
            if time_left() > 10.0:
                bary_opt, score = _local_optimize_lbfgs(bary_sa, deadline, beta=80.0)
                if score > best_score:
                    best_score = score
                    best_bary = bary_opt.copy()
    
    # Phase 7: Final high-precision Nelder-Mead
    if best_bary is not None and time_left() > 5.0:
        x0 = best_bary.flatten()
        
        def obj_flat(x):
            b = x.reshape(N, 3)
            b = _project_to_simplex(b)
            return _objective_exact(b)
        
        try:
            res = minimize(
                obj_flat, x0, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            bary_res = _project_to_simplex(res.x.reshape(N, 3))
            score_res = _min_area_from_bary(bary_res)
            if score_res > best_score:
                best_score = score_res
                best_bary = bary_res.copy()
        except Exception:
            pass
    
    # Fallback
    if best_bary is None:
        best_bary = _generate_fps_initialization(rng)
    
    # Final safety: project to valid triangle
    best_bary = _project_to_simplex(best_bary)
    
    # Convert to Cartesian
    best_points = _bary_to_cart(best_bary)
    
    # Verify all points are inside the triangle (with small tolerance)
    for i in range(len(best_points)):
        x, y = best_points[i]
        # Clamp to valid region
        if y < 0:
            y = 0
        if y > H:
            y = H
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        # Left edge: y <= sqrt(3)*x
        if y > S3 * x:
            y = S3 * x
        # Right edge: y <= sqrt(3)*(1-x)
        if y > S3 * (1 - x):
            y = S3 * (1 - x)
        best_points[i] = [x, y]
    
    return np.array(best_points)
