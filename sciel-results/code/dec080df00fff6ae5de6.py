"""Heilbronn triangles: Sequential Convex Programming (LP-based) with multi-start."""
import numpy as np
from scipy.optimize import linprog
import time

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)
DEADLINE_MARGIN = 10.0  # safety margin before 150s kill

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N)
                     for j in range(i + 1, N)
                     for k in range(j + 1, N)])
_NT = _TRIPLES.shape[0]  # 165


def _det_matrix():
    """Build the 165 x 22 matrix M such that det_i = M[i] @ x,
    where x = [u_0, v_0, u_1, v_1, ..., u_10, v_10] in barycentric coords."""
    M = np.zeros((_NT, 2 * N))
    for t_idx, (i, j, k) in enumerate(_TRIPLES):
        # det = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
        # with w = 1 - u - v, expand in terms of u and v:
        # u_i*v_j - u_i*v_k - u_i*u_j + u_i*u_k + v_i*u_j - v_i*u_k - v_i*v_j + v_i*v_k
        # + (1-u_i-v_i)*(u_j*v_k - u_k*v_j)
        # = u_i*v_j - u_i*v_k - u_i*u_j + u_i*u_k + v_i*u_j - v_i*u_k - v_i*v_j + v_i*v_k
        #   + u_j*v_k - u_k*v_j - u_i*u_j*v_k + u_i*u_k*v_j - v_i*u_j*v_k + v_i*u_k*v_j
        # This is quadratic in the variables, so a linear LP on x alone won't work directly.
        # We need a different approach.
        pass
    return M


def _all_dets_from_bary(bary):
    """Compute all 165 determinants (twice the area ratio) from barycentric coords.
    bary: (N, 2) array with columns (u, v), w = 1-u-v."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    i_idx = _TRIPLES[:, 0]
    j_idx = _TRIPLES[:, 1]
    k_idx = _TRIPLES[:, 2]
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det


def _min_triangle_area(points):
    """Compute the minimum triangle area from Cartesian points."""
    bary = _cart_to_bary(points)
    dets = _all_dets_from_bary(bary)
    return 0.5 * np.min(np.abs(dets))


def _bary_to_cart(bary):
    """Convert barycentric (u, v) to Cartesian. Vertex 0=(0,0), 1=(1,0), 2=(0.5,H)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = H * w
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian to barycentric (u, v)."""
    x = points[:, 0]
    y = points[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])


def _project_to_bary_simplex(bary):
    """Project to valid barycentric simplex: u>=0, v>=0, u+v<=1."""
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    points[:, 0] = x
    points[:, 1] = y
    return points


def _scp_iteration(bary, active_set, max_lp_iter=50):
    """
    One SCP iteration: solve LP to maximize t subject to linearized area constraints.
    
    For each triangle i in the active set, we linearize |det_i| around the current bary.
    Let d_i = det_i(current). If d_i > 0, we use det_i >= t. If d_i < 0, we use -det_i >= t.
    For non-active triangles, we use |det_i| >= t (both linearizations).
    
    Since det_i is bilinear in the barycentric variables, we linearize it:
    det_i(new) ≈ det_i(old) + grad_i @ (x_new - x_old)
    
    But det is quadratic, so we use a first-order Taylor approximation.
    """
    x = bary.flatten()  # 22 variables
    dets = _all_dets_from_bary(bary)
    
    # Compute gradients of each det_i with respect to x (22-dim)
    # det_i = sum over permutations ... it's easier to compute numerically or analytically.
    # For a bilinear/quadratic form, we can compute the gradient analytically.
    # 
    # Actually, let's use a simpler approach: since we're doing SCP, we can use
    # the fact that at the current point, the sign of det_i tells us which
    # side of the zero we're on. We linearize the constraint.
    #
    # For triangle i with det_i = D_i (current value):
    #   If D_i > 0: constraint is D_i_new >= t, linearized as D_i + g_i^T (x_new - x) >= t
    #   If D_i < 0: constraint is -D_i_new >= t, linearized as -D_i - g_i^T (x_new - x) >= t
    #
    # We need the gradient g_i of det_i with respect to x.
    #
    # Let's compute gradients numerically for robustness.
    
    # Compute gradients of all 165 dets with respect to 22 variables
    # Using central differences for accuracy
    eps = 1e-6
    grads = np.zeros((_NT, 2 * N))
    for d in range(2 * N):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[d] += eps
        x_minus[d] -= eps
        bary_plus = _project_to_bary_simplex(x_plus.reshape(N, 2))
        bary_minus = _project_to_bary_simplex(x_minus.reshape(N, 2))
        dets_plus = _all_dets_from_bary(bary_plus)
        dets_minus = _all_dets_from_bary(bary_minus)
        grads[:, d] = (dets_plus - dets_minus) / (2 * eps)
    
    # Build LP: maximize t
    # Variables: [x_0, ..., x_21, t] -> 23 variables
    # Objective: minimize -t -> c = [0, ..., 0, -1]
    c = np.zeros(2 * N + 1)
    c[-1] = -1.0  # minimize -t = maximize t
    
    # Constraints: A_ub @ vars <= b_ub
    # For each triangle i:
    #   If active and D_i > 0: D_i + g_i^T x_new - g_i^T x_old >= t
    #     => -g_i^T x_new + t <= D_i - g_i^T x_old
    #   If active and D_i < 0: -D_i - g_i^T x_new + g_i^T x_old >= t
    #     => g_i^T x_new + t <= -D_i + g_i^T x_old
    #   If not active: both |D_i + g_i^T(x_new - x_old)| >= t
    #     => D_i + g_i^T x_new - g_i^T x_old >= t  AND  -(D_i + g_i^T x_new - g_i^T x_old) >= t
    #     => -g_i^T x_new + t <= D_i - g_i^T x_old
    #     => g_i^T x_new + t <= -D_i + g_i^T x_old
    
    A_rows = []
    b_rows = []
    
    for i in range(_NT):
        D_i = dets[i]
        g_i = grads[i]
        g_dot_x = np.dot(g_i, x)
        
        if i in active_set:
            if D_i >= 0:
                # D_i + g_i^T x_new - g_i^T x >= t
                # -g_i^T x_new + t <= D_i - g_i^T x
                row = np.zeros(2 * N + 1)
                row[:2 * N] = -g_i
                row[-1] = 1.0
                A_rows.append(row)
                b_rows.append(D_i - g_dot_x)
            else:
                # -D_i - g_i^T x_new + g_i^T x >= t
                # g_i^T x_new + t <= -D_i + g_i^T x
                row = np.zeros(2 * N + 1)
                row[:2 * N] = g_i
                row[-1] = 1.0
                A_rows.append(row)
                b_rows.append(-D_i + g_dot_x)
        else:
            # Both constraints
            # -g_i^T x_new + t <= D_i - g_i^T x
            row1 = np.zeros(2 * N + 1)
            row1[:2 * N] = -g_i
            row1[-1] = 1.0
            A_rows.append(row1)
            b_rows.append(D_i - g_dot_x)
            
            # g_i^T x_new + t <= -D_i + g_i^T x
            row2 = np.zeros(2 * N + 1)
            row2[:2 * N] = g_i
            row2[-1] = 1.0
            A_rows.append(row2)
            b_rows.append(-D_i + g_dot_x)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= u_i <= 1, 0 <= v_i <= 1, t >= 0
    bounds = [(0, 1)] * (2 * N) + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            x_new = res.x[:2 * N]
            t_new = res.x[-1]
            bary_new = _project_to_bary_simplex(x_new.reshape(N, 2))
            return bary_new, t_new
        else:
            return bary, None
    except Exception:
        return bary, None


def _scp_optimize(bary_init, n_outer=10, active_size=20, time_limit=20.0):
    """
    Full SCP optimization: iterate LP solves, updating active set.
    """
    start = time.time()
    bary = bary_init.copy()
    best_bary = bary.copy()
    best_score = 0.5 * np.min(np.abs(_all_dets_from_bary(bary)))
    
    for outer in range(n_outer):
        if time.time() - start > time_limit:
            break
        
        dets = _all_dets_from_bary(bary)
        abs_dets = np.abs(dets)
        min_val = np.min(abs_dets)
        current_score = 0.5 * min_val
        
        if current_score > best_score:
            best_score = current_score
            best_bary = bary.copy()
        
        # Select active set: triangles with smallest |det|
        active_indices = np.argsort(abs_dets)[:active_size]
        active_set = set(active_indices.tolist())
        
        # Solve LP
        bary_new, t_val = _scp_iteration(bary, active_set)
        
        if bary_new is not None and t_val is not None:
            # Check if the new solution actually improves
            dets_new = _all_dets_from_bary(bary_new)
            score_new = 0.5 * np.min(np.abs(dets_new))
            if score_new >= current_score - 1e-9:
                bary = bary_new
            else:
                # LP didn't help, try with larger active set
                active_set_large = set(np.argsort(abs_dets)[:min(50, _NT)].tolist())
                bary_new2, t_val2 = _scp_iteration(bary, active_set_large)
                if bary_new2 is not None:
                    dets_new2 = _all_dets_from_bary(bary_new2)
                    score_new2 = 0.5 * np.min(np.abs(dets_new2))
                    if score_new2 >= current_score - 1e-9:
                        bary = bary_new2
                # If still no improvement, break
                if score_new < current_score - 1e-9 and (bary_new2 is None or score_new2 < current_score - 1e-9):
                    break
        else:
            break
    
    return best_bary, best_score


def _generate_initializations():
    """Generate diverse initial configurations in barycentric coordinates."""
    inits = []
    
    # 1. FPS-based
    def _fps_bary(n_points=N, seed=42):
        rng = np.random.RandomState(seed)
        bary = [np.array([1/3, 1/3])]
        for _ in range(n_points - 1):
            n_cand = 2000
            r1 = rng.rand(n_cand)
            r2 = rng.rand(n_cand)
            mask = r1 + r2 > 1
            r1[mask] = 1 - r1[mask]
            r2[mask] = 1 - r2[mask]
            cand = np.column_stack([r1, r2])
            pts_arr = np.array(bary)
            # Distance in barycentric space
            diff = cand[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            min_dist_sq = np.min(dist_sq, axis=1)
            best_idx = np.argmax(min_dist_sq)
            bary.append(cand[best_idx])
        return np.array(bary)
    
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        bary = _fps_bary(seed=seed)
        bary = _project_to_bary_simplex(bary)
        inits.append(bary)
    
    # 2. Symmetric configurations
    # Hexagonal lattice
    bary_hex = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.25, 0.25], [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary_hex))
    
    # Concentric
    bary_conc = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [1/3, 0.0], [2/3, 0.0], [0.0, 1/3],
        [0.0, 2/3], [1/3, 1/3], [2/3, 1/3],
        [1/3, 2/3], [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary_conc))
    
    # Grid
    bary_grid = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.25, 0.0], [0.75, 0.0], [0.0, 0.25],
        [0.0, 0.75], [0.25, 0.25], [0.5, 0.25],
        [0.25, 0.5], [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary_grid))
    
    # Random with structured bias
    rng = np.random.RandomState(777)
    for _ in range(5):
        bary = []
        bary.append([0.0, 0.0])
        bary.append([1.0, 0.0])
        bary.append([0.0, 1.0])
        bary.append([1/3, 1/3])
        for _ in range(7):
            u = rng.uniform(0.05, 0.95)
            v = rng.uniform(0.05, 0.95)
            if u + v > 0.9:
                u *= 0.9
                v *= 0.9
            bary.append([u, v])
        bary = _project_to_bary_simplex(np.array(bary[:N]))
        inits.append(bary)
    
    # 3. Perturbed versions of known good patterns
    base = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.33, 0.0], [0.67, 0.0], [0.0, 0.33],
        [0.0, 0.67], [0.33, 0.33], [0.67, 0.33],
        [0.33, 0.67], [0.5, 0.5],
    ])
    for scale in [0.01, 0.02, 0.03]:
        bary_pert = base + rng.normal(0, scale, size=base.shape)
        bary_pert = _project_to_bary_simplex(bary_pert)
        inits.append(bary_pert)
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_bary = None
    best_score = -np.inf
    
    inits = _generate_initializations()
    
    # Phase 1: Evaluate all initializations
    for bary in inits:
        bary = _project_to_bary_simplex(bary)
        dets = _all_dets_from_bary(bary)
        score = 0.5 * np.min(np.abs(dets))
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
    
    # Phase 2: SCP optimization from multiple starts
    # Sort initializations by score, try best ones first
    scored_inits = []
    for bary in inits:
        bary = _project_to_bary_simplex(bary)
        dets = _all_dets_from_bary(bary)
        score = 0.5 * np.min(np.abs(dets))
        scored_inits.append((score, bary))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Try SCP from top initializations and perturbations of best
    candidates = [b for _, b in scored_inits[:15]]
    
    # Add perturbations of current best
    if best_bary is not None:
        rng = np.random.RandomState(42)
        for scale in [0.01, 0.02, 0.03, 0.05]:
            for _ in range(2):
                bary_pert = best_bary + rng.randn(N, 2) * scale
                bary_pert = _project_to_bary_simplex(bary_pert)
                candidates.append(bary_pert)
    
    # Run SCP on candidates
    for i, bary_init in enumerate(candidates):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time()
        time_per_scp = min(15.0, remaining / max(1, len(candidates) - i))
        
        try:
            bary_opt, score_opt = _scp_optimize(
                bary_init, n_outer=8, active_size=20, time_limit=time_per_scp
            )
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Final refinement with larger active set on the best
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bary_final, score_final = _scp_optimize(
                best_bary, n_outer=15, active_size=40, time_limit=10.0
            )
            if score_final > best_score:
                best_score = score_final
                best_bary = bary_final.copy()
        except Exception:
            pass
    
    # Fallback
    if best_bary is None:
        best_bary = inits[0]
    
    # Convert to Cartesian
    points = _bary_to_cart(best_bary)
    points = _clip_to_triangle(points)
    
    return points
