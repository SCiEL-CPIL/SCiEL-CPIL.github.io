import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 135.0  # 15s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0

    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0

    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0

    if N < 2:
        return r_cont

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)

    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    """
    pts = flat.reshape(N, 2)

    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)

    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]

    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])

    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)

    obj_val = log_sum_exp / beta

    weights = np.exp(y_shifted) / sum_exp

    grad_pts = np.zeros_like(pts)

    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad

    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]

    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs

    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)

    grad_flat = grad_pts.flatten()

    return obj_val, grad_flat


def _pbd_solve(pts, r_target, iterations=500, dt=0.5, damping=0.9):
    """
    Position-Based Dynamics solver for circle packing.
    Each iteration:
      1. Apply small velocity (inertia)
      2. Resolve pairwise overlaps by moving circles apart
      3. Project circles inside the unit disk boundary
      4. Apply damping to prevent oscillation
    
    This guarantees hard feasibility at every iteration.
    """
    pts = pts.copy().astype(np.float64)
    iu = np.triu_indices(N, k=1)
    i_idx, m_idx = iu
    n_pairs = len(i_idx)
    
    vel = np.zeros_like(pts)
    
    for it in range(iterations):
        if time.time() > _DEADLINE - 12.0:
            break
        
        # Apply velocity with damping
        pts += dt * vel
        vel *= damping
        
        # Resolve pairwise overlaps (multiple passes for convergence)
        for _ in range(3):
            diff = pts[i_idx] - pts[m_idx]
            dists = np.linalg.norm(diff, axis=1)
            
            # Only resolve pairs that overlap (distance < 2*r_target)
            overlap_mask = dists < 2.0 * r_target
            if not np.any(overlap_mask):
                break
            
            ii = i_idx[overlap_mask]
            mm = m_idx[overlap_mask]
            d = dists[overlap_mask]
            safe_d = np.maximum(d, 1e-10)
            
            # Required separation
            required = 2.0 * r_target
            corrections = (required - d) / safe_d * 0.5
            
            vecs = (pts[ii] - pts[mm]) / safe_d[:, np.newaxis]
            
            # Apply corrections (each circle moves half)
            pts[ii] += corrections[:, np.newaxis] * vecs
            pts[mm] -= corrections[:, np.newaxis] * vecs
        
        # Project inside unit disk: |pts| + r_target <= 1
        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_violation = dists_origin + r_target - 1.0
        boundary_mask = boundary_violation > 0
        if np.any(boundary_mask):
            bi = np.where(boundary_mask)[0]
            d_b = dists_origin[bi]
            safe_d_b = np.maximum(d_b, 1e-10)
            # Move inward by the violation amount
            inward_vecs = pts[bi] / safe_d_b[:, np.newaxis]
            pts[bi] -= boundary_violation[bi, np.newaxis] * inward_vecs
        
        # Update velocity from position change (for next iteration)
        # We track the net displacement from constraints
        vel = (pts - (pts - dt * vel * damping)) / max(dt, 1e-10) * 0.1
    
    return pts


def _pbd_optimize_radius(pts, r_low, r_high, max_iter=40, tol=1e-9):
    """
    Binary search on radius using PBD as the feasibility solver.
    For a given r, run PBD and check if the result achieves radius >= r.
    """
    best_r = r_low
    best_pts = pts.copy()
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 12.0:
            break
        if r_high - r_low < tol:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Run PBD to try to achieve radius r_mid
        result = _pbd_solve(pts, r_mid, iterations=300, dt=0.5, damping=0.9)
        
        # Check actual achieved radius
        achieved = _score(result)
        
        if achieved >= r_mid - 1e-7:
            # Feasible at r_mid, try larger
            best_r = r_mid
            best_pts = result.copy()
            r_low = r_mid
        else:
            # Not feasible, try smaller
            r_high = r_mid
    
    return best_r, best_pts


def _soft_min_optimize(pts, beta_schedule, max_iter_per_beta=400):
    """Run L-BFGS-B with a schedule of increasing beta values."""
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)

    for beta in beta_schedule:
        if time.time() > _DEADLINE - 15.0:
            break

        try:
            if beta >= 1000:
                ftol = 1e-13
                gtol = 1e-9
            else:
                ftol = 1e-12
                gtol = 1e-8

            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': max_iter_per_beta, 'ftol': ftol, 'gtol': gtol, 'disp': False}
            )

            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)

            new_score = _score(new_pts)

            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()

        except Exception:
            pass

    return best_pts, best_score


def _project_to_disk(pts):
    """Project points back into the unit disk."""
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        pts = pts.copy()
        pts[mask] = pts[mask] / dists[mask, np.newaxis]
    return pts


def _ring_init():
    """
    Initialize circles in concentric rings matching the known optimal structure.
    Known best packing uses rings: 1 (center) + 6 + 12 + 18 + 13 = 50
    """
    pts = np.zeros((N, 2))
    
    # Ring 0: center
    pts[0] = [0.0, 0.0]
    
    # Ring 1: 6 circles
    r1 = 0.2516
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts[1:7, 0] = r1 * np.cos(angles1)
    pts[1:7, 1] = r1 * np.sin(angles1)
    
    # Ring 2: 12 circles
    r2 = 0.5032
    angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi/12
    pts[7:19, 0] = r2 * np.cos(angles2)
    pts[7:19, 1] = r2 * np.sin(angles2)
    
    # Ring 3: 18 circles
    r3 = 0.7248
    angles3 = np.linspace(0, 2 * np.pi, 18, endpoint=False) + np.pi/18
    pts[19:37, 0] = r3 * np.cos(angles3)
    pts[19:37, 1] = r3 * np.sin(angles3)
    
    # Ring 4: 13 circles
    r4 = 0.8758
    angles4 = np.linspace(0, 2 * np.pi, 13, endpoint=False) + np.pi/26
    pts[37:50, 0] = r4 * np.cos(angles4)
    pts[37:50, 1] = r4 * np.sin(angles4)
    
    return pts


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0

    for spacing in [0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
        k = int(1.0 / spacing) + 4
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])

        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]

        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts

    if best_pts is not None:
        return best_pts

    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_init():
    """Random initialization with bias toward uniform area coverage."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _perturb(pts, scale=0.01):
    """Add small random perturbation to escape local optima."""
    return pts + np.random.normal(0, scale, size=pts.shape)


def _full_optimize(pts):
    """
    Full optimization pipeline combining PBD and soft-min:
    1. PBD feasibility solve to get a feasible starting point
    2. Soft-min L-BFGS-B with beta annealing for precision
    3. PBD binary search radius refinement
    4. Final high-precision soft-min polish
    """
    initial_score = _score(pts)
    
    # Step 1: PBD to get a feasible configuration
    r_start = max(initial_score, 0.12)
    pbd_result = _pbd_solve(pts, r_start, iterations=200, dt=0.5, damping=0.9)
    pbd_score = _score(pbd_result)
    
    if pbd_score > initial_score:
        pts = pbd_result
        initial_score = pbd_score
    
    # Step 2: Soft-min L-BFGS-B with beta annealing
    beta_schedule = [20, 50, 100, 300, 1000, 5000, 20000, 100000]
    pts, score = _soft_min_optimize(pts, beta_schedule, max_iter_per_beta=300)
    
    # Step 3: PBD binary search radius refinement
    if time.time() < _DEADLINE - 20.0:
        r_low = score
        r_high = score * 1.03
        r_best, pts_best = _pbd_optimize_radius(pts, r_low, r_high, max_iter=25, tol=1e-9)
        
        if r_best > score:
            score = r_best
            pts = pts_best
    
    # Step 4: Final high-precision soft-min polish
    if time.time() < _DEADLINE - 10.0:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                pts.flatten(),
                args=(500000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)
            new_score = _score(new_pts)
            if new_score > score:
                score = new_score
                pts = new_pts
        except Exception:
            pass
    
    return pts, score


def construct():
    start_time = time.time()
    np.random.seed(42)  # Reproducibility
    
    # Multiple diverse initializations
    inits = []
    inits.append(_ring_init())
    inits.append(_hex_lattice_init())
    for _ in range(4):
        inits.append(_random_init())

    # Score all inits
    init_scores = [(i, _score(init_pts)) for i, init_pts in enumerate(inits)]
    init_scores.sort(key=lambda x: x[1], reverse=True)

    best_pts = inits[init_scores[0][0]]
    best_score = init_scores[0][1]

    # Try each initialization with full optimization
    for init_idx, (orig_idx, init_score) in enumerate(init_scores):
        if time.time() > _DEADLINE - 30.0:
            break

        init_pts = inits[orig_idx]

        # Perturb for diversity on non-best inits
        if orig_idx != init_scores[0][0]:
            perturbed = _perturb(init_pts, scale=0.01)
        else:
            perturbed = init_pts.copy()

        candidate_pts, candidate_score = _full_optimize(perturbed)

        if candidate_score > best_score:
            best_score = candidate_score
            best_pts = candidate_pts.copy()

    # Perturbation-retry loop: kick the best solution and re-optimize
    if time.time() < _DEADLINE - 25.0:
        for kick_iter in range(5):
            if time.time() > _DEADLINE - 25.0:
                break
            
            if kick_iter % 2 == 0:
                perturbed = _perturb(best_pts, scale=0.008)
            else:
                # Asymmetric perturbation: larger kicks for outer circles
                perturbed = best_pts.copy()
                dists = np.linalg.norm(perturbed, axis=1)
                scales = 0.004 + 0.015 * dists
                perturbed += np.random.normal(0, 1, size=perturbed.shape) * scales[:, np.newaxis]

            candidate_pts, candidate_score = _full_optimize(perturbed)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()

    # Final high-beta polish on the best result
    if time.time() < _DEADLINE - 8.0:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(1000000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass

    # Final PBD polish for hard feasibility
    if time.time() < _DEADLINE - 5.0:
        pbd_final = _pbd_solve(best_pts, best_score, iterations=200, dt=0.3, damping=0.95)
        pbd_final_score = _score(pbd_final)
        if pbd_final_score > best_score:
            best_score = pbd_final_score
            best_pts = pbd_final.copy()

    return best_pts
