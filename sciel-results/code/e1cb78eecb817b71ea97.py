import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 142.0  # 8s safety margin before 150s kill


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

    # The objective is the soft-min of the set of distances:
    # d_k for all pairs (k=0..M-1) and (1 - |p_i|) for all i (k=M..M+N-1)
    # We want to maximize the min distance, so we minimize -min distance.
    # soft_min(x) = -soft_max(-x) = - (1/beta) ln(sum(exp(beta * x_k)))
    # Wait, the standard soft-min is (1/beta) ln(sum(exp(-beta * x_k))).
    # Let's stick to the standard formulation: minimize soft_min(distances).
    
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])

    # Stable log-sum-exp computation
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    
    # Handle case where sum_exp is 0 (should not happen with finite inputs)
    if sum_exp == 0:
        return 0.0, np.zeros_like(flat)
        
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta

    weights = np.exp(y_shifted) / sum_exp

    grad_pts = np.zeros_like(pts)

    # Boundary gradient
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    # d/dp_i (1 - |p_i|) = -p_i / |p_i|
    # d/dp_i soft_min = sum w_k * d(x_k)/dp_i
    # x_k = 1 - |p_i|, so d(x_k)/dp_i = -p_i/|p_i|
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * (-pts)
    grad_pts += boundary_grad

    # Pairwise gradient
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    # x_k = 0.5 * |p_i - p_j|
    # d(x_k)/dp_i = 0.5 * (p_i - p_j) / |p_i - p_j|
    # d(x_k)/dp_j = 0.5 * (p_j - p_i) / |p_i - p_j|
    
    grad_contrib_i = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = -0.5 * w_pairs[:, np.newaxis] * unit_vecs

    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)

    grad_flat = grad_pts.flatten()

    return obj_val, grad_flat


def _force_directed_relax(pts, steps=50, dt=0.01):
    """
    Simple force-directed relaxation to remove overlaps and push circles inside.
    """
    pts = pts.copy()
    iu = np.triu_indices(N, k=1)
    i_idx, m_idx = iu
    
    for _ in range(steps):
        if time.time() > _DEADLINE - 5.0:
            break
            
        force = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only apply force if overlapping or very close
        # We want to push them apart until dist > 2*r_est
        # Let's use a target radius based on current min dist
        min_dist = np.min(dist_mat)
        r_est = min_dist / 2.0
        if r_est < 1e-6:
            r_est = 0.1
            
        # Repulsion strength
        d_pairs = dist_mat[iu]
        vecs = pts[i_idx] - pts[m_idx]
        units = vecs / np.where(d_pairs < 1e-8, 1.0, d_pairs)[:, np.newaxis]
        
        # Force is proportional to overlap
        overlap = np.maximum(0, 2.0 * r_est - d_pairs)
        f_pairs = overlap[:, np.newaxis] * units
        
        np.add.at(force, i_idx, f_pairs)
        np.add.at(force, m_idx, -f_pairs)
        
        # Boundary attraction/repulsion
        dists_origin = np.linalg.norm(pts, axis=1)
        # If outside or close to boundary, push in
        boundary_violation = np.maximum(0, dists_origin - (1.0 - r_est))
        units_origin = pts / np.where(dists_origin < 1e-8, 1.0, dists_origin)[:, np.newaxis]
        force -= boundary_violation[:, np.newaxis] * units_origin
        
        pts += dt * force
        
        # Project inside
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]

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


def _ring_init():
    """
    Initialize circles in concentric rings matching the known optimal structure.
    """
    pts = np.zeros((N, 2))
    
    # Ring 0: center
    pts[0] = [0.0, 0.0]
    
    # Ring 1: 6 circles
    r1 = 0.25
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts[1:7, 0] = r1 * np.cos(angles1)
    pts[1:7, 1] = r1 * np.sin(angles1)
    
    # Ring 2: 12 circles
    r2 = 0.50
    angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi/12
    pts[7:19, 0] = r2 * np.cos(angles2)
    pts[7:19, 1] = r2 * np.sin(angles2)
    
    # Ring 3: 18 circles
    r3 = 0.72
    angles3 = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    pts[19:37, 0] = r3 * np.cos(angles3)
    pts[19:37, 1] = r3 * np.sin(angles3)
    
    # Ring 4: 13 circles
    r4 = 0.87
    angles4 = np.linspace(0, 2 * np.pi, 13, endpoint=False) + np.pi/26
    pts[37:50, 0] = r4 * np.cos(angles4)
    pts[37:50, 1] = r4 * np.sin(angles4)
    
    return pts


def _random_init():
    """Random initialization."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _optimize_from_start(pts, beta_schedule, max_iter_per_beta=300):
    """Run L-BFGS-B with a schedule of increasing beta values."""
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)

    for beta in beta_schedule:
        if time.time() > _DEADLINE - 10.0:
            break

        try:
            if beta >= 1000:
                ftol = 1e-13
                gtol = 1e-9
                max_iter = max_iter_per_beta
            else:
                ftol = 1e-12
                gtol = 1e-8
                max_iter = max_iter_per_beta

            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
            )

            new_pts = res.x.reshape(N, 2)

            # Ensure points are inside the unit disk
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

            new_score = _score(new_pts)

            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()

        except Exception:
            pass

    return best_pts, best_score


def construct():
    start_time = time.time()

    # Generate diverse initializations
    inits = []
    
    # 1. Hex lattice
    hex_pts = _hex_lattice_init()
    inits.append(hex_pts)
    
    # 2. Ring structure
    ring_pts = _ring_init()
    inits.append(ring_pts)
    
    # 3. Random inits
    for _ in range(4):
        inits.append(_random_init())

    best_pts = inits[0]
    best_score = _score(best_pts)

    # Pre-condition with force-directed relaxation for each init
    # This helps remove overlaps and get a valid starting point
    for i, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 20.0:
            break
        
        relaxed_pts = _force_directed_relax(init_pts, steps=30, dt=0.02)
        inits[i] = relaxed_pts
        
        # Update best if relaxed version is better
        s = _score(relaxed_pts)
        if s > best_score:
            best_score = s
            best_pts = relaxed_pts.copy()

    # Optimization schedules
    # Start with lower beta for global movement, then increase for precision
    beta_schedule_1 = [10, 50, 200, 1000]
    beta_schedule_2 = [20, 100, 500, 2000]
    beta_schedule_3 = [50, 200, 1000, 5000]

    # Optimize from each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 15.0:
            break

        if init_idx % 3 == 0:
            schedule = beta_schedule_1
        elif init_idx % 3 == 1:
            schedule = beta_schedule_2
        else:
            schedule = beta_schedule_3

        # Add small perturbation to escape symmetry for non-hex inits
        if init_idx > 0:
            perturbed = init_pts + np.random.normal(0, 0.01, size=init_pts.shape)
            candidate_pts, candidate_score = _optimize_from_start(perturbed, schedule, max_iter_per_beta=200)
        else:
            candidate_pts, candidate_score = _optimize_from_start(init_pts, schedule, max_iter_per_beta=400)

        if candidate_score > best_score:
            best_score = candidate_score
            best_pts = candidate_pts.copy()

    # Perturbation-retry loop on the best solution
    if time.time() < _DEADLINE - 10.0:
        for kick_iter in range(3):
            if time.time() > _DEADLINE - 10.0:
                break
            
            # Asymmetric perturbation
            perturbed = best_pts.copy()
            dists = np.linalg.norm(perturbed, axis=1)
            scales = 0.01 + 0.02 * dists
            perturbed += np.random.normal(0, 1, size=perturbed.shape) * scales[:, np.newaxis]
            
            # Relax the perturbed points
            relaxed = _force_directed_relax(perturbed, steps=20, dt=0.01)
            
            schedule = [200, 1000, 5000]
            candidate_pts, candidate_score = _optimize_from_start(relaxed, schedule, max_iter_per_beta=300)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()

    # Final high-beta polish
    if time.time() < _DEADLINE - 5.0:
        polish_betas = [10000, 50000, 200000]
        for beta in polish_betas:
            if time.time() > _DEADLINE - 3.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    best_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                new_pts = res.x.reshape(N, 2)
                dists = np.linalg.norm(new_pts, axis=1)
                mask = dists > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()

            except Exception:
                pass

    return best_pts
