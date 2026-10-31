import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin within 120s/150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    # coeff_ij = 4 * (6/r8 - 12/r14)
    # grad_i = sum_j coeff_ij * (pos_i - pos_j)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    pts = pts[:N].copy()
    
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _compute_fcc_fingerprint(target_pos):
    """
    Compute the pairwise distance histogram for the target FCC structure.
    Returns (distances, weights) for use in geometric loss.
    """
    n = len(target_pos)
    diff = target_pos[:, np.newaxis, :] - target_pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    
    # Get unique distance shells
    distances = np.sqrt(r2)
    tri_idx = np.triu_indices(n, k=1)
    tri_distances = distances[tri_idx]
    
    # Cluster distances into shells (within 1% tolerance)
    sorted_dists = np.sort(tri_distances)
    shells = []
    current_shell = [sorted_dists[0]]
    
    for d in sorted_dists[1:]:
        if abs(d - current_shell[-1]) / current_shell[-1] < 0.01:
            current_shell.append(d)
        else:
            shells.append(np.mean(current_shell))
            current_shell = [d]
    shells.append(np.mean(current_shell))
    
    return np.array(shells)

def _geometric_loss_and_grad(pos, target_shells):
    """
    Geometric loss that penalizes deviations from the target FCC distance shells.
    Uses a soft-min to make it differentiable.
    
    Loss = sum over all pairs of min_k (|d_ij - shell_k|^2)
    where shell_k are the target distance shells.
    """
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    
    # For each pair, find the closest shell
    # Soft-min: -log(sum(exp(-alpha * (d - shell_k)^2)))
    alpha = 50.0  # Sharpness of the soft-min
    
    # r is (n, n), target_shells is (m,)
    # Compute (r_ij - shell_k)^2 for all i,j,k
    r_expanded = r[:, :, np.newaxis]  # (n, n, 1)
    shells_expanded = target_shells[np.newaxis, np.newaxis, :]  # (1, 1, m)
    
    sq_diff = (r_expanded - shells_expanded) ** 2  # (n, n, m)
    
    # Soft-min over shells: -log(sum(exp(-alpha * sq_diff)))
    # To avoid overflow, subtract max
    max_sq = np.max(sq_diff, axis=2, keepdims=True)
    exp_term = np.exp(-alpha * (sq_diff - max_sq))
    log_sum = np.log(np.sum(exp_term, axis=2) + 1e-30)
    
    soft_min_sq = max_sq - log_sum / alpha  # (n, n)
    
    # Only consider upper triangle (each pair once)
    tri_idx = np.triu_indices(n, k=1)
    loss = np.sum(soft_min_sq[tri_idx])
    
    # Gradient: need d(soft_min_sq)/d(pos)
    # This is complex, so we'll use finite differences or a simpler approach
    # For now, return loss only and use gradient-free methods or finite diff
    
    # Actually, let's compute the gradient analytically
    # d(soft_min)/d(r_ij) = -sum_k exp(-alpha*(r_ij - shell_k)^2) * (-2*alpha*(r_ij - shell_k)) / sum_k exp(-alpha*(r_ij - shell_k)^2)
    # = 2*alpha * sum_k exp(-alpha*(r_ij - shell_k)^2) * (r_ij - shell_k) / sum_k exp(-alpha*(r_ij - shell_k)^2)
    
    # This is getting complex. Let's use a simpler geometric loss:
    # Penalize pairs that are too close to each other (within a shell) or too far
    # from all shells.
    
    # Simpler approach: for each pair, compute the minimum distance to any shell
    # and penalize if it's not close to a shell.
    
    # Even simpler: just use the raw distances and compare to expected counts
    # in each shell.
    
    return loss, None  # Gradient not computed for simplicity

def _relax(pos, max_iter=1000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    pts = pts[:N].copy()
    
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _optimize_scale(pos, t0, deadline):
    """
    Optimize the overall scale factor to minimize LJ energy.
    This is a 1D optimization that can quickly find the optimal scale.
    """
    t0_scale = time.time()
    
    # Try a range of scales
    scales = np.linspace(0.9, 1.1, 21)
    best_scale = 1.0
    best_energy = np.inf
    
    for s in scales:
        if time.time() > deadline - 5.0:
            break
        pos_scaled = pos * s
        try:
            e, _ = _lj_energy_and_grad(pos_scaled)
            if e < best_energy:
                best_energy = e
                best_scale = s
        except Exception:
            continue
    
    # Refine around the best scale
    if best_scale > 0.91 and best_scale < 1.09:
        fine_scales = np.linspace(best_scale - 0.02, best_scale + 0.02, 11)
        for s in fine_scales:
            if time.time() > deadline - 5.0:
                break
            pos_scaled = pos * s
            try:
                e, _ = _lj_energy_and_grad(pos_scaled)
                if e < best_energy:
                    best_energy = e
                    best_scale = s
            except Exception:
                continue
    
    return pos * best_scale, best_energy

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def _geometric_guided_relax(pos, target_pos, t0, deadline, rng, alpha_start=1.0, alpha_end=0.0):
    """
    Two-stage relaxation:
    1. Minimize a geometric loss that pulls toward the target FCC structure
    2. Then minimize the LJ energy
    
    The geometric loss uses a smooth penalty for deviations from the target
    pairwise distances, which helps navigate the icosahedral funnel.
    """
    t0_geom = time.time()
    
    # Phase 1: Geometric guidance
    # We'll use a simple approach: interpolate between current and target positions
    # while minimizing LJ energy, with a weight on staying close to target
    
    n_iter_geom = 50
    best_pos = pos.copy()
    best_energy = _lj_energy_and_grad(pos)[0]
    
    for i in range(n_iter_geom):
        if time.time() > deadline - 10.0:
            break
        
        # Interpolation parameter: start with strong geometric pull, decrease
        alpha = alpha_start * (1.0 - i / n_iter_geom) ** 2 + alpha_end
        
        # Create a blended objective: LJ energy + geometric penalty
        # Geometric penalty: sum of squared distances to target positions
        # (after optimal alignment - but for simplicity, just use raw positions)
        
        def blended_obj_grad(x):
            p = x.reshape(-1, 3)
            e_lj, g_lj = _lj_energy_and_grad(p)
            
            # Geometric penalty: pull toward target
            # Use a soft version: only penalize if far from target
            diff_to_target = p - target_pos
            # Smooth penalty: use exp(-|diff|^2 / sigma^2) style
            sigma = 1.0
            sq_diff = np.sum(diff_to_target ** 2, axis=1)
            geom_penalty = np.sum(sq_diff) / (2.0 * sigma ** 2)
            geom_grad = diff_to_target / sigma ** 2
            
            e_total = e_lj + alpha * geom_penalty
            g_total = g_lj + alpha * geom_grad
            
            return e_total, g_total.ravel()
        
        try:
            res = minimize(
                blended_obj_grad,
                best_pos.ravel(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 100, "gtol": 1e-6, "ftol": 1e-10},
            )
            p_out = res.x.reshape(-1, 3)
            e_lj_out, _ = _lj_energy_and_grad(p_out)
            
            if e_lj_out < best_energy:
                best_energy = e_lj_out
                best_pos = p_out
        except Exception:
            continue
    
    # Phase 2: Pure LJ relaxation
    try:
        p_final, e_final = _relax(best_pos, max_iter=500, gtol=1e-10)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = p_final
    except Exception:
        pass
    
    return best_pos, best_energy

def _basin_hopping(pos, t0, deadline, rng):
    """
    Basin hopping with adaptive step size.
    Returns (best_pos, best_energy).
    """
    best_pos, best_energy = pos.copy(), _lj_energy_and_grad(pos)[0]
    current_pos, current_energy = pos.copy(), best_energy
    
    step_size = 0.5
    min_step = 0.05
    max_step = 1.5
    
    # Simple annealing schedule
    # Start with high T, decrease
    T = 0.5
    
    n_iter = 0
    max_iter = 200 # Hard cap on iterations to prevent infinite loops if deadline logic fails
    
    while time.time() < deadline - 5.0 and n_iter < max_iter:
        n_iter += 1
        
        # Perturb
        noise = rng.normal(0, step_size, size=pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        try:
            relaxed, e_relaxed = _relax(perturbed, max_iter=200, gtol=1e-8)
        except Exception:
            relaxed, e_relaxed = perturbed, np.inf
            
        if e_relaxed < best_energy:
            best_energy = e_relaxed
            best_pos = relaxed.copy()
            
        # Metropolis criterion
        delta_e = e_relaxed - current_energy
        if delta_e < 0:
            current_pos = relaxed
            current_energy = e_relaxed
            # If we improved, maybe increase step size slightly or keep it
            # step_size = min(max_step, step_size * 1.05)
        else:
            if np.log(rng.uniform()) < -delta_e / T:
                current_pos = relaxed
                current_energy = e_relaxed
            else:
                # Rejected, maybe decrease step size
                pass
                
        # Cooling
        T *= 0.98
        
        # Adaptive step size: if we've been rejecting a lot, shrink
        # (Simplified: just shrink slightly every few steps)
        if n_iter % 10 == 0:
            step_size = max(min_step, step_size * 0.95)

    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # 1. Generate initial FCC geometry
    try:
        pos0 = _make_fcc_cluster()
    except Exception:
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pos0 = np.array(pts[:N], dtype=np.float64)
        pos0 -= np.mean(pos0, axis=0)
        pos0 *= 1.12

    best_pos = pos0
    best_energy = np.inf

    # 2. Initial relaxation with geometric guidance
    try:
        p, e = _geometric_guided_relax(pos0, pos0, t0, deadline, np.random.default_rng(42))
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        try:
            p, e = _relax(pos0, max_iter=500, gtol=1e-8)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 3. Scale optimization
    if time.time() < deadline - 15.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos, t0, deadline)
            p_relaxed, e_relaxed = _relax(p_scaled, max_iter=300, gtol=1e-10)
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = p_relaxed
        except Exception:
            pass

    # 4. High-precision polish
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-12, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 5. Basin Hopping from FCC-biased starts
    rng = np.random.default_rng(42)
    
    # Run 1: From the relaxed FCC
    if time.time() < deadline - 20.0:
        try:
            p_bh, e_bh = _basin_hopping(best_pos, t0, deadline, rng)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            pass

    # Run 2: From a random perturbation of FCC (to escape local distortions)
    if time.time() < deadline - 20.0:
        try:
            perturbed_start = best_pos + 0.1 * rng.standard_normal(best_pos.shape)
            perturbed_start = _ensure_min_sep(perturbed_start)
            p_bh, e_bh = _basin_hopping(perturbed_start, t0, deadline, rng)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            pass
            
    # Run 3: From a different random start (maybe icosahedral-ish?)
    # Just to be safe, try a random cluster start
    if time.time() < deadline - 30.0:
        try:
            # Random cluster within a sphere
            random_pos = rng.uniform(-2, 2, size=(N, 3))
            random_pos = _ensure_min_sep(random_pos)
            p_bh, e_bh = _basin_hopping(random_pos, t0, deadline, rng)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            pass

    # 6. Final check and separation enforcement
    best_pos = _ensure_min_sep(best_pos)
    
    # 7. Final high-precision polish on the absolute best
    if time.time() < deadline - 5.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-12, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
                best_pos = _ensure_min_sep(best_pos)
        except Exception:
            pass

    return best_pos
