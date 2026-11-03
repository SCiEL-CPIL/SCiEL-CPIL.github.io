"""Packing 50 equal circles in a unit circle using ring-structured optimization and binary-search polishing."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _pairwise_distances(points):
    """Compute all pairwise distances for n points."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _score_vec(points):
    """Vectorized score computation: max common radius."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    dists = _pairwise_distances(points)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _build_contact_graph(points, r, n_contacts=6):
    """Build a contact graph based on nearest neighbors."""
    n = len(points)
    dists = _pairwise_distances(points)
    np.fill_diagonal(dists, np.inf)
    
    edges = set()
    for i in range(n):
        nearest = np.argsort(dists[i])[:n_contacts]
        for j in nearest:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    
    return np.array(list(edges)) if edges else np.array([], dtype=int).reshape(0, 2)


def _force_directed_energy(points, r, contact_edges, k_contact=50.0, k_boundary=100.0, k_repel=10.0):
    """Force-directed layout energy."""
    n = len(points)
    energy = 0.0
    
    if len(contact_edges) > 0:
        ci = points[contact_edges[:, 0]]
        cj = points[contact_edges[:, 1]]
        d = np.linalg.norm(ci - cj, axis=1)
        energy += k_contact * np.sum((d - 2 * r) ** 2)
    
    radii = np.linalg.norm(points, axis=1)
    energy += k_boundary * np.sum((radii - (1.0 - r)) ** 2)
    
    dists = _pairwise_distances(points)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu]
    violations = np.maximum(0, 2 * r - pair_dists)
    energy += k_repel * np.sum(violations ** 2)
    
    return energy


def _force_directed_optimize(points, r, time_limit=10.0, max_iters=300):
    """Optimize positions using force-directed layout with adaptive cooling."""
    n = len(points)
    pts = points.copy()
    best_pts = pts.copy()
    best_energy = _force_directed_energy(pts, r, _build_contact_graph(pts, r))
    
    t_start = time.time()
    T = 0.005
    step_size = 0.003
    
    for it in range(max_iters):
        if time.time() - t_start >= time_limit:
            break
        
        contact_edges = _build_contact_graph(pts, r)
        
        forces = np.zeros_like(pts)
        if len(contact_edges) > 0:
            ci = pts[contact_edges[:, 0]]
            cj = pts[contact_edges[:, 1]]
            d = np.linalg.norm(ci - cj, axis=1)
            diff_vec = ci - cj
            diff_vec_norm = diff_vec / np.maximum(d[:, np.newaxis], 1e-10)
            spring_force = 50.0 * (d - 2 * r)[:, np.newaxis] * diff_vec_norm
            for idx in range(len(contact_edges)):
                i, j = contact_edges[idx]
                forces[i] += spring_force[idx]
                forces[j] -= spring_force[idx]
        
        radii = np.linalg.norm(pts, axis=1)
        radii_safe = np.maximum(radii, 1e-10)
        boundary_force = 100.0 * (radii - (1.0 - r))[:, np.newaxis] * pts / radii_safe[:, np.newaxis]
        forces -= boundary_force
        
        dists = _pairwise_distances(pts)
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(n, 1)
        pair_dists = dists[iu]
        violations = np.maximum(0, 2 * r - pair_dists)
        
        for idx in range(len(pair_dists)):
            i, j = iu[0][idx], iu[1][idx]
            if violations[idx] > 0:
                d = dists[i, j]
                if d > 1e-10:
                    force_mag = 10.0 * violations[idx] / d
                    forces[i] += force_mag * (pts[i] - pts[j])
                    forces[j] -= force_mag * (pts[i] - pts[j])
        
        alpha = step_size * (0.5 + 0.5 * (1 - it / max_iters))
        pts_new = pts + alpha * forces
        pts_new += T * np.random.randn(n, 2)
        
        new_energy = _force_directed_energy(pts_new, r, _build_contact_graph(pts_new, r))
        if new_energy < best_energy or np.random.rand() < np.exp(-(new_energy - best_energy) / max(T, 1e-10)):
            pts = pts_new
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = pts_new.copy()
        
        T *= 0.99
    
    return best_pts


def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.15, 0.28, 30):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points


def _ring_init(n, ring_radii=None, ring_angles=None):
    """
    Initialize with ring structure matching known optimal packing topology.
    Ring 0: center (1 circle)
    Ring 1: 6 circles
    Ring 2: 12 circles
    Ring 3: 18 circles
    Ring 4: 13 circles (remaining)
    """
    if ring_radii is None:
        ring_radii = [0.0, 0.252, 0.504, 0.756, 0.950]
    if ring_angles is None:
        ring_angles = [0.0, 0.0, np.pi/12, 0.0, np.pi/13]
    
    ring_counts = [1, 6, 12, 18, 13]
    points = []
    
    for ring_idx, (count, radius, angle_offset) in enumerate(zip(ring_counts, ring_radii, ring_angles)):
        for k in range(count):
            angle = 2 * np.pi * k / count + angle_offset
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    return np.array(points[:n])


def _softmin_logsumexp(vals, beta):
    """Numerically stable log-sum-exp for soft-min."""
    m = np.min(vals)
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta


def _smooth_objective(x, beta, pair_idx):
    """Smooth min objective: minimize the soft-min of all constraints."""
    c = x.reshape(N, 2)
    ci = c[pair_idx[:, 0]]
    cj = c[pair_idx[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2.0
    b = 1.0 - np.hypot(c[:, 0], c[:, 1])
    all_vals = np.concatenate([d, b])
    return -_softmin_logsumexp(all_vals, beta)


def _local_optimize(c, time_limit=10.0, beta_schedule=None):
    """Local optimization using L-BFGS-B with smooth-min objective and epigraph polish."""
    pair_idx = _pairwise_indices(N)
    
    if beta_schedule is None:
        beta_schedule = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    
    best_c = c.copy()
    best_r = _score_vec(best_c)
    
    t_start = time.time()
    
    for beta in beta_schedule:
        if time.time() - t_start >= time_limit - 2:
            break
        
        def neg_soft_min(x):
            return _smooth_objective(x, beta, pair_idx)
        
        def callback(x):
            if time.time() - t_start >= time_limit - 1:
                raise Exception("timeout")
        
        try:
            res = minimize(
                neg_soft_min,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50},
                callback=callback
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score_vec(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    # Final epigraph polish: use current best_r as target
    if time.time() - t_start < time_limit - 1:
        r_target = best_r * 1.001
        def obj_epigraph(x):
            c_local = x.reshape(N, 2)
            ci = c_local[pair_idx[:, 0]]
            cj = c_local[pair_idx[:, 1]]
            d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2.0
            b = 1.0 - np.hypot(c_local[:, 0], c_local[:, 1])
            all_vals = np.concatenate([d, b])
            viol = np.maximum(0.0, r_target - all_vals)
            return np.sum(viol ** 2)
        
        try:
            res = minimize(
                obj_epigraph,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score_vec(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _binary_search_polish(c, r_low, r_high, time_limit=5.0):
    """
    Binary search on radius using epigraph formulation with auxiliary variable t.
    Minimize t subject to all constraints >= r_target - t (soft).
    """
    pair_idx = _pairwise_indices(N)
    best_c = c.copy()
    best_r = r_low
    
    t_start = time.time()
    iters = 0
    max_iters = 15
    
    while time.time() - t_start < time_limit and iters < max_iters:
        iters += 1
        r_mid = 0.5 * (r_low + r_high)
        
        # Epigraph: minimize t + penalty*(t^2) s.t. all_vals >= r_mid - t
        # Use soft constraint: sum of max(0, r_mid - t - all_vals)^2
        def obj_at_r(x):
            c_local = x.reshape(N, 2)
            ci = c_local[pair_idx[:, 0]]
            cj = c_local[pair_idx[:, 1]]
            d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2.0
            b = 1.0 - np.hypot(c_local[:, 0], c_local[:, 1])
            all_vals = np.concatenate([d, b])
            # t is the last variable
            t = x[-1]
            viol = np.maximum(0.0, r_mid - t - all_vals)
            return t + 1e-6 * t * t + np.sum(viol ** 2)
        
        # Initial guess for t: small positive
        t_init = max(0.0, r_mid - _score_vec(best_c)) + 0.001
        x0 = np.concatenate([best_c.flatten(), [t_init]])
        
        # Bounds: t >= 0
        bounds = [(None, None)] * (2 * N) + [(0.0, 0.1)]
        
        try:
            res = minimize(
                obj_at_r,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x[:2*N].reshape(N, 2)
                r_achieved = _score_vec(cand)
                if r_achieved >= r_mid - 1e-9:
                    best_c = cand.copy()
                    best_r = max(best_r, r_achieved)
                    r_low = r_mid
                else:
                    r_high = r_mid
        except Exception:
            r_high = r_mid
    
    return best_c, best_r


def _compute_r(c):
    """Compute the max common radius for given centers."""
    return _score_vec(c)


def _de_ring_optimize(time_limit=20.0):
    """
    Use differential evolution to optimize ring parameters (radii and angle offsets).
    This explores the ring-structured configuration space efficiently.
    """
    t_start = time.time()
    
    # Parameters: 5 ring radii + 5 angle offsets
    bounds = [
        (0.0, 0.1),      # ring 0 radius (center, should stay ~0)
        (0.15, 0.35),    # ring 1
        (0.35, 0.60),    # ring 2
        (0.55, 0.85),    # ring 3
        (0.75, 1.0),     # ring 4
        (0.0, 2*np.pi),  # angle offset 0
        (0.0, 2*np.pi),  # angle offset 1
        (0.0, 2*np.pi),  # angle offset 2
        (0.0, 2*np.pi),  # angle offset 3
        (0.0, 2*np.pi),  # angle offset 4
    ]
    
    def de_objective(params):
        if time.time() - t_start > time_limit:
            return 0.0
        radii = params[:5]
        angles = params[5:]
        # Keep center at origin
        radii[0] = 0.0
        c = _ring_init(N, ring_radii=radii, ring_angles=angles)
        return -_score_vec(c)
    
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=50,
            popsize=8,
            seed=42,
            tol=1e-6,
            polish=False,
            update_interval=1,
            callback=lambda x, convergence: None
        )
        if np.all(np.isfinite(result.x)):
            radii = result.x[:5]
            radii[0] = 0.0
            angles = result.x[5:]
            c = _ring_init(N, ring_radii=radii, ring_angles=angles)
            r = _score_vec(c)
            return c, r
    except Exception:
        pass
    
    return None, 0.0


def construct():
    t_start = time.time()
    
    # Start with multiple initializations and pick the best
    inits = [
        _hexagonal_init(N),
        _ring_init(N),
    ]
    
    best_points = None
    best_score = 0.0
    
    for init in inits:
        r = _score_vec(init)
        if r > best_score:
            best_score = r
            best_points = init.copy()
    
    # DE ring optimization to find better ring parameters
    if time.time() < DEADLINE - 50:
        de_c, de_r = _de_ring_optimize(time_limit=min(20.0, DEADLINE - (time.time() - t_start) - 50))
        if de_c is not None and de_r > best_score:
            best_score = de_r
            best_points = de_c.copy()
    
    # Force-directed layout refinement
    if time.time() < DEADLINE - 35:
        fd_points = _force_directed_optimize(best_points, best_score, time_limit=min(10.0, DEADLINE - (time.time() - t_start) - 35))
        fd_score = _score_vec(fd_points)
        if fd_score > best_score:
            best_score = fd_score
            best_points = fd_points.copy()
    
    # Local optimization with smooth-min
    if time.time() < DEADLINE - 25:
        c_opt, r_opt = _local_optimize(best_points, time_limit=min(15.0, DEADLINE - (time.time() - t_start) - 25))
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Binary search polish to push radius higher
    if time.time() < DEADLINE - 15:
        c_bs, r_bs = _binary_search_polish(best_points, best_score, min(best_score * 1.02, 0.13), time_limit=min(10.0, DEADLINE - (time.time() - t_start) - 15))
        if r_bs > best_score:
            best_score = r_bs
            best_points = c_bs.copy()
    
    # Second round: local optimize + binary search
    if time.time() < DEADLINE - 10:
        c_opt2, r_opt2 = _local_optimize(best_points, time_limit=min(6.0, DEADLINE - (time.time() - t_start) - 10), beta_schedule=[200.0, 400.0, 800.0])
        if r_opt2 > best_score:
            best_score = r_opt2
            best_points = c_opt2.copy()
        
        c_bs2, r_bs2 = _binary_search_polish(best_points, best_score, min(best_score * 1.01, 0.13), time_limit=min(6.0, DEADLINE - (time.time() - t_start) - 10))
        if r_bs2 > best_score:
            best_score = r_bs2
            best_points = c_bs2.copy()
    
    # Multi-start: perturb best and try local optimization
    n_restarts = 0
    while time.time() < DEADLINE - 10 and n_restarts < 4:
        perturbation_scale = 0.01 / (1.0 + 0.3 * n_restarts)
        x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        n_restarts += 1
        
        try:
            c_opt, r_opt = _local_optimize(x0.reshape(N, 2), time_limit=min(6.0, DEADLINE - (time.time() - t_start) - 10))
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
        except Exception:
            pass
    
    # Final polish with high beta and binary search
    if time.time() < DEADLINE - 5:
        c_final, r_final = _local_optimize(best_points, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 5), beta_schedule=[800.0, 1600.0, 3200.0])
        if r_final > best_score:
            best_score = r_final
            best_points = c_final.copy()
        
        if time.time() < DEADLINE - 3:
            c_bs3, r_bs3 = _binary_search_polish(best_points, best_score, min(best_score * 1.005, 0.13), time_limit=min(3.0, DEADLINE - (time.time() - t_start) - 3))
            if r_bs3 > best_score:
                best_score = r_bs3
                best_points = c_bs3.copy()
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    return c_final
