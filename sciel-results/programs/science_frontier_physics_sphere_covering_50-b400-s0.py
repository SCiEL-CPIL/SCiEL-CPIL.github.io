import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def compute_hull_data(p):
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None
        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None
        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]
        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        return facets, circumcenters, radii

    def get_max_radius(pts):
        _, _, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def epigraph_objective(params, t, beta=200.0):
        X = normalize(params.reshape(N, 3))
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        r_max = np.max(radii)
        shifted = beta * (radii - r_max)
        shifted = np.clip(shifted, -50, 50)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp + beta * max(0.0, t - log_sum_exp)

    def soft_max_objective(params, beta=200.0):
        X = normalize(params.reshape(N, 3))
        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10
        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        shifted = np.clip(shifted, -50, 50)
        return np.log(np.sum(np.exp(shifted))) / beta + r_max

    def repulsion_relax(pts, n_iters=500, alpha_start=0.05):
        pts = pts.copy()
        alpha = alpha_start
        current_energy = None

        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)

        for it in range(n_iters):
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts

            step = alpha
            improved = False
            for _ in range(8):
                new_pts = normalize(pts + step * forces_tan)
                new_energy = repulsion_energy(new_pts)
                if current_energy is None or new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5

            if not improved:
                alpha *= 0.8
            else:
                alpha *= 1.01
            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-6)

            if time.time() > deadline - 5.0:
                break
        return pts

    def facet_local_search(pts, n_iters=120):
        pts = pts.copy()
        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            threshold = max_rad * 0.75
            large_facets = np.where(radii > threshold)[0]
            if len(large_facets) == 0:
                break

            updates = np.zeros_like(pts)
            counts = np.zeros(N)
            for f in large_facets:
                facet_pts = facets[f]
                cc = circumcenters[f]
                rad = radii[f]
                if rad < 1e-12:
                    continue
                weight = ((rad - threshold) / (max_rad - threshold + 1e-15)) ** 2
                weight = max(weight, 0.1)
                for idx in facet_pts:
                    updates[idx] += weight * (cc - pts[idx])
                    counts[idx] += weight

            mask = counts > 1e-10
            if np.any(mask):
                mean_rad = np.mean(radii)
                gap = max_rad - mean_rad
                step_size = 0.1 * (gap / (mean_rad + 1e-15))
                step_size = min(step_size, 0.2)
                step_size = max(step_size, 0.01)

                best_pts_trial = pts.copy()
                best_rad_trial = max_rad
                for trial_step in [step_size, step_size * 0.5, step_size * 0.25, step_size * 0.125, step_size * 0.0625]:
                    trial_pts = pts.copy()
                    trial_pts[mask] += trial_step * updates[mask] / counts[mask, np.newaxis]
                    trial_pts = normalize(trial_pts)
                    trial_rad = get_max_radius(trial_pts)
                    if trial_rad < best_rad_trial:
                        best_rad_trial = trial_rad
                        best_pts_trial = trial_pts.copy()

                if best_rad_trial < max_rad:
                    pts = best_pts_trial
                else:
                    break

            pts = normalize(pts)
            if time.time() > deadline - 5.0:
                break
        return pts

    def icosahedral_init():
        t = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        ico_faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        points = [ico[i] for i in range(12)]
        edge_mids = set()
        for face in ico_faces:
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                if (a, b) not in edge_mids:
                    edge_mids.add((a, b))
                    mid = (ico[a] + ico[b]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    points.append(mid)
        face_centers = []
        for face in ico_faces:
            center = np.mean(ico[face], axis=0)
            center = center / np.linalg.norm(center)
            face_centers.append(center)
        all_candidates = np.array(points + face_centers)
        selected = [all_candidates[0]]
        for _ in range(N - 1):
            min_dists = np.array([min(np.linalg.norm(cand - s) for s in selected) for cand in all_candidates])
            for s in selected:
                idx = np.where(np.all(all_candidates == s, axis=1))[0]
                if len(idx) > 0:
                    min_dists[idx] = -1
            best_idx = np.argmax(min_dists)
            selected.append(all_candidates[best_idx])
        return np.array(selected)

    def greedy_init(seed_val):
        np.random.seed(seed_val)
        pts = [np.random.randn(3)]
        pts[0] = pts[0] / np.linalg.norm(pts[0])
        for _ in range(N - 1):
            candidates = np.random.randn(3000, 3)
            candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)

    # Generate initial configurations
    initial_configs = []

    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initial_configs.append(fib_pts)

    # Fibonacci variants
    for phase in [0.37, 0.618]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + phase)
        r2 = np.sqrt(1.0 - z2 * z2)
        fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initial_configs.append(fib2_pts)

    # Icosahedral
    try:
        ico_pts = icosahedral_init()
        initial_configs.append(ico_pts)
    except Exception:
        pass

    # Random
    for seed_offset in range(2):
        np.random.seed(42 + seed_offset * 100)
        pts = np.random.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        initial_configs.append(pts)

    # Greedy
    initial_configs.append(greedy_init(101))
    initial_configs.append(greedy_init(202))

    best_pts = initial_configs[0].copy()
    best_radius = get_max_radius(best_pts)

    # Phase 1: Repulsion relaxation
    for idx, init_pts in enumerate(initial_configs[:6]):
        if time.time() > deadline - 60.0:
            break
        relaxed = repulsion_relax(init_pts, n_iters=200, alpha_start=0.03)
        rad = get_max_radius(relaxed)
        if rad < best_radius:
            best_radius = rad
            best_pts = relaxed.copy()

    # Phase 2: Soft-max optimization with annealing
    beta_schedule = [30, 80, 150, 300, 600, 1200, 2500]
    for beta in beta_schedule:
        if time.time() > deadline - 25.0:
            break

        def obj_sm(params, _beta=beta):
            return soft_max_objective(params, beta=_beta)

        starts_to_try = [best_pts]
        if len(initial_configs) > 1:
            starts_to_try.append(initial_configs[1])

        for start_idx, init_pts in enumerate(starts_to_try):
            if time.time() > deadline - 25.0:
                break
            maxiter = 50 if start_idx == 0 else 30
            try:
                res = minimize(obj_sm, init_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 300})
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass

    # Phase 3: Facet-based local search
    if time.time() < deadline - 15.0:
        facet_pts = facet_local_search(best_pts, n_iters=100)
        rad = get_max_radius(facet_pts)
        if rad < best_radius:
            best_radius = rad
            best_pts = facet_pts.copy()

    # Phase 4: More soft-max refinement
    if time.time() < deadline - 10.0:
        for beta in [500, 1000, 2000]:
            if time.time() > deadline - 10.0:
                break
            def obj_refine(params, _beta=beta):
                return soft_max_objective(params, beta=_beta)
            try:
                res = minimize(obj_refine, best_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 60, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 500})
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass

    # Phase 5: Final facet polish
    if time.time() < deadline - 5.0:
        facet_pts2 = facet_local_search(best_pts, n_iters=60)
        rad = get_max_radius(facet_pts2)
        if rad < best_radius:
            best_radius = rad
            best_pts = facet_pts2.copy()

    # Final high-precision polish
    if time.time() < deadline - 2.0:
        def obj_final(params):
            return soft_max_objective(params, beta=5000.0)
        try:
            res = minimize(obj_final, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 1000})
            new_pts = normalize(res.x.reshape(N, 3))
            current_radius = get_max_radius(new_pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = new_pts.copy()
        except Exception:
            pass

    # Final facet polish
    if time.time() < deadline - 0.5:
        facet_pts3 = facet_local_search(best_pts, n_iters=40)
        rad = get_max_radius(facet_pts3)
        if rad < best_radius:
            best_radius = rad
            best_pts = facet_pts3.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_pts
