import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 30.0
    N = 16

    def get_score(pts):
        """Compute R = (d_min / d_max)^2. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def get_cost(pts):
        """Return 1/R for minimization. Lower is better."""
        s = get_score(pts)
        if s < 1e-12:
            return np.inf
        return 1.0 / s

    def get_cost_vec(params):
        pts = params.reshape(16, 2)
        return get_cost(pts)

    def smooth_obj_and_grad(params, k):
        """
        Smooth objective: minimize soft_max(d) / soft_min(d) where d are pairwise distances.
        Uses log-sum-exp for numerical stability.
        """
        pts = params.reshape(16, 2)
        d = pdist(pts)
        d = np.maximum(d, 1e-12)

        # Soft max: (1/k) * log(sum(exp(k*d)))
        # Soft min: -(1/k) * log(sum(exp(-k*d)))
        kd = k * d
        # Shift for numerical stability
        kd_max = np.max(kd)
        kd_min = np.min(kd)

        soft_max = (1.0 / k) * (kd_max + np.log(np.sum(np.exp(kd - kd_max))))
        soft_min = -(1.0 / k) * (kd_min + np.log(np.sum(np.exp(-kd - kd_min))))

        if soft_min < 1e-12:
            return 1e10, np.zeros_like(params)

        cost = soft_max / soft_min

        # Gradients
        # d(soft_max)/d(d_i) = exp(k*d_i) / sum(exp(k*d_j))
        # d(soft_min)/d(d_i) = -exp(-k*d_i) / sum(exp(-k*d_j))
        exp_kd = np.exp(kd - kd_max)
        sum_exp_kd = np.sum(exp_kd)
        grad_soft_max = exp_kd / sum_exp_kd  # d(soft_max)/d(d_i)

        neg_kd = -kd - kd_min
        exp_neg_kd = np.exp(neg_kd)
        sum_exp_neg_kd = np.sum(exp_neg_kd)
        grad_soft_min = -exp_neg_kd / sum_exp_neg_kd  # d(soft_min)/d(d_i)

        # d(cost)/d(d_i) = (grad_soft_max * soft_min - soft_max * grad_soft_min) / soft_min^2
        d_cost_d_d = (grad_soft_max * soft_min - soft_max * grad_soft_min) / (soft_min ** 2)

        # Now chain rule: d(cost)/d(pts) via d(cost)/d(d_i) * d(d_i)/d(pts)
        # d_i = ||p_a - p_b|| for pair (a,b)
        # d(d_i)/d(p_a) = (p_a - p_b) / d_i
        # d(d_i)/d(p_b) = (p_b - p_a) / d_i

        # Build mapping from pdist index to pair (a,b)
        # pdist uses upper triangle: index i corresponds to pair (i//2, i%2) is wrong
        # Actually pdist for n points: pairs are (0,1),(0,2),...,(0,n-1),(1,2),...,(n-2,n-1)
        n = 16
        # Compute pair indices
        pair_idx = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_idx.append((i, j))
                idx += 1

        grad_pts = np.zeros((16, 2))
        for idx2, (a, b) in enumerate(pair_idx):
            if d[idx2] < 1e-12:
                continue
            diff = pts[a] - pts[b]
            # d(d_idx2)/d(pts[a]) = diff / d_idx2
            # d(d_idx2)/d(pts[b]) = -diff / d_idx2
            grad_pts[a] += d_cost_d_d[idx2] * diff / d[idx2]
            grad_pts[b] += d_cost_d_d[idx2] * (-diff) / d[idx2]

        return cost, grad_pts.flatten()

    def smooth_obj(params, k):
        c, _ = smooth_obj_and_grad(params, k)
        return c

    def smooth_grad(params, k):
        _, g = smooth_obj_and_grad(params, k)
        return g

    def hexagonal_init():
        """Create 16 points from a hexagonal lattice subset."""
        # Hexagonal lattice points
        a = 1.0
        points = []
        for row in range(-3, 4):
            for col in range(-3, 4):
                x = col * a + (row % 2) * a / 2
                y = row * a * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        # Select 16 points closest to center
        dists = np.sum(points ** 2, axis=1)
        sorted_idx = np.argsort(dists)
        sel = points[sorted_idx[:16]]
        # Center
        sel = sel - np.mean(sel, axis=0)
        return sel

    def ring_init(n_rings, points_per_ring, radius_ratio, phase_offset):
        """Create points on concentric rings."""
        pts_list = []
        for ring in range(n_rings):
            r = radius_ratio ** ring
            n = points_per_ring[ring] if isinstance(points_per_ring, list) else points_per_ring
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase_offset * ring
            pts_ring = r * np.column_stack([np.cos(angles), np.sin(angles)])
            pts_list.append(pts_ring)
        pts = np.vstack(pts_list)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def regular_polygon_init(n, radius=1.0):
        """Regular n-gon."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        pts = pts - np.mean(pts, axis=0)
        return pts

    # Generate diverse initial configurations
    initial_configs = []

    # 1. Two octagons with various parameters
    for ratio in [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]:
        for phase in [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]:
            n = 8
            inner_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            outer_angles = inner_angles + phase
            inner_pts = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            outer_pts = ratio * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            pts = np.vstack([inner_pts, outer_pts])
            pts = pts - np.mean(pts, axis=0)
            initial_configs.append(pts)

    # 2. Hexagonal lattice subset
    initial_configs.append(hexagonal_init())

    # 3. 4x4 grid
    g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    g = g - np.mean(g, axis=0)
    initial_configs.append(g)

    # 4. Regular 16-gon
    initial_configs.append(regular_polygon_init(16))

    # 5. Three rings: 4 + 8 + 4
    for r1, r2 in [(0.5, 1.0), (0.6, 1.0), (0.7, 1.0), (0.8, 1.0)]:
        a4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        a8 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        pts1 = r1 * np.column_stack([np.cos(a4), np.sin(a4)])
        pts2 = r2 * np.column_stack([np.cos(a8), np.sin(a8)])
        pts3 = r1 * np.column_stack([np.cos(a4 + np.pi/4), np.sin(a4 + np.pi/4)])
        pts = np.vstack([pts1, pts2, pts3])
        pts = pts - np.mean(pts, axis=0)
        initial_configs.append(pts)

    # 6. Four rings: 4 + 4 + 4 + 4 with offsets
    for r1, r2, r3 in [(0.3, 0.6, 1.0), (0.4, 0.7, 1.0), (0.5, 0.8, 1.0)]:
        a4a = np.linspace(0, 2*np.pi, 4, endpoint=False)
        a4b = a4a + np.pi/4
        a4c = a4a
        a4d = a4b
        p1 = r1 * np.column_stack([np.cos(a4a), np.sin(a4a)])
        p2 = r2 * np.column_stack([np.cos(a4b), np.sin(a4b)])
        p3 = r3 * np.column_stack([np.cos(a4c), np.sin(a4c)])
        p4 = r3 * np.column_stack([np.cos(a4d), np.sin(a4d)])
        pts = np.vstack([p1, p2, p3, p4])
        pts = pts - np.mean(pts, axis=0)
        initial_configs.append(pts)

    # 7. Add random perturbations to some configs
    np.random.seed(42)
    for cfg in initial_configs[:10]:
        perturbed = cfg + np.random.normal(0, 0.1, cfg.shape)
        initial_configs.append(perturbed)

    # Find best initial configuration
    best_pts = None
    best_cost = np.inf
    for cfg in initial_configs:
        c = get_cost(cfg)
        if c < best_cost:
            best_cost = c
            best_pts = cfg.copy()

    # Phase 2: Smooth gradient-based optimization with adaptive k
    time_elapsed = time.time() - start_time
    if (TIME_LIMIT - time_elapsed) > 60 and best_pts is not None:
        # Start with moderate k, then increase
        k_values = [20, 50, 100, 200, 500]
        for k in k_values:
            time_elapsed = time.time() - start_time
            if (TIME_LIMIT - time_elapsed) < 30:
                break
            try:
                res = minimize(
                    lambda p: smooth_obj(p, k),
                    best_pts.flatten(),
                    jac=lambda p: smooth_grad(p, k),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 50}
                )
                if np.isfinite(res.fun):
                    res_pts = res.x.reshape(16, 2)
                    res_cost = get_cost(res_pts)
                    if res_cost < best_cost:
                        best_pts = res_pts
                        best_cost = res_cost
            except Exception:
                pass

    # Phase 3: Simulated Annealing for global exploration
    time_elapsed = time.time() - start_time
    if (TIME_LIMIT - time_elapsed) > 120 and best_pts is not None:
        sa_start_temp = 0.5
        sa_end_temp = 0.001
        sa_iterations = 5000
        current_pts = best_pts.copy()
        current_cost = best_cost
        best_sa_pts = best_pts.copy()
        best_sa_cost = best_cost

        for i in range(sa_iterations):
            time_elapsed = time.time() - start_time
            if (TIME_LIMIT - time_elapsed) < 60:
                break

            # Temperature schedule
            temp = sa_start_temp * (sa_end_temp / sa_start_temp) ** (i / sa_iterations)

            # Generate perturbation
            scale = 0.1 * temp
            perturbation = np.random.normal(0, scale, (16, 2))
            new_pts = current_pts + perturbation
            new_cost = get_cost(new_pts)

            # Acceptance criterion
            if new_cost < current_cost or np.random.random() < np.exp(-(new_cost - current_cost) / temp):
                current_pts = new_pts
                current_cost = new_cost
                if new_cost < best_sa_cost:
                    best_sa_pts = new_pts.copy()
                    best_sa_cost = new_cost

        if best_sa_cost < best_cost:
            best_pts = best_sa_pts
            best_cost = best_sa_cost

    # Phase 4: Nelder-Mead polish on the exact objective
    time_elapsed = time.time() - start_time
    if (TIME_LIMIT - time_elapsed) > 60 and best_pts is not None:
        try:
            res3 = minimize(get_cost_vec, best_pts.flatten(), method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if np.isfinite(res3.fun):
                res_pts3 = res3.x.reshape(16, 2)
                res_cost3 = get_cost(res_pts3)
                if res_cost3 < best_cost:
                    best_pts = res_pts3
                    best_cost = res_cost3
        except Exception:
            pass

    # Phase 5: Final smooth polish with high k
    time_elapsed = time.time() - start_time
    if (TIME_LIMIT - time_elapsed) > 30 and best_pts is not None:
        try:
            res4 = minimize(
                lambda p: smooth_obj(p, 1000),
                best_pts.flatten(),
                jac=lambda p: smooth_grad(p, 1000),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-13, 'ftol': 1e-15, 'maxls': 40}
            )
            if np.isfinite(res4.fun):
                res_pts4 = res4.x.reshape(16, 2)
                res_cost4 = get_cost(res_pts4)
                if res_cost4 < best_cost:
                    best_pts = res_pts4
                    best_cost = res_cost4
        except Exception:
            pass

    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)):
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        g = g - np.mean(g, axis=0)
        final_pts = g

    return final_pts
