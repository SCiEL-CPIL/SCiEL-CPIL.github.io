"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Random initialization + Nelder-Mead with smooth soft-min objective.
    """
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin

    def get_radius(x):
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) / 2.0
        min_boundary = 1.0 - np.max(d_origin)
        return min(min_pair, min_boundary)

    def objective(x, sharpness):
        """
        Negative smooth min of all relevant distances.
        Uses log-sum-exp trick for numerical stability.
        """
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        half_pairs = np.sqrt(d2) / 2.0
        boundary = 1.0 - d_origin
        
        # Collect all distances
        all_d = np.concatenate([half_pairs.flatten(), boundary])
        all_d = np.maximum(all_d, 1e-15)
        
        # Soft min via log-sum-exp: min(d) ≈ -log(sum(exp(-alpha*d)))/alpha
        alpha = sharpness
        scaled = -alpha * all_d
        max_s = np.max(scaled)
        log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        soft_min = -log_sum / alpha
        
        return -soft_min

    def init_ring_pattern(seed):
        """Concentric ring initialization with jitter."""
        rng = np.random.default_rng(seed)
        pts = []
        rings = [(1, 0.0), (6, 0.25), (12, 0.50), (15, 0.75), (16, 0.92)]
        for n_ring, r_ring in rings:
            angles = 2 * np.pi * np.arange(n_ring) / n_ring + rng.uniform(0, 2 * np.pi)
            for a in angles:
                rr = r_ring + rng.normal(0, 0.01)
                pts.append([rr * np.cos(a), rr * np.sin(a)])
        pts = np.array(pts[:N])
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.98
        if np.any(too_far):
            pts[too_far] *= (0.98 / dists[too_far])[:, None]
        return pts.flatten()

    def init_hex_grid(seed):
        """Hexagonal lattice initialization."""
        rng = np.random.default_rng(seed)
        pts = []
        s = 0.22
        dx, dy = s, s * np.sqrt(3) / 2
        for row in range(-4, 5):
            for col in range(-4, 5):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                if x * x + y * y < 0.88:
                    pts.append([x, y])
        if len(pts) < N:
            # Fill remaining with random points
            while len(pts) < N:
                r = np.sqrt(rng.uniform(0, 0.88))
                a = rng.uniform(0, 2 * np.pi)
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts[:N])
        pts += rng.normal(0, 0.008, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            pts[too_far] *= (0.97 / dists[too_far])[:, None]
        return pts.flatten()

    def init_random_polar(seed):
        """Random polar coordinates with beta-distributed radii."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, N)
        r = rng.beta(2.5, 1.5, N) * 0.96
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts += rng.normal(0, 0.01, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            pts[too_far] *= (0.97 / dists[too_far])[:, None]
        return pts.flatten()

    def init_random_uniform(seed):
        """Uniform random in disk."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, N)
        r = np.sqrt(rng.uniform(0, 1, N)) * 0.96
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts += rng.normal(0, 0.008, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            pts[too_far] *= (0.97 / dists[too_far])[:, None]
        return pts.flatten()

    def init_fibonacci_sphere(seed):
        """Fibonacci sphere-like distribution projected to disk."""
        rng = np.random.default_rng(seed)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        pts = []
        for i in range(N):
            t = i / N
            z = 1.0 - 2.0 * t
            r = np.sqrt(max(0, 1.0 - z * z))
            theta = golden_angle * i + rng.uniform(0, 0.1)
            x = r * np.cos(theta) * 0.95
            y = r * np.sin(theta) * 0.95
            pts.append([x, y])
        pts = np.array(pts)
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.97
        if np.any(too_far):
            pts[too_far] *= (0.97 / dists[too_far])[:, None]
        return pts.flatten()

    def local_refine(x, n_iter=500):
        """Vectorized local refinement: move points away from nearest neighbors."""
        pts = x.reshape(N, 2).copy()
        for it in range(n_iter):
            if time.time() - start_time > time_budget - 5:
                break
            diff = pts[:, None, :] - pts[None, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)
            
            # For each point, find nearest neighbor and push away
            min_d = np.min(d, axis=1)
            min_idx = np.argmin(d, axis=1)
            
            # Push each point away from its nearest neighbor
            for i in range(N):
                j = min_idx[i]
                if d[i, j] < 0.25:  # Only if too close
                    vec = pts[i] - pts[j]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-10:
                        push = vec / norm * 0.002
                        pts[i] += push
            
            # Push toward center if near boundary
            d_origin = np.linalg.norm(pts, axis=1)
            for k in range(N):
                if d_origin[k] > 0.875:
                    pts[k] *= 0.875 / d_origin[k]
            
            # Clamp inside unit circle
            d_origin = np.linalg.norm(pts, axis=1)
            too_far = d_origin > 0.99
            if np.any(too_far):
                pts[too_far] *= (0.99 / d_origin[too_far])[:, None]
        
        return pts.flatten()

    # Generate diverse initializations
    inits = [
        init_ring_pattern(1),
        init_ring_pattern(2),
        init_ring_pattern(3),
        init_hex_grid(10),
        init_hex_grid(20),
        init_random_polar(30),
        init_random_polar(40),
        init_random_polar(50),
        init_random_uniform(60),
        init_random_uniform(70),
        init_fibonacci_sphere(80),
        init_fibonacci_sphere(90),
    ]

    best_x, best_r = None, -1.0

    # Phase 1: Coarse optimization (lower sharpness, more iterations per start)
    for i, x0 in enumerate(inits):
        if time.time() - start_time > time_budget - 30:
            break
        try:
            res = minimize(lambda x: objective(x, 20.0), x0,
                          method='Nelder-Mead',
                          options={'maxiter': 1500, 'maxfev': 2000,
                                   'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
            r = get_radius(res.x)
            if r > best_r:
                best_r, best_x = r, res.x.copy()
        except Exception:
            pass

    if best_x is None:
        return init_ring_pattern(1).reshape(N, 2)

    # Phase 2: Medium sharpness
    if time.time() - start_time < time_budget - 20:
        try:
            res2 = minimize(lambda x: objective(x, 40.0), best_x,
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'maxfev': 3000,
                                    'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True})
            r2 = get_radius(res2.x)
            if r2 > best_r:
                best_x, best_r = res2.x.copy(), r2
        except Exception:
            pass

    # Phase 3: Local refinement
    if time.time() - start_time < time_budget - 15:
        try:
            refined = local_refine(best_x, n_iter=400)
            r3 = get_radius(refined)
            if r3 > best_r:
                best_x, best_r = refined, r3
        except Exception:
            pass

    # Phase 4: Higher sharpness after refinement
    if time.time() - start_time < time_budget - 10:
        try:
            res4 = minimize(lambda x: objective(x, 80.0), best_x,
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'maxfev': 3000,
                                    'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True})
            r4 = get_radius(res4.x)
            if r4 > best_r:
                best_x, best_r = res4.x.copy(), r4
        except Exception:
            pass

    # Phase 5: Another local refinement
    if time.time() - start_time < time_budget - 5:
        try:
            refined2 = local_refine(best_x, n_iter=300)
            r5 = get_radius(refined2)
            if r5 > best_r:
                best_x, best_r = refined2, r5
        except Exception:
            pass

    # Phase 6: Very high sharpness fine-tune
    if time.time() - start_time < time_budget - 2:
        try:
            res6 = minimize(lambda x: objective(x, 150.0), best_x,
                           method='Nelder-Mead',
                           options={'maxiter': 1500, 'maxfev': 2000,
                                    'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
            r6 = get_radius(res6.x)
            if r6 > best_r:
                best_x, best_r = res6.x.copy(), r6
        except Exception:
            pass

    # Final local polish
    if time.time() - start_time < time_budget - 1:
        try:
            final_refined = local_refine(best_x, n_iter=200)
            r7 = get_radius(final_refined)
            if r7 > best_r:
                best_x, best_r = final_refined, r7
        except Exception:
            pass

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
