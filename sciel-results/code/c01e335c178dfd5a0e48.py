"""
Packing 50 equal circles in a unit circle.
Combined approach: ring-structured initialization + multi-stage smooth optimization
with log-sum-exp soft-min, binary-search polishing, and direct constraint repair.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 50
HARD_LIMIT = 145.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score: minimum of half pairwise distances and boundary distances."""
    pts = pts.reshape(-1, 2)
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return max(min(min_pair, min_boundary), 0.0)


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp for numerical stability."""
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum == 0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    """
    Smooth objective: minimize -softmin of all constraint margins.
    The softmin approximates the minimum, so minimizing -softmin maximizes the minimum.
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10

    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0

    # Get all finite values
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]

    if len(all_vals) == 0:
        return 1e10

    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _ring_initialization(ring_radii, ring_counts, angles_offset=None):
    """Generate points from a ring structure."""
    centers = []
    for idx, (count, radius) in enumerate(zip(ring_counts, ring_radii)):
        if count == 1 and radius < 1e-10:
            centers.append(np.array([0.0, 0.0]))
        else:
            if angles_offset is not None and idx < len(angles_offset):
                offset = angles_offset[idx]
            else:
                offset = 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for i in range(count):
                centers.append(np.array([xs[i], ys[i]]))

    pts = np.array(centers[:N])
    # Project inside unit circle
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def _hexagonal_init(n):
    """
    Initialize with a hexagonal lattice structure.
    This is a strong starting point for circle packing.
    """
    # Generate hexagonal grid points
    a = 1.0
    sqrt3 = np.sqrt(3)

    def hex_to_cart(q, r):
        x = a * 1.5 * q
        y = a * sqrt3 * (r + q / 2.0)
        return np.array([x, y])

    max_ring = 5
    candidates = []
    for q in range(-max_ring, max_ring + 1):
        for r in range(-max_ring, max_ring + 1):
            candidates.append((q, r))

    cart_points = []
    for q, r in candidates:
        p = hex_to_cart(q, r)
        d = np.linalg.norm(p)
        cart_points.append((d, p))

    cart_points.sort(key=lambda x: x[0])
    selected_points = [p for d, p in cart_points[:N]]
    points = np.array(selected_points)

    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        scale = 0.8 / max_r
        points = points * scale

    return points


def _ring_init(n, n_rings=5):
    """
    Initialize with ring structure matching known optimal packing topology.
    Ring 0: center (1 circle)
    Ring 1: 6 circles
    Ring 2: 12 circles
    Ring 3: 18 circles
    Ring 4: 13 circles (remaining)
    """
    points = []

    # Center
    points.append([0.0, 0.0])

    # Ring 1: 6 circles at radius ~0.25
    r1 = 0.25
    for k in range(6):
        angle = 2 * np.pi * k / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])

    # Ring 2: 12 circles at radius ~0.5
    r2 = 0.5
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 6  # offset
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])

    # Ring 3: 18 circles at radius ~0.75
    r3 = 0.75
    for k in range(18):
        angle = 2 * np.pi * k / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])

    # Ring 4: remaining circles at radius ~0.9
    remaining = n - len(points)
    r4 = 0.9
    for k in range(remaining):
        angle = 2 * np.pi * k / remaining + np.pi / remaining
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])

    return np.array(points[:n])


def _optimize_from_init(pts, deadline, max_beta_stages=5):
    """Multi-stage annealing optimization from initial points."""
    current_pts = pts.copy()
    current_score = _score(current_pts)

    # Progressive beta values for smooth optimization
    betas = [5.0, 15.0, 40.0, 100.0, 250.0]

    for beta in betas:
        if time.time() > deadline - 10:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 5000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass

    # High-precision polish with larger betas
    for beta in [500.0, 1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 5:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 20000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass

    return current_pts, current_score


def _binary_search_polish(pts, r_target, deadline, max_iters=500):
    """
    Polish by iteratively pushing points to achieve target radius r_target.
    This is a direct constraint satisfaction approach.
    """
    pts = pts.copy()

    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break

        # Compute current margins
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))

        # Find the most violated constraint
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)

        if min_pair_margin < min_boundary_margin - 1e-14:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is binding
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di

        # Check convergence
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break

    return pts


def _de_ring_optimize(deadline):
    """
    Differential evolution over ring parameters to find good ring configurations.
    """
    from scipy.optimize import differential_evolution

    t_start = time.time()

    # Typical ring structure for 50 circles
    ring_counts = [1, 6, 12, 18, 13]
    n_rings = len(ring_counts)

    def ring_params_to_points(params):
        """Convert ring parameters to point coordinates."""
        points = []
        # Center point
        points.append([0.0, 0.0])

        for i, (count, radius) in enumerate(zip(ring_counts, params[:n_rings * 2 : 2])):
            offset = params[i * 2 + 1]
            if count == 1:
                continue  # Already added center
            for k in range(int(count)):
                angle = 2 * np.pi * k / count + offset
                points.append([radius * np.cos(angle), radius * np.sin(angle)])

        # Pad or trim to N points
        points = np.array(points[:N])
        if len(points) < N:
            remaining = N - len(points)
            extra = np.random.randn(remaining, 2) * 0.01
            points = np.vstack([points, extra])

        return points[:N]

    # Define bounds for ring parameters: [radius, offset] for each ring (except center)
    bounds = [(0.0, 0.95), (0.0, 2 * np.pi)] * (n_rings - 1)

    def objective(x):
        if time.time() - t_start > (deadline - t_start - 5):
            return 1e10
        points = ring_params_to_points(x)
        return -_score(points)

    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=30,
            tol=1e-6,
            polish=False,
            init='sobol'
        )

        if result.success:
            best_params = result.x
            points = ring_params_to_points(best_params)
            score = _score(points)
            return points, score
    except Exception:
        pass

    return None, 0.0


def _local_optimize(c, time_limit=10.0, beta_schedule=None):
    """Local optimization using L-BFGS-B with smooth-min objective."""
    pair_idx = _pairwise_indices(N)
    n_pairs = len(pair_idx)

    if beta_schedule is None:
        beta_schedule = [200.0, 300.0, 500.0]

    best_c = c.copy()
    best_r = _score(best_c)

    t_start = time.time()

    for beta in beta_schedule:
        if time.time() - t_start >= time_limit - 2:
            break

        def neg_soft_min(x):
            c_local = x.reshape(N, 2)
            ci = c_local[pair_idx[:, 0]]
            cj = c_local[pair_idx[:, 1]]
            d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
            b = 1 - np.hypot(c_local[:, 0], c_local[:, 1])
            all_vals = np.concatenate([d, b])
            return np.log(np.sum(np.exp(-beta * all_vals))) / beta

        def callback(x):
            if time.time() - t_start >= time_limit - 1:
                raise Exception("timeout")

        try:
            res = minimize(
                neg_soft_min,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-15},
                callback=callback
            )
            if res.success and np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass

    return best_c, best_r


def _anneal(points, t_start, deadline, seed=None):
    """Simulated annealing to escape local minima."""
    if seed is not None:
        np.random.seed(seed)

    n = len(points)
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(best_pts)

    # SA parameters
    T = 0.05  # initial temperature
    T_min = 0.0001
    cooling = 0.9995
    step_size = 0.02

    while time.time() < deadline:
        if T < T_min:
            break

        # Propose a move: random perturbation of one or two points
        n_moves = np.random.randint(1, 4)
        new_pts = pts.copy()
        for _ in range(n_moves):
            idx = np.random.randint(n)
            delta = np.random.randn(2) * step_size * (1 + T)
            new_pts[idx] += delta

        new_score = _score(new_pts)

        # Accept with probability exp(-(new_score - best_score) / T) if new_score < best_score
        if new_score > best_score:
            pts = new_pts
            best_score = new_score
            best_pts = new_pts.copy()
        else:
            delta_cost = best_score - new_score
            if np.random.rand() < np.exp(-delta_cost / T):
                pts = new_pts

        T *= cooling

    return best_pts, best_score


def construct():
    t_start = time.time()
    deadline = t_start + HARD_LIMIT

    # Start with multiple initializations and pick the best
    inits = [
        _hexagonal_init(N),
        _ring_init(N),
    ]

    best_points = None
    best_score = 0.0

    for init in inits:
        r = _score(init)
        if r > best_score:
            best_score = r
            best_points = init.copy()

    # Global search over ring parameter space
    if time.time() < deadline - 60:
        ring_points, ring_score = _de_ring_optimize(deadline)
        if ring_points is not None and ring_score > best_score:
            best_score = ring_score
            best_points = ring_points.copy()

    # Multi-stage smooth optimization from best init
    if time.time() < deadline - 30:
        c_opt, r_opt = _optimize_from_init(best_points, deadline - 5)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()

    # Binary search polish for final refinement
    if time.time() < deadline - 20:
        bs_points = _binary_search_polish(best_points, best_score, deadline - 5)
        bs_score = _score(bs_points)
        if bs_score > best_score:
            best_score = bs_score
            best_points = bs_points.copy()

    # Final local optimization with high beta
    if time.time() < deadline - 10:
        c_final, r_final = _local_optimize(best_points, time_limit=min(8.0, deadline - (time.time() - t_start) - 10), beta_schedule=[500.0, 800.0])
        if r_final > best_score:
            best_points = c_final
            best_score = r_final

    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale

    return c_final
