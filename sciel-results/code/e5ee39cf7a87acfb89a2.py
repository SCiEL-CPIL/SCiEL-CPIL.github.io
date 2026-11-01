"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_score(pts):
        """Compute the exact packing radius for given centers."""
        if pts.shape != (N, 2):
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0 + 1e-9):
            return -1.0
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return max(0.0, r)

    def hex_lattice_init(scale=1.0, rotation=0.0):
        """Generate hexagonal lattice points clipped to unit circle."""
        spacing = 2.0 / (np.sqrt(N) * scale)
        nx = int(2.5 / spacing) + 3
        ny = int(2.5 / (spacing * np.sqrt(3.0) / 2.0)) + 3

        x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0

        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0:
                    pts_list.append([px, py])

        if len(pts_list) < N:
            return None

        pts = np.array(pts_list)
        # Apply rotation
        if rotation != 0.0:
            c, s = np.cos(rotation), np.sin(rotation)
            R = np.array([[c, -s], [s, c]])
            pts = pts @ R.T

        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:N]]
        return pts

    def ring_init(n_rings=12):
        """Generate ring-based initialization."""
        ring_radii = np.linspace(0.0, 0.95, n_rings)
        ring_points = []

        for r_ring in ring_radii:
            if r_ring < 0.01:
                ring_points.append([0.0, 0.0])
                continue
            # Estimate number of circles that fit on this ring
            est_spacing = 0.11  # approx 2*r for r~0.055
            n_circ = max(1, int(2 * np.pi * r_ring / est_spacing))
            angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            for a in angles:
                x = r_ring * np.cos(a)
                y = r_ring * np.sin(a)
                if np.hypot(x, y) <= 0.99:
                    ring_points.append([x, y])

        ring_points = np.array(ring_points)
        if len(ring_points) > N:
            norms = np.linalg.norm(ring_points, axis=1)
            idx = np.argsort(norms)[:N]
            ring_points = ring_points[idx]
        elif len(ring_points) < N:
            return None
        return ring_points

    def random_init(seed=42):
        """Generate random initialization."""
        rng = np.random.RandomState(seed)
        pts = rng.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        norms[norms == 0] = 1.0
        pts = pts * (0.95 / norms)[:, None]
        return pts

    def soft_min_val(x, y, beta):
        """Soft min of two scalars using log-sum-exp."""
        m = min(x, y)
        a = -beta * (x - m)
        b = -beta * (y - m)
        max_ab = max(a, b)
        if max_ab > 500:
            return m
        if max_ab < -500:
            return m + 1.0 / beta
        log_val = max_ab + np.log(np.exp(a - max_ab) + np.exp(b - max_ab))
        return m - (log_val) / beta

    def objective_and_grad(params, beta):
        """
        Smooth objective: negative soft-min radius.
        Returns (value, gradient).
        """
        if time.time() > deadline:
            return 1e10, np.zeros(2 * N)

        pts = params.reshape(-1, 2)

        if not np.all(np.isfinite(pts)):
            return 1e10, np.zeros(2 * N)

        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            penalty = np.sum(np.maximum(dists - 1.0, 0.0))
            return 1e10 + penalty, np.zeros(2 * N)

        # Pairwise distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)

        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)

        x = min_pair_dist / 2.0
        y = min_radial_dist

        val = soft_min_val(x, y, beta)
        fval = -val

        # Gradient computation
        grad = np.zeros(2 * N)

        # Gradient w.r.t. pairwise distance term
        # d(min_pair_dist)/d(pts) for the minimum pair
        min_pair_idx = np.argmin(pair_dists)
        ii = i_idx[min_pair_idx]
        jj = j_idx[min_pair_idx]
        diff_min = pts[ii] - pts[jj]
        dist_min = pair_dists[min_pair_idx]
        if dist_min > 1e-12:
            unit_min = diff_min / dist_min
            # d(min_pair_dist/2)/d(pts[ii]) = unit_min / 2
            # d(min_pair_dist/2)/d(pts[jj]) = -unit_min / 2
            grad[2*ii:2*ii+2] += 0.5 * unit_min
            grad[2*jj:2*jj+2] += -0.5 * unit_min

        # Gradient w.r.t. radial distance term
        max_radial_idx = np.argmax(dists)
        pt_max = pts[max_radial_idx]
        dist_max = dists[max_radial_idx]
        if dist_max > 1e-12:
            unit_max = pt_max / dist_max
            # d(1 - max_dist)/d(pts[max_radial_idx]) = -unit_max
            grad[2*max_radial_idx:2*max_radial_idx+2] += -unit_max

        # Combine gradients with soft-min weights
        # w_x = beta * exp(-beta*(x-m)) / (exp(-beta*(x-m)) + exp(-beta*(y-m)))
        # w_y = 1 - w_x
        m = min(x, y)
        ax = -beta * (x - m)
        ay = -beta * (y - m)
        max_ab = max(ax, ay)
        ex = np.exp(ax - max_ab)
        ey = np.exp(ay - max_ab)
        denom = ex + ey
        w_x = ex / denom
        w_y = ey / denom

        grad = w_x * grad  # This is incorrect - need to recompute properly

        # Recompute gradient properly
        grad = np.zeros(2 * N)

        # For the pairwise term
        min_pair_idx = np.argmin(pair_dists)
        ii = i_idx[min_pair_idx]
        jj = j_idx[min_pair_idx]
        diff_min = pts[ii] - pts[jj]
        dist_min = pair_dists[min_pair_idx]
        if dist_min > 1e-12:
            unit_min = diff_min / dist_min
            grad[2*ii:2*ii+2] += 0.5 * w_x * unit_min
            grad[2*jj:2*jj+2] += -0.5 * w_x * unit_min

        # For the radial term
        max_radial_idx = np.argmax(dists)
        pt_max = pts[max_radial_idx]
        dist_max = dists[max_radial_idx]
        if dist_max > 1e-12:
            unit_max = pt_max / dist_max
            grad[2*max_radial_idx:2*max_radial_idx+2] += -w_y * unit_max

        return fval, -grad  # Negative because we're minimizing -val

    def optimize_from(pts, betas, max_iters_per_beta):
        """Run progressive beta annealing optimization from initial points."""
        current_pts = pts.copy()
        current_score = compute_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score

        for beta in betas:
            if time.time() > deadline - 5:
                break

            x0 = current_pts.flatten()
            bounds = [(-0.9999, 0.9999)] * (2 * N)

            try:
                result = minimize(
                    objective_and_grad, x0, args=(beta,),
                    method='L-BFGS-B', bounds=bounds, jac=True,
                    options={'maxiter': max_iters_per_beta, 'ftol': 1e-14,
                             'gtol': 1e-12, 'maxfun': max_iters_per_beta * 5}
                )

                if result.fun < 1e9:
                    new_points = result.x.reshape(-1, 2)
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_points.copy()
                        current_pts = new_points.copy()
            except Exception:
                pass

        return best_pts, best_score

    # Generate multiple initializations
    inits = []

    # Hexagonal lattices with different scales and rotations
    for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
        for rot in [0.0, np.pi/6, np.pi/3]:
            pts = hex_lattice_init(scale, rot)
            if pts is not None and pts.shape == (N, 2):
                inits.append(pts)

    # Ring-based
    for nr in [10, 12, 14, 16]:
        pts = ring_init(nr)
        if pts is not None and pts.shape == (N, 2):
            inits.append(pts)

    # Random
    for seed in [42, 123, 456]:
        pts = random_init(seed)
        inits.append(pts)

    # Score all initializations
    init_scores = [(compute_score(p), i, p) for i, p in enumerate(inits)]
    init_scores.sort(key=lambda x: -x[0])

    # Take top diverse initializations
    selected = []
    for score, idx, pts in init_scores:
        if len(selected) >= 6:
            break
        # Check diversity
        is_diverse = True
        for sel_pts in selected:
            if np.allclose(pts, sel_pts, atol=0.01):
                is_diverse = False
                break
        if is_diverse:
            selected.append(pts)

    # Optimize from each selected initialization
    best_overall_pts = None
    best_overall_score = -1.0

    betas_stage1 = [20.0, 100.0, 500.0]
    betas_stage2 = [2000.0, 10000.0]

    for i, init_pts in enumerate(selected):
        if time.time() > deadline - 30:
            break

        # Stage 1: coarse to medium beta
        pts, score = optimize_from(init_pts, betas_stage1, max_iters_per_beta=40)

        # Stage 2: high beta polish
        pts, score = optimize_from(pts, betas_stage2, max_iters_per_beta=30)

        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts.copy()

    # Final ultra-high precision polish
    if best_overall_pts is not None and time.time() < deadline - 10:
        final_betas = [50000.0, 200000.0]
        best_overall_pts, best_overall_score = optimize_from(
            best_overall_pts, final_betas, max_iters_per_beta=50
        )

    # Final validation
    final_score = compute_score(best_overall_pts)
    if final_score <= 0 or best_overall_pts is None:
        # Fallback to best hexagonal
        pts = hex_lattice_init(1.0, 0.0)
        if pts is None:
            pts = random_init(42)
        return pts

    return best_overall_pts
# EVOLVE-BLOCK-END
