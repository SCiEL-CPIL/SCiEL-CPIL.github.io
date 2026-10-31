"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill
    rng = np.random.default_rng(12345)

    # ---- Initialization: grid + jitter, then a few random restarts ----
    def make_grid_init(seed):
        r = np.random.default_rng(seed)
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append([(i + 0.5) / side, (j + 0.5) / side])
        # Shuffle so we take a spread
        r.shuffle(cells)
        centers = np.array(cells[:N], dtype=float)
        # Small jitter to break symmetry
        centers += r.uniform(-0.02, 0.02, size=centers.shape)
        centers = np.clip(centers, 0.0, 1.0)
        radii = np.full(N, 0.5 / side - 0.01, dtype=float)
        return centers, radii

    centers, radii = make_grid_init(0)

    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = -np.inf

    def compute_score(c, rad):
        # Sum of radii
        s = rad.sum()
        # Feasibility checks (soft)
        # Boundary: center - r >= 0, center + r <= 1
        # Pairwise: dist >= r_i + r_j
        return s

    def gradient_step(c, rad, lr, iters):
        c = c.copy()
        rad = rad.copy()
        n = len(rad)
        # Precompute pairwise indices
        ii, jj = np.triu_indices(n, k=1)
        for it in range(iters):
            if time.time() > deadline:
                break
            # Gradients
            g_c = np.zeros_like(c)
            g_r = np.zeros_like(rad)

            # Objective: maximize sum r -> grad r += 1
            g_r += 1.0

            # Boundary constraints (soft penalty)
            # c - r >= 0  =>  penalty if violated
            eps = 1e-6
            # Lower bound: c_x - r < 0
            for dim in range(2):
                viol_low = np.maximum(0.0, -(c[:, dim] - rad))
                viol_high = np.maximum(0.0, (c[:, dim] + rad) - 1.0)
                # Smooth penalty: quadratic
                g_c[:, dim] += -2.0 * viol_low + 2.0 * viol_high
                g_r += 2.0 * viol_low + 2.0 * viol_high

            # Pairwise non-overlap: dist >= r_i + r_j
            d = c[ii] - c[jj]
            dist = np.sqrt(np.sum(d**2, axis=1) + eps)
            overlap = np.maximum(0.0, rad[ii] + rad[jj] - dist)
            # Gradient wrt centers: for pair (i,j), grad c_i += 2*overlap * (d_ij / dist)
            # Use vectorized accumulation
            for k in range(len(ii)):
                i, j = ii[k], jj[k]
                if overlap[k] > 0:
                    dir_ = d[k] / dist[k]
                    g_c[i] += 2.0 * overlap[k] * dir_
                    g_c[j] += -2.0 * overlap[k] * dir_
                    g_r[i] += 2.0 * overlap[k]
                    g_r[j] += 2.0 * overlap[k]

            # Gradient ascent with step size
            c += lr * g_c
            rad += lr * g_r
            rad = np.maximum(rad, 1e-6)
            # Clip centers to stay in [r, 1-r] approximately
            rad_safe = np.maximum(rad, 1e-6)
            c = np.clip(c, rad_safe, 1.0 - rad_safe)

            if it % 50 == 0:
                s = compute_score(c, rad)
                if s > best_score:
                    best_score = s
                    best_centers = c.copy()
                    best_radii = rad.copy()

        s = compute_score(c, rad)
        if s > best_score:
            best_score = s
            best_centers = c.copy()
            best_radii = rad.copy()
        return c, rad

    # Multiple phases with decreasing learning rate
    phases = [
        (0.005, 300),
        (0.002, 300),
        (0.001, 300),
        (0.0005, 400),
        (0.0002, 400),
    ]

    for lr, iters in phases:
        if time.time() > deadline:
            break
        centers, radii = gradient_step(centers, radii, lr, iters)

    # Final polish: more iterations with tiny step
    if time.time() < deadline:
        centers, radii = gradient_step(centers, radii, 1e-5, 500)

    # Ensure validity: clip and shrink if needed
    c = best_centers.copy()
    rad = best_radii.copy()
    # Enforce boundary strictly
    for i in range(N):
        rad[i] = min(rad[i], c[i][0], 1.0 - c[i][0], c[i][1], 1.0 - c[i][1])
        rad[i] = max(rad[i], 1e-6)
        c[i] = np.clip(c[i], rad[i], 1.0 - rad[i])
    # Enforce pairwise non-overlap by shrinking
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((c[i] - c[j])**2))
                req = rad[i] + rad[j]
                if d < req:
                    # Shrink both
                    shrink = (req - d) / 2.0 + 1e-6
                    rad[i] = max(rad[i] - shrink, 1e-6)
                    rad[j] = max(rad[j] - shrink, 1e-6)
                    changed = True
        if not changed:
            break

    result = np.column_stack([c, rad])
    return result
# EVOLVE-BLOCK-END
