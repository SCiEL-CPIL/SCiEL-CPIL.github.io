"""Cover the unit sphere with 50 equal caps.

Strategy:
  1. Build diverse initial configurations (Fibonacci spiral, random, antipodal).
  2. Run a fast Lloyd iteration to get a good starting point.
  3. Use a gradient-based refinement that targets the worst facets.
  4. Final high-precision polish with very small steps.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 50
DEADLINE_SECONDS = 115  # safety margin below 150s hard limit

def _covering_radius_deg(pts):
    """Compute the covering radius in degrees using convex hull circumcenters."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: dense sampling
        n_samples = 20000
        i_s = np.arange(n_samples) + 0.5
        z_s = 1.0 - 2.0 * i_s / n_samples
        phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
        r_s = np.sqrt(np.maximum(0, 1.0 - z_s * z_s))
        samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
        max_dist = 0.0
        chunk = 5000
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            s = samples[start:end]
            dots = s @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            min_dist = np.arccos(dots).min(axis=1)
            max_dist = max(max_dist, min_dist.max())
        return math.degrees(max_dist), None, None

    max_circ_r = 0.0
    worst_simplex = None
    worst_cc = None

    for simplex in hull.simplices:
        tri = pts[simplex]
        a, b, c = tri[0], tri[1], tri[2]

        # Spherical circumcenter
        ab = np.cross(a, b)
        bc = np.cross(b, c)
        ca = np.cross(c, a)

        cc_vec = ab + bc + ca
        cc_norm = np.linalg.norm(cc_vec)

        if cc_norm < 1e-12:
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            cc_vec = n / n_norm
            cc_norm = 1.0

        cc = cc_vec / cc_norm

        centroid = (a + b + c) / 3.0
        if np.dot(cc, centroid) < 0:
            cc = -cc

        dot_a = np.clip(np.dot(cc, a), -1.0, 1.0)
        circ_r = np.arccos(dot_a)

        if circ_r > max_circ_r:
            max_circ_r = circ_r
            worst_simplex = simplex
            worst_cc = cc

    return math.degrees(max_circ_r), worst_simplex, worst_cc


def _fast_lloyd(pts, n_iters=30, deadline=None):
    """Fast Lloyd iteration for initial optimization."""
    pts = pts.copy()
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    n_samples = 8000
    i_s = np.arange(n_samples) + 0.5
    z_s = 1.0 - 2.0 * i_s / n_samples
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
    r_s = np.sqrt(np.maximum(0, 1.0 - z_s * z_s))
    samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])

    best_pts = pts.copy()
    best_cr, _, _ = _covering_radius_deg(pts)

    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break

        dots = samples @ pts.T
        nearest = np.argmax(dots, axis=1)

        new_pts = np.zeros_like(pts)
        for k in range(N):
            mask = (nearest == k)
            if mask.sum() > 0:
                new_pts[k] = samples[mask].mean(axis=0)
            else:
                new_pts[k] = pts[k]

        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_pts = new_pts / norms

        # Blend to avoid large jumps
        alpha = 0.3
        candidate = (1 - alpha) * pts + alpha * new_pts
        candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)

        cr, _, _ = _covering_radius_deg(candidate)
        if cr < best_cr:
            best_cr = cr
            best_pts = candidate.copy()
            pts = candidate

    return best_pts, best_cr


def _gradient_refine(pts, deadline, max_iters=200):
    """Gradient-based refinement targeting worst facets."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr, _, _ = _covering_radius_deg(pts)

    step = 0.03
    for it in range(max_iters):
        if time.time() > deadline:
            break

        cr, worst_simplex, worst_cc = _covering_radius_deg(pts)
        if worst_simplex is None or worst_cc is None:
            break

        improved = False
        for trial_step in [step, step * 0.5, step * 0.25, step * 0.125]:
            candidate = pts.copy()
            for i in worst_simplex:
                direction = worst_cc - pts[i]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    candidate[i] = pts[i] + trial_step * direction
                    candidate[i] = candidate[i] / np.linalg.norm(candidate[i])

            cr_new, _, _ = _covering_radius_deg(candidate)
            if cr_new < best_cr - 1e-12:
                best_cr = cr_new
                best_pts = candidate.copy()
                pts = candidate
                improved = True
                break

        if not improved:
            step *= 0.8
            if step < 1e-5:
                break

    return best_pts, best_cr


def _micro_polish(pts, deadline, max_iters=300):
    """Very fine polish with tiny steps."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr, _, _ = _covering_radius_deg(pts)

    step = 0.005
    for it in range(max_iters):
        if time.time() > deadline:
            break

        cr, worst_simplex, worst_cc = _covering_radius_deg(pts)
        if worst_simplex is None or worst_cc is None:
            break

        improved = False
        for trial_step in [step, step * 0.5, step * 0.25]:
            candidate = pts.copy()
            for i in worst_simplex:
                direction = worst_cc - pts[i]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    candidate[i] = pts[i] + trial_step * direction
                    candidate[i] = candidate[i] / np.linalg.norm(candidate[i])

            cr_new, _, _ = _covering_radius_deg(candidate)
            if cr_new < best_cr - 1e-13:
                best_cr = cr_new
                best_pts = candidate.copy()
                pts = candidate
                improved = True
                break

        if not improved:
            step *= 0.7
            if step < 1e-7:
                break

    return best_pts, best_cr


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_init(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    best_pts = None
    best_cr = float('inf')

    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_spiral(N))
    for seed in [1, 2, 3, 7, 13, 42, 99, 123, 456, 789]:
        inits.append(_random_init(N, seed))

    # Antipodal pairs
    for seed in [10, 20, 30]:
        rng = np.random.RandomState(seed)
        half = N // 2
        pts_half = rng.randn(half, 3)
        pts_half = pts_half / np.linalg.norm(pts_half, axis=1, keepdims=True)
        pts = np.vstack([pts_half, -pts_half])
        if N % 2 == 1:
            extra = rng.randn(1, 3)
            extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
            pts = np.vstack([pts, extra])
        inits.append(pts)

    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 15:
            break

        # Quick validity check
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(init_pts[i] - init_pts[j])
                min_dist = min(min_dist, d)
        if min_dist < 0.01:
            continue

        # Check if origin is inside convex hull
        try:
            hull = ConvexHull(init_pts)
        except Exception:
            continue

        # Phase 1: Fast Lloyd
        try:
            lloyd_pts, lloyd_cr = _fast_lloyd(init_pts, n_iters=30, deadline=deadline - 10)
        except Exception:
            continue

        if lloyd_cr < best_cr:
            best_cr = lloyd_cr
            best_pts = lloyd_pts.copy()

        # Phase 2: Gradient refinement
        if time.time() < deadline - 8:
            try:
                grad_pts, grad_cr = _gradient_refine(lloyd_pts, deadline - 5, max_iters=150)
                if grad_cr < best_cr:
                    best_cr = grad_cr
                    best_pts = grad_pts.copy()
            except Exception:
                pass

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_cr, _, _ = _covering_radius_deg(best_pts)

    # Phase 3: Micro polish
    if time.time() < deadline - 3:
        try:
            polished_pts, polished_cr = _micro_polish(best_pts, deadline - 1, max_iters=300)
            if polished_cr < best_cr:
                best_cr = polished_cr
                best_pts = polished_pts
        except Exception:
            pass

    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)

    return best_pts
