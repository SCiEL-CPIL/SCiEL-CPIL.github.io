# EVOLVE-BLOCK-START
import time
import random
import sys

import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 16


def _radial_warp(points, alpha, beta):
    """Radial distortion: r' = r * (1 + alpha*r + beta*r^2)."""
    r = np.sqrt(np.sum(points ** 2, axis=1, keepdims=True))
    factor = 1.0 + alpha * r + beta * r ** 2
    return points * factor


def _projective_warp(points, a, b, c, d, e, f):
    """2D projective (homography) warp of a point set."""
    denom = 1.0 + c * points[:, 0] + f * points[:, 1]
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    x_new = (a * points[:, 0] + b * points[:, 1] + c) / denom
    y_new = (d * points[:, 0] + e * points[:, 1] + f) / denom
    return np.column_stack([x_new, y_new])


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Returns 16 points in 2D (shape (16,2)) maximizing R = (d_min/d_max)^2.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin under the 630s kill

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    def compute_score(points):
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = d.min()
        d_max = d.max()
        if d_max == 0.0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize_points(points):
        d = pdist(points)
        d_max = d.max()
        if d_max > 1e-12:
            return points / d_max
        return points

    def soft_min_value(points, beta):
        """Smooth approximation of the minimum pairwise distance (d_max normalized to 1)."""
        d = pdist(points)
        d_max = d.max()
        if d_max < 1e-12:
            return 0.0
        dn = d / d_max
        vals = np.exp(-beta * dn)
        s = vals.sum()
        if s == 0.0:
            return 0.0
        return -1.0 / beta * np.log(s)

    def soft_min_grad(points, beta):
        """
        Returns (value, grad) where value = soft_min(d)/d_max and grad is d(value)/d(x).
        """
        d = pdist(points)
        d_max = d.max()
        if d_max < 1e-12:
            return 0.0, np.zeros_like(points)
        dn = d / d_max
        e = np.exp(-beta * dn)
        s = e.sum()
        if s == 0.0:
            return 0.0, np.zeros_like(points)
        p = e / s
        val = -1.0 / beta * np.log(s)
        ii, jj = np.triu_indices(N, k=1)
        grad = np.zeros(2 * N)
        diff = points[ii] - points[jj]
        inv = p / (dn * d_max)
        grad_i = -(diff * inv[:, None])
        grad_j = +(diff * inv[:, None])
        np.add.at(grad, ii * 2, grad_i[:, 0])
        np.add.at(grad, ii * 2 + 1, grad_i[:, 1])
        np.add.at(grad, jj * 2, grad_j[:, 0])
        np.add.at(grad, jj * 2 + 1, grad_j[:, 1])
        return val, grad.reshape(N, 2)

    def anneal(points, time_budget, betas, method="L-BFGS-B", maxfun=1500):
        """Anneal the soft-min temperature, improving the point set."""
        points = normalize_points(points.copy())
        best = points.copy()
        best_score = compute_score(best)
        t0 = time.time()
        for beta in betas:
            if time.time() - t0 > time_budget:
                break
            if method == "L-BFGS-B":
                def obj(x):
                    return soft_min_value(x.reshape(N, 2), beta)

                def objg(x):
                    v, g = soft_min_grad(x.reshape(N, 2), beta)
                    return v, g.flatten()

                try:
                    res = minimize(objg, points.flatten(), method="L-BFGS-B",
                                   jac=True,
                                   options={"maxiter": 400, "ftol": 1e-15,
                                            "gtol": 1e-12, "maxfun": maxfun})
                    cand = normalize_points(res.x.reshape(N, 2))
                except Exception:
                    continue
            else:
                def obj(x):
                    return -soft_min_value(x.reshape(N, 2), beta)

                try:
                    res = minimize(obj, points.flatten(), method="Nelder-Mead",
                                   options={"maxiter": 2000, "xatol": 1e-9,
                                            "fatol": 1e-10})
                    cand = normalize_points(res.x.reshape(N, 2))
                except Exception:
                    continue
            sc = compute_score(cand)
            if sc > best_score:
                best_score = sc
                best = cand.copy()
            points = best.copy()
            if time.time() - t0 > time_budget:
                break
        return best, best_score

    def nm_polish(points, time_budget, maxiter=8000, xatol=1e-11, fatol=1e-11):
        """Final Nelder-Mead on the exact (non-smooth) objective."""
        def obj(x):
            p = normalize_points(x.reshape(N, 2))
            d = pdist(p)
            dmin, dmax = d.min(), d.max()
            if dmax == 0:
                return 0.0
            return -(dmin / dmax) ** 2

        x0 = normalize_points(points.copy()).flatten()
        try:
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol})
            cand = normalize_points(res.x.reshape(N, 2))
            sc = compute_score(cand)
            if sc > compute_score(points):
                return cand, sc
        except Exception:
            pass
        return points, compute_score(points)

    # ------------------------------------------------------------------
    # SDP-inspired: Gram matrix optimization
    # ------------------------------------------------------------------
    def points_to_gram(points):
        """Compute Gram matrix G = X X^T from centered points."""
        c = points - points.mean(axis=0)
        return c @ c.T

    def gram_to_points_2d(G):
        """Extract best 2D embedding from Gram matrix via SVD."""
        # Symmetrize
        G = 0.5 * (G + G.T)
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        # Take top 2 eigenvalues
        idx = np.argsort(eigvals)[::-1][:2]
        vals = eigvals[idx]
        vecs = eigvecs[:, idx]
        if vals[0] < 1e-15:
            return np.zeros((N, 2))
        # X = V * sqrt(lambda)
        X = vecs * np.sqrt(np.maximum(vals, 0))
        return X

    def gram_soft_min_obj(beta):
        """
        Objective on Gram matrix entries: maximize soft_min of distances derived from G.
        Distance squared: d_ij^2 = G_ii + G_jj - 2*G_ij
        We work with G having unit diagonal (normalized), so d_ij^2 = 2 - 2*G_ij = 2(1-G_ij).
        Actually let's keep general: d_ij = sqrt(G_ii + G_jj - 2 G_ij)
        For simplicity, assume unit diagonal: d_ij = sqrt(2 - 2*G_ij)
        soft_min of d_ij / d_max where d_max = max d_ij.
        """
        def obj_and_grad(g_flat):
            G = g_flat.reshape(N, N)
            G = 0.5 * (G + G.T)
            # Extract off-diagonal
            ii, jj = np.triu_indices(N, k=1)
            # d^2 = G_ii + G_jj - 2*G_ij
            d2 = G[ii, ii] + G[jj, jj] - 2.0 * G[ii, jj]
            d2 = np.maximum(d2, 1e-15)
            d = np.sqrt(d2)
            d_max = d.max()
            if d_max < 1e-12:
                return 0.0, np.zeros_like(g_flat)
            dn = d / d_max
            e = np.exp(-beta * dn)
            s = e.sum()
            if s == 0.0:
                return 0.0, np.zeros_like(g_flat)
            p = e / s
            val = -1.0 / beta * np.log(s)
            # Gradient: d(val)/d(G_ij) for each pair
            # d(val)/d(dn_k) = -p_k
            # d(dn_k)/d(d_k) = 1/d_max
            # d(d_k)/d(d2_k) = 1/(2*d_k)
            # d(d2_k)/d(G_ii) = 1, d(d2_k)/d(G_jj) = 1, d(d2_k)/d(G_ij) = -2
            grad = np.zeros((N, N))
            # d(val)/d(G_ii) += sum_k p_k * (1/d_max) * (1/(2*d_k)) * 1
            # = sum_k p_k / (2 * d_k * d_max)
            # d(val)/d(G_ij) += sum_k p_k * (1/d_max) * (1/(2*d_k)) * (-2)
            # = -sum_k p_k / (d_k * d_max)
            coeff = p / (d * d_max)  # p_k / (d_k * d_max)
            np.add.at(grad, (ii, ii), coeff)   # +coeff for G_ii
            np.add.at(grad, (jj, jj), coeff)   # +coeff for G_jj
            np.add.at(grad, (ii, jj), -2.0 * coeff)  # -2*coeff for G_ij
            np.add.at(grad, (jj, ii), -2.0 * coeff)  # symmetric
            return val, grad.flatten()
        return obj_and_grad

    def sdp_anneal(points, time_budget, betas, maxfun=2000):
        """
        SDP-inspired annealing: optimize in Gram matrix space,
        extract 2D points via SVD, then refine in coordinate space.
        """
        best = normalize_points(points.copy())
        best_score = compute_score(best)
        t0 = time.time()
        
        # Start from current points' Gram
        G0 = points_to_gram(best)
        g0 = G0.flatten()
        
        for beta in betas:
            if time.time() - t0 > time_budget:
                break
            try:
                objg = gram_soft_min_obj(beta)
                # L-BFGS-B on Gram matrix (symmetric, so we optimize all entries and symmetrize)
                res = minimize(objg, g0, method="L-BFGS-B", jac=True,
                               options={"maxiter": 300, "ftol": 1e-14,
                                        "gtol": 1e-10, "maxfun": maxfun})
                G_opt = res.x.reshape(N, N)
                G_opt = 0.5 * (G_opt + G_opt.T)
                # Extract 2D points
                cand = gram_to_points_2d(G_opt)
                cand = normalize_points(cand)
                sc = compute_score(cand)
                if sc > best_score:
                    best_score = sc
                    best = cand.copy()
                    g0 = points_to_gram(best).flatten()
            except Exception:
                continue
            if time.time() - t0 > time_budget:
                break
        return best, best_score

    # ------------------------------------------------------------------
    # 1. Build a diverse pool of starting configurations
    # ------------------------------------------------------------------
    candidates = []

    # 4x4 square grid (centered)
    gx = np.array([[-1.5, -0.5, 0.5, 1.5]] * 4).flatten()
    gy = np.array([-1.5, -0.5, 0.5, 1.5] * 4).flatten()
    grid = np.column_stack([gx, gy])
    candidates.append(grid.copy())

    # Radially warped 4x4 grids
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for beta_w in [0.0, -0.05, -0.1, -0.15, 0.05, 0.1]:
            w = _radial_warp(grid, alpha, beta_w)
            w = w - w.mean(axis=0)
            candidates.append(w)

    # Projective warps of the grid
    proj_list = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0, 0.0, 0.05, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.05),
        (0.05, 0.0, 0.0, 0.0, 0.05, 0.05),
        (0.1, 0.0, 0.0, 0.0, 0.1, 0.0),
        (0.0, 0.1, 0.0, 0.1, 0.0, 0.0),
        (0.05, 0.05, 0.0, 0.05, 0.05, 0.0),
        (-0.05, 0.0, 0.0, 0.0, -0.05, 0.0),
        (0.0, 0.0, 0.02, 0.0, 0.0, 0.02),
        (0.0, 0.0, -0.02, 0.0, 0.0, -0.02),
        (0.08, 0.0, 0.01, 0.0, 0.08, 0.01),
        (0.0, 0.08, 0.0, 0.08, 0.0, 0.01),
    ]
    for a, b, c, d, e, f in proj_list:
        w = _projective_warp(grid, a, b, c, d, e, f)
        w = w - w.mean(axis=0)
        candidates.append(w)

    # Triangular lattice cluster
    s = 1.0
    h = s * np.sqrt(3) / 2
    lat = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            lat.append([i * s + (j % 2) * s / 2, j * h])
    lat = np.array(lat)
    cen = lat.mean(axis=0)
    lat = lat[np.argsort(np.sum((lat - cen) ** 2, axis=1))[:N]]
    candidates.append(lat)

    # Regular 16-gon
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    candidates.append(np.column_stack([np.cos(ang), np.sin(ang)]))

    # Concentric circles
    for n_in, n_out, r_in in [(4, 12, 0.4), (6, 10, 0.45), (8, 8, 0.5)]:
        ia = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
        oa = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + np.pi / n_out
        inner = np.column_stack([r_in * np.cos(ia), r_in * np.sin(ia)])
        outer = np.column_stack([np.cos(oa), np.sin(oa)])
        candidates.append(np.vstack([inner, outer]))

    # Hexagonal close-packed cluster
    s_hex = 1.0
    hex_pts = []
    hex_pts.append([0, 0])
    for k in range(6):
        ang_k = k * np.pi / 3
        hex_pts.append([s_hex * np.cos(ang_k), s_hex * np.sin(ang_k)])
    for k in range(12):
        ang_k = k * np.pi / 6
        r2 = 2 * s_hex if k % 2 == 0 else s_hex * np.sqrt(3)
        hex_pts.append([r2 * np.cos(ang_k), r2 * np.sin(ang_k)])
    hex_pts = np.array(hex_pts[:N])
    hex_pts = hex_pts - hex_pts.mean(axis=0)
    candidates.append(hex_pts)

    # Two concentric hexagons
    for scale in [0.6, 0.7, 0.8]:
        inner_hex = []
        outer_hex = []
        for k in range(6):
            ang_k = k * np.pi / 3
            inner_hex.append([scale * np.cos(ang_k), scale * np.sin(ang_k)])
        for k in range(10):
            ang_k = k * 2 * np.pi / 10 + np.pi / 10
            outer_hex.append([np.cos(ang_k), np.sin(ang_k)])
        combined = np.vstack([np.array(inner_hex), np.array(outer_hex)])
        combined = combined - combined.mean(axis=0)
        candidates.append(combined)

    # Fibonacci spiral
    ga = np.pi * (3 - np.sqrt(5))
    sp = []
    for i in range(16):
        r = np.sqrt(i + 1) / 4
        th = i * ga
        sp.append([r * np.cos(th), r * np.sin(th)])
    candidates.append(np.array(sp))

    # Random seeds
    for seed in range(14):
        np.random.seed(seed * 131 + 17)
        rp = np.random.randn(N, 2)
        rp = normalize_points(rp)
        candidates.append(rp)

    # Jittered grid
    for jseed in range(6):
        np.random.seed(5000 + jseed)
        jg = grid + np.random.randn(N, 2) * 0.08
        candidates.append(jg)

    # ------------------------------------------------------------------
    # 2. Rank candidates by initial score, optimize the best first
    # ------------------------------------------------------------------
    scored = sorted(((compute_score(c), c) for c in candidates),
                    key=lambda t: t[0], reverse=True)

    best_points = scored[0][1].copy()
    best_score = scored[0][0]

    # Warm-start: quick anneal on top few
    for sc, cand in scored[:4]:
        if remaining() < 40:
            break
        p, s = anneal(cand, time_budget=8.0,
                      betas=[10, 30, 100, 300, 800, 2000], maxfun=1200)
        if s > best_score:
            best_score = s
            best_points = p.copy()

    # Full optimization pass: alternate coordinate-space and Gram-space optimization
    for rank, (sc, cand) in enumerate(scored):
        if remaining() < 30:
            break
        if rank < 4:
            budget = 18.0
        elif rank < 12:
            budget = 10.0
        elif rank < 24:
            budget = 6.0
        else:
            budget = 3.5
        if sc + 0.5 < best_score and rank >= 12:
            budget = min(budget, 2.0)
        
        # Coordinate-space anneal
        p, s = anneal(cand, time_budget=budget * 0.6,
                      betas=[10, 30, 100, 300, 800, 2000], maxfun=1500)
        if s > best_score:
            best_score = s
            best_points = p.copy()
        
        # Gram-space (SDP-inspired) anneal on the best found so far
        if remaining() > 20:
            p2, s2 = sdp_anneal(p, time_budget=budget * 0.4,
                                betas=[20, 50, 100, 300, 800], maxfun=1500)
            if s2 > best_score:
                best_score = s2
                best_points = p2.copy()

    # ------------------------------------------------------------------
    # 3. High-precision refinement: alternate coord and Gram spaces
    # ------------------------------------------------------------------
    if remaining() > 80:
        p, s = anneal(best_points, time_budget=min(30.0, remaining() - 50),
                      betas=[500, 1500, 4000, 10000, 25000], maxfun=3000)
        if s > best_score:
            best_score = s
            best_points = p.copy()
        
        if remaining() > 50:
            p, s = sdp_anneal(best_points, time_budget=min(30.0, remaining() - 30),
                              betas=[100, 300, 800, 2000, 5000], maxfun=2500)
            if s > best_score:
                best_score = s
                best_points = p.copy()

    # ------------------------------------------------------------------
    # 4. Exact Nelder-Mead polish
    # ------------------------------------------------------------------
    if remaining() > 40:
        p, s = nm_polish(best_points, time_budget=min(30.0, remaining() - 25))
        if s > best_score:
            best_score = s
            best_points = p.copy()

    # ------------------------------------------------------------------
    # 5. Final cycle: Gram-space + coord-space + polish
    # ------------------------------------------------------------------
    if remaining() > 50:
        p, s = sdp_anneal(best_points, time_budget=min(25.0, remaining() - 35),
                          betas=[2000, 5000, 15000], maxfun=2000)
        if s > best_score:
            best_score = s
            best_points = p.copy()
        
        if remaining() > 30:
            p, s = anneal(best_points, time_budget=min(15.0, remaining() - 20),
                          betas=[5000, 15000, 40000], maxfun=2000)
            if s > best_score:
                best_score = s
                best_points = p.copy()
        
        if remaining() > 15:
            p, s = nm_polish(best_points, time_budget=min(12.0, remaining() - 10))
            if s > best_score:
                best_score = s
                best_points = p.copy()

    if best_points is None:
        best_points = scored[0][1]

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
