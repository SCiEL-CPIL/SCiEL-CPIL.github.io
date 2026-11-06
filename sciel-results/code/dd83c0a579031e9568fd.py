"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s limit

    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])
    theta = 2 * np.pi * np.arange(n) / n

    def log_score(pts):
        """log of score Delta/n^n; Delta = product over ordered pairs of distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        log_delta = 2.0 * np.sum(np.log(d_ij))
        return log_delta - n * np.log(n)

    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)

    def neg_log_score_and_grad(x):
        """
        Objective: -log_score(pts) where pts = x.reshape(n,2)
        We maximize log_score = 2*sum(log(d_ij)) - n*log(n)
        So we minimize -log_score = -2*sum(log(d_ij)) + n*log(n)
        
        Gradient of -2*sum(log(d_ij)) w.r.t. pts:
        d/d(pts[i]) [-2*log(d_ij)] = -2 * (1/d_ij) * d(d_ij)/d(pts[i])
        d(d_ij)/d(pts[i]) = (pts[i] - pts[j]) / d_ij
        So d/d(pts[i]) [-2*log(d_ij)] = -2 * (pts[i]-pts[j]) / d_ij^2
        """
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        
        val = -2.0 * np.sum(np.log(d_ij)) + n * np.log(n)
        
        grad = np.zeros_like(pts)
        for idx in range(num_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = d_ij[idx]
            if d < 1e-15:
                continue
            diff = pts[i] - pts[j]
            coef = -2.0 / (d * d)
            grad[i] += coef * diff
            grad[j] -= coef * diff
        
        return val, grad.flatten()

    def diameter_constraint(pts):
        """Return max(0, diameter - 2) as a constraint violation."""
        return max(0.0, diameter(pts) - 2.0)

    def soft_diameter_penalty(pts, W, T):
        """
        Smooth approximation of W * max(0, diam^2 - 4) using log-sum-exp.
        soft_max = (1/T) * logsumexp(T * (d_ij^2 - 4))
        """
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        sq = d_ij * d_ij
        a = T * (sq - 4.0)
        m = np.max(a)
        if not np.isfinite(m):
            return W * max(0.0, (np.max(sq) - 4.0))
        e = np.exp(a - m)
        s = np.sum(e)
        smax = (m + np.log(s)) / T
        return W * max(smax, 0.0)

    def obj_constrained(x, W, T):
        """Combined objective: -log_score + penalty for diameter > 2."""
        pts = x.reshape(n, 2)
        val, grad = neg_log_score_and_grad(x)
        
        # Add soft diameter penalty
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        sq = d_ij * d_ij
        a = T * (sq - 4.0)
        m = np.max(a)
        if np.isfinite(m):
            e = np.exp(a - m)
            s = np.sum(e)
            smax = (m + np.log(s)) / T
            w = e / s
            if smax > 0.0:
                val += W * smax
                # Gradient of penalty
                for idx in range(num_pairs):
                    i, j = iu[0][idx], iu[1][idx]
                    d = d_ij[idx]
                    if d < 1e-15:
                        continue
                    diff = pts[i] - pts[j]
                    coef = W * 2.0 * d * w[idx]
                    grad[i * 2] += coef * diff[0]
                    grad[i * 2 + 1] += coef * diff[1]
                    grad[j * 2] -= coef * diff[0]
                    grad[j * 2 + 1] -= coef * diff[1]
        
        return val, grad

    def generate_initial_points():
        inits = []
        # Regular octagon
        octagon = np.column_stack([np.cos(theta), np.sin(theta)])
        inits.append(octagon.copy())
        
        # Slightly perturbed octagons
        rng = np.random.RandomState(42)
        for scale in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            inits.append((octagon + rng.randn(n, 2) * scale).copy())
        
        # Two-cluster configurations
        for r1 in [0.3, 0.4, 0.5, 0.6]:
            for r2 in [0.3, 0.4, 0.5, 0.6]:
                for off in [0, np.pi/4, np.pi/3, np.pi/2]:
                    c1 = np.column_stack([np.cos(theta[:4] + off), np.sin(theta[:4] + off)]) * r1
                    c2 = np.column_stack([np.cos(theta[4:] + off), np.sin(theta[4:] + off)]) * r2
                    inits.append(np.vstack([c1, c2]).copy())
        
        # Square + midpoints
        inits.append(np.array([
            [1, 1], [1, -1], [-1, -1], [-1, 1],
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=float).copy())
        
        # Random points on various radii
        for _ in range(8):
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = rng.uniform(0.4, 1.0, n)
            inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).copy())
        
        # Random points in disk
        for _ in range(5):
            pts = rng.randn(n, 2)
            norms = np.linalg.norm(pts, axis=1)
            pts = pts / norms[:, None] * rng.uniform(0.3, 1.0, n)
            inits.append(pts.copy())
        
        return inits

    def scaled_log_score(pts):
        d = diameter(pts)
        if d <= 1e-12:
            return -np.inf
        return log_score(pts * (2.0 / d))

    def run_lbfgs(x0, W, T, maxiter, maxfun):
        if time.time() > deadline - 5:
            return None
        try:
            res = minimize(
                lambda v: obj_constrained(v, W, T)[0],
                x0,
                jac=lambda v: obj_constrained(v, W, T)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': maxfun},
            )
            return res.x.reshape(n, 2)
        except Exception:
            return None

    def run_nelder_mead(x0, maxiter):
        if time.time() > deadline - 5:
            return None
        try:
            res = minimize(
                lambda v: -log_score(v.reshape(n, 2)) + 100000.0 * max(0.0, diameter(v.reshape(n, 2)) - 2.0)**2,
                x0,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            return res.x.reshape(n, 2)
        except Exception:
            return None

    # --- Stage 1: Evaluate diverse starts ---
    best_score = -np.inf
    best_points = None
    for init_pts in generate_initial_points():
        if time.time() > deadline - 20:
            break
        ls = scaled_log_score(init_pts)
        if ls > best_score:
            best_score = ls
            d = diameter(init_pts)
            best_points = (init_pts * (2.0 / d)).copy()

    # --- Stage 2: Constrained optimization from best starts ---
    scored = []
    for init_pts in generate_initial_points():
        ls = scaled_log_score(init_pts)
        if ls > -np.inf:
            d = diameter(init_pts)
            scored.append((ls, init_pts * (2.0 / d)))
    scored.sort(key=lambda t: -t[0])

    penalty_weights = [10.0, 50.0, 100.0, 200.0, 500.0]
    temps = [50.0, 100.0, 200.0, 400.0]
    
    for _, sp in scored[:8]:
        for pw in penalty_weights[:4]:
            for smt in temps[:3]:
                if time.time() > deadline - 15:
                    break
                opt = run_lbfgs(sp.flatten(), pw, smt, 500, 6000)
                if opt is None:
                    continue
                d = diameter(opt)
                if d <= 1e-10:
                    continue
                opt = opt * (2.0 / d)
                ls = log_score(opt)
                if ls > best_score:
                    best_score = ls
                    best_points = opt.copy()

    # --- Stage 3: Local refinement with perturbations ---
    if best_points is not None and time.time() < deadline - 20:
        rng2 = np.random.RandomState(123)
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            for scale in [0.01, 0.03, 0.05, 0.1]:
                if time.time() > deadline - 8:
                    break
                pert = best_points + rng2.randn(n, 2) * scale
                for pw in [100.0, 200.0, 500.0]:
                    for smt in [100.0, 200.0, 400.0]:
                        if time.time() > deadline - 5:
                            break
                        opt = run_lbfgs(pert.flatten(), pw, smt, 400, 5000)
                        if opt is None:
                            continue
                        d = diameter(opt)
                        if d <= 1e-10:
                            continue
                        opt = opt * (2.0 / d)
                        ls = log_score(opt)
                        if ls > best_score:
                            best_score = ls
                            best_points = opt.copy()

    # --- Stage 4: Nelder-Mead polish ---
    if best_points is not None and time.time() < deadline - 10:
        opt = run_nelder_mead(best_points.flatten(), 5000)
        if opt is not None:
            d = diameter(opt)
            if d > 1e-10:
                opt = opt * (2.0 / d)
                ls = log_score(opt)
                if ls > best_score:
                    best_score = ls
                    best_points = opt.copy()

    # --- Stage 5: Final high-precision polish ---
    if best_points is not None and time.time() < deadline - 5:
        for pw in [500.0, 1000.0, 2000.0]:
            for smt in [200.0, 400.0, 800.0]:
                if time.time() > deadline - 3:
                    break
                opt = run_lbfgs(best_points.flatten(), pw, smt, 1000, 15000)
                if opt is None:
                    continue
                d = diameter(opt)
                if d <= 1e-10:
                    continue
                opt = opt * (2.0 / d)
                ls = log_score(opt)
                if ls > best_score:
                    best_score = ls
                    best_points = opt.copy()

    # --- Finalize ---
    if best_points is None:
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])

    d = diameter(best_points)
    if d > 1e-10:
        best_points = best_points * (2.0 / d)

    # Ensure distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                best_points[j] += np.array([1e-8, 1e-8])

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e5:
        best_points = best_points * (1e5 / max_coord)
        d = diameter(best_points)
        if d > 1e-10:
            best_points = best_points * (2.0 / d)

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
