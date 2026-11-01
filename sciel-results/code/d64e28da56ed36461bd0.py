"""Sphere covering with 50 points: minimize max facet circumradius."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def project(pts):
        n = np.linalg.norm(pts, axis=1, keepdims=True)
        n = np.where(n < 1e-12, 1.0, n)
        return pts / n

    def facet_data(pts):
        """Return arrays: radii (M,), centers (M,3), simplices (M,3)."""
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None
        simp = hull.simplices
        M = len(simp)
        radii = np.empty(M)
        centers = np.empty((M, 3))
        for k, s in enumerate(simp):
            v1, v2, v3 = pts[s[0]], pts[s[1]], pts[s[2]]
            n = np.cross(v2 - v1, v3 - v1)
            nn = np.linalg.norm(n)
            if nn < 1e-14:
                radii[k] = np.pi
                centers[k] = v1
                continue
            c = n / nn
            if np.dot(c, v1) < 0:
                c = -c
            d = np.linalg.norm(c - v1)
            radii[k] = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            centers[k] = c
        return radii, centers, simp

    def covering_radius(pts):
        fd = facet_data(pts)
        if fd is None:
            return np.pi, np.zeros(3)
        radii, centers, simp = fd
        j = int(np.argmax(radii))
        return radii[j], centers[j]

    def soft_max_objective_and_grad(params, tau):
        pts = project(params.reshape(N, 3))
        fd = facet_data(pts)
        if fd is None:
            return 1e10, np.zeros_like(params)
        radii, centers, simp = fd
        M = len(radii)
        scaled = radii / tau
        mx = np.max(scaled)
        exps = np.exp(scaled - mx)
        Z = np.sum(exps)
        obj = tau * (mx + np.log(Z))
        # weights for soft-max
        w = exps / Z  # sum to 1, these are softmax weights
        # gradient of obj w.r.t. params: obj = tau*logsumexp(radii/tau)
        # d obj/d radii_j = w_j
        # d radii_j / d v  (for the 3 vertices of facet j)
        # Build gradient w.r.t. pts (N,3)
        gpts = np.zeros_like(pts)
        # For each facet, compute d r_j / d v_i for i in facet
        # Use analytic-ish: recompute radius with perturbed vertices via central diff on the 3 verts
        h = 1e-6
        for j in range(M):
            if w[j] < 1e-14:
                continue
            s = simp[j]
            a, b, c = pts[s[0]], pts[s[1]], pts[s[2]]
            # helper: radius of triangle (p,q,r)
            def tri_radius(p, q, r):
                n = np.cross(q - p, r - p)
                nn = np.linalg.norm(n)
                if nn < 1e-14:
                    return np.pi
                cc = n / nn
                if np.dot(cc, p) < 0:
                    cc = -cc
                d = np.linalg.norm(cc - p)
                return 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            r0 = radii[j]
            verts = [a, b, c]
            idxs = [s[0], s[1], s[2]]
            for vi, v in enumerate(verts):
                # central difference in 3D, then project to tangent
                for d3 in range(3):
                    e = np.zeros(3)
                    e[d3] = h
                    vp = project(v + e)
                    vm = project(v - e)
                    other = [verts[k] for k in range(3) if k != vi]
                    rp = tri_radius(vp, other[0], other[1])
                    rm = tri_radius(vm, other[0], other[1])
                    # derivative w.r.t. the d3 coordinate of v (before projection)
                    # but we move in tangent; approximate d r / d (tangent step)
                    # Use directional: the actual displacement is (vp - v)
                    disp = vp - v
                    # gradient contribution in 3D approx:
                    dr = (rp - rm) / (2.0 * h)
                    # disp is ~ e (unit in d3) projected; map dr to direction
                    # We want gpts such that gpts . disp = dr
                    # Simplest: gpts[idx] += dr * e / (e . disp)  but disp ~ e
                    # Just accumulate dr along e direction in 3D
                    gpts[idxs[vi], d3] += dr
            # project facet gradient to tangent for each vertex
            for vi in range(3):
                idx = idxs[vi]
                v = pts[idx]
                g = gpts[idx]
                gpts[idx] = g - np.dot(g, v) * v
        gparams = gpts.reshape(-1)
        return obj, gparams

    def lbfgs_from(pts, tau, maxiter, maxfun):
        x0 = pts.flatten()
        def f(x):
            return soft_max_objective_and_grad(x, tau)[0]
        def fg(x):
            o, g = soft_max_objective_and_grad(x, tau)
            return o, g
        try:
            res = minimize(fg, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'maxfun': maxfun,
                                    'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            return project(res.x.reshape(N, 3))
        except Exception:
            return pts

    def local_polish(pts, iters=400, step0=0.02):
        """Steepest descent on the true max facet, tangent space, adaptive step."""
        pts = pts.copy()
        step = step0
        prev_r = covering_radius(pts)[0]
        stall = 0
        for it in range(iters):
            if time.time() > deadline - 5:
                break
            r, wc = covering_radius(pts)
            if r >= prev_r - 1e-15:
                step *= 0.5
                stall += 1
                if step < 1e-9 or stall > 60:
                    break
            else:
                prev_r = r
                stall = 0
            fd = facet_data(pts)
            if fd is None:
                break
            radii, centers, simp = fd
            # find facets whose center ~ worst center
            targets = set()
            for k in range(len(radii)):
                if np.dot(centers[k], wc) > 0.9999:
                    for idx in simp[k]:
                        targets.add(idx)
            if not targets:
                break
            for idx in targets:
                v = pts[idx]
                d = wc - v
                d = d - np.dot(d, v) * v
                dn = np.linalg.norm(d)
                if dn > 1e-12:
                    d /= dn
                    pts[idx] = project(v + step * d)
        return pts

    # ---- main search ----
    def try_config(pts, label):
        nonlocal best, best_r
        r, _ = covering_radius(pts)
        if r < best_r:
            best_r = r
            best = pts.copy()
        return r

    # Phase 1: Fibonacci start
    fib = fibonacci_sphere(N)
    best = fib.copy()
    best_r = covering_radius(best)[0]

    # Phase 2: gradient-based L-BFGS with annealed temperature
    cur = fib.copy()
    temps = [0.15, 0.10, 0.06, 0.04, 0.025, 0.015, 0.010, 0.006, 0.004, 0.0025, 0.0015]
    for t in temps:
        if time.time() > deadline - 40:
            break
        cur = lbfgs_from(cur, t, maxiter=120, maxfun=1500)
        r = try_config(cur, 'lbfgs')

    # Phase 3: local polish
    if time.time() < deadline - 30:
        cur = local_polish(cur, iters=600, step0=0.02)
        try_config(cur, 'polish')

    # Phase 4: random restarts
    rng = np.random.default_rng(12345)
    n_restarts = 0
    while time.time() < deadline - 35:
        n_restarts += 1
        # perturb best
        scale = 0.05 / (1 + 0.1 * n_restarts)
        pert = best.copy()
        for i in range(N):
            rv = rng.standard_normal(3)
            rv /= np.linalg.norm(rv)
            pert[i] = project(pert[i] + scale * rv)
        # quick lbfgs at moderate temp then polish
        if time.time() < deadline - 35:
            pert = lbfgs_from(pert, 0.03, maxiter=80, maxfun=900)
        if time.time() < deadline - 35:
            pert = local_polish(pert, iters=300, step0=0.01)
        try_config(pert, 'restart')

    # Phase 5: final high-precision polish
    if time.time() < deadline - 3:
        fin = local_polish(best, iters=3000, step0=0.005)
        try_config(fin, 'final')

    # validation
    best = project(best)
    if not np.all(np.isfinite(best)) or np.any(np.linalg.norm(best, axis=1) < 1e-10):
        best = fibonacci_sphere(N)
    return best
