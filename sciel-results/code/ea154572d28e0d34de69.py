import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start = time.time()
    DEADLINE = start + 625

    n = 14
    d = 3

    def score(pts):
        dd = pdist(pts)
        if dd.max() <= 0:
            return 0.0
        return (dd.min() / dd.max()) ** 2

    def neg_metric(x):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        return -score(pts)

    def make_nm(x0, maxiter=20000):
        r = minimize(neg_metric, x0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        return r.x

    def make_lbfgs(x0, maxiter=1000):
        r = minimize(neg_metric, x0, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-14})
        return r.x

    best_pts = None
    best_score = -1.0
    rng = np.random.RandomState(42)

    # ---- Strategy 1: Icosahedral symmetry parameterization (from inspiration) ----
    phi = (1 + np.sqrt(5)) / 2
    I12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    I12 = I12 / np.linalg.norm(I12, axis=1, keepdims=True)
    R0 = 1.0 / np.sqrt(1 + phi**2)

    def make_points_ico(p):
        # p = [r, z1, z2]
        r = p[0]
        pts = np.vstack([r * I12, [[0, 0, p[1]], [0, 0, p[2]]]])
        return pts

    def neg_metric_ico(p):
        pts = make_points_ico(p)
        dd = pdist(pts)
        if dd.max() <= 0:
            return 0.0
        return -(dd.min() / dd.max()) ** 2

    def make_nm_ico(p0, maxiter=20000):
        r = minimize(neg_metric_ico, p0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
        return r.x

    def make_lbfgs_ico(p0, maxiter=4000):
        r = minimize(neg_metric_ico, p0, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-14})
        return r.x

    ico_candidates = []
    # Symmetric z: z1 = -z2 = a
    for a in np.linspace(0.0, 0.6, 13):
        ico_candidates.append(np.array([R0, -a, a]))
    # Slightly asymmetric
    for a in np.linspace(0.1, 0.5, 8):
        for da in [0.0, 0.01, -0.01, 0.02, -0.02]:
            ico_candidates.append(np.array([R0, -a + da, a]))
    # Vary outer radius
    for r in [R0*0.95, R0*0.97, R0, R0*1.01, R0*1.03]:
        for a in np.linspace(0.0, 0.5, 7):
            ico_candidates.append(np.array([r, -a, a]))
    # Random
    rng_ico = np.random.RandomState(7)
    for _ in range(40):
        r = R0 * (0.9 + 0.2 * rng_ico.rand())
        a = rng_ico.rand() * 0.6
        da = (rng_ico.rand() - 0.5) * 0.05
        ico_candidates.append(np.array([r, -a + da, a]))

    for p in ico_candidates:
        if time.time() > DEADLINE - 60:
            break
        try:
            p1 = make_lbfgs_ico(p)
        except Exception:
            p1 = p
        try:
            p2 = make_nm_ico(p1)
        except Exception:
            p2 = p1
        for pp in [p1, p2]:
            pts = make_points_ico(pp)
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    # ---- Strategy 2: General multi-start with Coulomb + NM (from current) ----
    def coulomb_obj(x):
        pts = x.reshape((n, d))
        dd = pdist(pts)
        dd = np.where(dd < 1e-10, 1e-10, dd)
        return np.sum(1.0 / dd)

    def make_coulomb(x0):
        r = minimize(coulomb_obj, x0, method='L-BFGS-B',
                     options={'maxiter': 1000, 'ftol': 1e-10})
        return r.x

    num_starts = 30
    for i in range(num_starts):
        if time.time() > DEADLINE - 90:
            break

        x0 = rng.randn(n * d)
        try:
            x_coulomb = make_coulomb(x0)
        except Exception:
            x_coulomb = x0

        try:
            x_nm = make_nm(x_coulomb)
        except Exception:
            x_nm = x_coulomb

        pts = x_nm.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        s = score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # ---- Strategy 3: Structural seeds ----
    # Cube + Octahedron
    c = 1.0
    o = 2.0
    cube = np.array([[c, c, c], [c, c, -c], [c, -c, c], [c, -c, -c],
                     [-c, c, c], [-c, c, -c], [-c, -c, c], [-c, -c, -c]])
    octa = np.array([[o, 0, 0], [-o, 0, 0], [0, o, 0], [0, -o, 0], [0, 0, o], [0, 0, -o]])
    x_seed = np.vstack([cube, octa]).flatten()
    if time.time() < DEADLINE - 30:
        try:
            x_nm = make_nm(x_seed)
            pts = x_nm.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

    # Distorted icosahedron + poles
    poles = np.array([[0, 0, 1.5], [0, 0, -1.5]])
    x_seed2 = np.vstack([I12, poles]).flatten()
    if time.time() < DEADLINE - 30:
        try:
            x_nm = make_nm(x_seed2)
            pts = x_nm.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

    # ---- Strategy 4: Final polish with longer NM on best result ----
    if best_pts is not None and time.time() < DEADLINE - 10:
        x_best = best_pts.flatten()
        try:
            x_final = make_nm(x_best, maxiter=50000)
            pts_final = x_final.reshape((n, d))
            pts_final = pts_final - np.mean(pts_final, axis=0)
            s_final = score(pts_final)
            if s_final > best_score:
                best_score = s_final
                best_pts = pts_final.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = np.zeros((n, d))
        for i in range(n):
            best_pts[i] = rng.randn(d)
        best_pts = best_pts - np.mean(best_pts, axis=0)

    if not np.all(np.isfinite(best_pts)):
        best_pts = np.vstack([I12, [[0, 0, -0.25], [0, 0, 0.25]]])
        best_pts = best_pts - np.mean(best_pts, axis=0)

    return best_pts
