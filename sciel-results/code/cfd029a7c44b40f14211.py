"""Program for covering the sphere with 130 equal caps."""
import time
import numpy as np
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _facet_circumcenters(pts):
    """Return (circumcenters, per-facet radius in radians, vertex index arrays)."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    simplices = hull.simplices
    cc_list = []
    rad_list = []
    for simplex in simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-12:
            continue
        normal = normal / nrm
        # Plane equation: normal . x = d
        d = np.dot(normal, a)
        # Circumcenter of the triangle in the plane: solve
        # |cc|^2 = |cc - a|^2  =>  2 cc.a = |a|^2  (since |a|^2=1, |cc|^2=1)
        # 2 cc.a = 1 ; 2 cc.b = 1 ; 2 cc.c = 1 ; normal . cc = d
        # Build 4x3 system, take least-squares / use 3 independent equations.
        A = np.array([a, b, c, normal])
        rhs = np.array([0.5, 0.5, 0.5, d])
        try:
            cc, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except Exception:
            continue
        ccn = np.linalg.norm(cc)
        if ccn < 1e-12:
            continue
        cc = cc / ccn
        dots = np.clip(np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)]), -1.0, 1.0)
        rad = np.arccos(dots).max()
        cc_list.append(cc)
        rad_list.append(rad)
    if not cc_list:
        return None, None, None
    return np.array(cc_list), np.array(rad_list), simplices

def _get_min_dists(P, samples):
    dots = samples @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest_idx = np.argmin(dists, axis=1)
    return min_dists, nearest_idx

def _covering_radius_deg(pts):
    pts = _normalize(pts)
    cc, rad, _ = _facet_circumcenters(pts)
    if rad is None:
        return 180.0
    return math.degrees(float(rad.max()))

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s kill)

    # 1. Initialization: Fibonacci spiral with small noise
    P = _fibonacci_sphere(N, seed_noise=0.01)

    # 2. Facet-based smooth objective. The covering radius is the max over
    #    facet circumradii. We use a softmax over the per-facet radii to make
    #    the max differentiable, then compute the gradient via finite
    #    differences on the manifold (tangent-space projection).
    def facet_objective_and_grad(P_cur, beta=80.0):
        def f(P):
            Pn = _normalize(P)
            cc, rad, _ = _facet_circumcenters(Pn)
            if rad is None:
                return math.pi
            # Softmax of radii ~ smooth max; adding a small linear term biases
            # toward the true max as beta grows.
            m = rad.max()
            w = np.exp(beta * (rad - m))
            s = w.sum()
            if s < 1e-12:
                return m
            soft = m + np.log(s) / beta
            # tiny tie-breaker: also add mean to keep it well-defined
            return float(soft + 0.01 * rad.mean())

        def g(P):
            Pn = _normalize(P)
            base = f(Pn)
            eps = 1e-4
            grad = np.zeros_like(Pn)
            for i in range(N):
                for j in range(3):
                    v = Pn[i].copy()
                    v[j] += eps
                    v = v / np.linalg.norm(v)
                    Pp = Pn.copy()
                    Pp[i] = v
                    fp = f(Pp)
                    # directional derivative along the unit sphere
                    grad[i, j] = (fp - base) / eps
            # Project to tangent space (remove radial component)
            grad = grad - np.sum(grad * Pn, axis=1, keepdims=True) * Pn
            return grad

        return f, g

    # Phase 1: L-BFGS-B on the flattened ambient coordinates with the facet
    # objective. We periodically re-normalize inside the objective, so the
    # ambient optimizer stays on the manifold in effect.
    try:
        f, g = facet_objective_and_grad(P, beta=60.0)
        res = minimize(
            f,
            g,
            P.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 120, 'ftol': 1e-9, 'gtol': 1e-5, 'maxls': 20, 'disp': False}
        )
        if res.x is not None:
            P = _normalize(res.x.reshape(N, 3))
    except Exception:
        pass

    # Phase 2: second L-BFGS pass with a larger beta (sharper max) if time allows
    if time.time() < deadline - 45.0:
        try:
            f2, g2 = facet_objective_and_grad(P, beta=120.0)
            res2 = minimize(
                f2,
                g2,
                P.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 120, 'ftol': 1e-10, 'gtol': 1e-5, 'maxls': 20, 'disp': False}
            )
            if res2.x is not None:
                P = _normalize(res2.x.reshape(N, 3))
        except Exception:
            pass

    # Phase 3: greedy local search polish on high-res samples.
    # Move the nearest point of the worst sample toward that sample, accepting
    # only if the sampled max distance decreases.
    S_polish = 40000
    samples_polish = _fibonacci_sphere(S_polish)
    min_dists, nearest_idx = _get_min_dists(P, samples_polish)
    current_max = float(np.max(min_dists))
    rng = np.random.default_rng(7)

    for _ in range(400):
        if time.time() > deadline - 5.0:
            break
        worst_idx = int(np.argmax(min_dists))
        worst_sample = samples_polish[worst_idx]
        p_idx = int(nearest_idx[worst_idx])
        p_old = P[p_idx].copy()

        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        if grad_norm < 1e-7:
            continue
        grad_dir = grad_dir / grad_norm

        # small random perturbation to escape local minima
        rnd = rng.standard_normal(3)
        rnd = rnd - np.dot(rnd, p_old) * p_old
        rn = np.linalg.norm(rnd)
        if rn > 1e-7:
            rnd = rnd / rn
        else:
            rnd = np.zeros(3)

        best_new_max = current_max
        best_p = p_old.copy()
        for step_size in [0.03, 0.015, 0.007, 0.003]:
            new_p = p_old + step_size * (0.85 * grad_dir + 0.15 * rnd)
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[p_idx] = new_p
            md, _ = _get_min_dists(P, samples_polish)
            new_max = float(np.max(md))
            if new_max < best_new_max:
                best_new_max = new_max
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        P[p_idx] = best_p
        current_max = best_new_max
        min_dists, nearest_idx = _get_min_dists(P, samples_polish)

    P = _normalize(P)
    return P
# EVOLVE-BLOCK-END
