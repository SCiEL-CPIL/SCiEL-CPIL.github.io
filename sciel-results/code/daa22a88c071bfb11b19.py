"""Program for covering the sphere with 130 equal caps."""
import time
import numpy as np
import math

N = 130


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


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets (empty caps)."""
    pts = _normalize(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) < 3:
            continue
        v0, v1, v2 = verts[0], verts[1], verts[2]
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        nl = np.linalg.norm(normal)
        if nl < 1e-15:
            continue
        normal = normal / nl
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    return math.degrees(max_rad)


def _facet_circumcenter(pts, simplex):
    """Return the circumcenter (on unit sphere) of a triangular facet, or None."""
    verts = pts[simplex]
    if len(verts) < 3:
        return None
    a, b, c = verts[0], verts[1], verts[2]
    e1 = b - a
    e2 = c - a
    normal = np.cross(e1, e2)
    nl = np.linalg.norm(normal)
    if nl < 1e-15:
        return None
    normal = normal / nl
    e1_sq = np.dot(e1, e1)
    e2_sq = np.dot(e2, e2)
    e1_e2 = np.dot(e1, e2)
    A = np.array([[2 * e1_sq, 2 * e1_e2], [2 * e1_e2, 2 * e2_sq]])
    bvec = np.array([e1_sq, e2_sq])
    try:
        st = np.linalg.solve(A, bvec)
    except np.linalg.LinAlgError:
        return None
    circ = a + st[0] * e1 + st[1] * e2
    cn = np.linalg.norm(circ)
    if cn < 1e-15:
        return None
    circ = circ / cn
    # orient circumcenter to the same side as the normal (outward)
    if np.dot(circ, normal) < 0:
        circ = -circ
    return circ


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Dense sample set for the worst-case objective (fixed, reused across phases)
    S = 20000
    samples = _fibonacci_sphere(S, seed=7)

    def min_dist_and_nearest(P):
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)
        return min_dists, nearest

    def worst_point_gradient(P, min_dists, nearest, k_focus):
        """Riemannian gradient toward the k worst sample points (subgradient of max)."""
        # pick the k worst samples
        k = min(k_focus, S)
        worst_idx = np.argpartition(min_dists, -k)[-k:]
        w_min = min_dists[worst_idx]
        w_nearest = nearest[worst_idx]
        w_samples = samples[worst_idx]

        # weight each worst point by its distance (worst gets most weight)
        wts = w_min + 1e-9
        wts = wts / np.sum(wts)

        # gradient of arccos(dot) wrt P_j: -dot/(sqrt(1-dot^2)) * sample
        dots_n = np.clip(w_samples @ P.T, -1.0, 1.0)
        # for each worst sample i, its nearest point j=w_nearest[i]
        dn = dots_n[np.arange(k), w_nearest]
        denom = np.sqrt(np.maximum(1e-12, 1.0 - dn * dn))
        # gradient contribution (in ambient space) for each (i, j):
        # d/dP_j arccos(dn) = -dn/denom * sample_i
        gP = np.zeros_like(P)
        # accumulate per point j
        coef = -wts * dn / denom  # shape (k,)
        np.add.at(gP, w_nearest, coef[:, None] * w_samples)

        # project to tangent space
        gP_dot = np.sum(gP * P, axis=1, keepdims=True)
        gP = gP - gP_dot * P
        return gP, worst_idx

    # ---------------- Phase 1: Riemannian SGD with annealing restarts ----------------
    P = _fibonacci_sphere(N, seed_noise=0.02, seed=1)
    best_P = P.copy()
    best_val = _covering_radius_deg(P)

    # temperature schedule
    T0 = 0.12
    T_end = 0.004
    n_steps = 2600
    k_focus = 6

    rng = np.random.default_rng(2024)

    for step in range(n_steps):
        if time.time() > deadline - 25.0:
            break
        # anneal temperature
        frac = step / max(1, n_steps - 1)
        T = T0 + (T_end - T0) * frac

        min_dists, nearest = min_dist_and_nearest(P)
        gP, _ = worst_point_gradient(P, min_dists, nearest, k_focus)

        # adaptive step size (normalize gradient magnitude)
        gnorm = np.linalg.norm(gP)
        if gnorm < 1e-12:
            continue
        gP = gP / gnorm

        lr = 0.06
        newP = P - lr * gP

        # thermal noise (annealing) to escape local minima
        noise = rng.normal(0, T, (N, 3))
        newP = newP + noise
        newP = _normalize(newP)

        # accept if it improves the sampled max, or with probability exp(-delta/T)
        min_dists_new, _ = min_dist_and_nearest(newP)
        cur_max = np.max(min_dists)
        new_max = np.max(min_dists_new)
        delta = new_max - cur_max
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-6)):
            P = newP
            # track best by true hull radius occasionally
            if step % 60 == 0:
                val = _covering_radius_deg(P)
                if val < best_val:
                    best_val = val
                    best_P = P.copy()

    P = best_P.copy()
    best_val = _covering_radius_deg(P)

    # ---------------- Phase 2: facet-circumcenter-aware polish ----------------
    # Directly shrink the largest hull facet circumradius by moving its vertices
    # toward the facet circumcenter (the true worst point of that facet).
    if time.time() < deadline - 10.0:
        from scipy.spatial import ConvexHull
        for it in range(400):
            if time.time() > deadline - 10.0:
                break
            try:
                hull = ConvexHull(P)
            except Exception:
                break

            best_rad = 0.0
            best_simplex = None
            best_circ = None
            for simplex in hull.simplices:
                circ = _facet_circumcenter(P, simplex)
                if circ is None:
                    continue
                # circumradius = angular distance from circ to any vertex
                rad = np.arccos(np.clip(np.dot(circ, P[simplex[0]]), -1.0, 1.0))
                if rad > best_rad:
                    best_rad = rad
                    best_simplex = simplex
                    best_circ = circ

            if best_simplex is None:
                break

            # move each vertex of the worst facet toward the circumcenter
            improved = False
            for v_idx in best_simplex:
                v = P[v_idx]
                proj = np.dot(v, best_circ)
                if proj > 0.999999:
                    continue
                grad_dir = best_circ - proj * v
                gn = np.linalg.norm(grad_dir)
                if gn < 1e-8:
                    continue
                grad_dir = grad_dir / gn

                best_step = 0.0
                best_new_rad = best_rad
                best_v = v.copy()
                for ss in [0.08, 0.04, 0.02, 0.01, 0.005]:
                    new_v = _normalize((v + ss * grad_dir).reshape(1, 3))[0]
                    P[v_idx] = new_v
                    new_circ = _facet_circumcenter(P, best_simplex)
                    if new_circ is None:
                        P[v_idx] = v
                        continue
                    new_rad = np.arccos(np.clip(np.dot(new_circ, P[best_simplex[0]]), -1.0, 1.0))
                    if new_rad < best_new_rad:
                        best_new_rad = new_rad
                        best_step = ss
                        best_v = new_v.copy()
                    else:
                        P[v_idx] = v
                if best_step > 0:
                    P[v_idx] = best_v
                    improved = True
                else:
                    P[v_idx] = v
            if not improved:
                break

            # update best
            val = _covering_radius_deg(P)
            if val < best_val:
                best_val = val
                best_P = P.copy()

    # ---------------- Phase 3: final greedy polish on dense samples ----------------
    if time.time() < deadline - 3.0:
        S_polish = 40000
        samples_p = _fibonacci_sphere(S_polish, seed=11)

        def min_dists_p(Pp):
            dots = samples_p @ Pp.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            return np.min(dists, axis=1), np.argmin(dists, axis=1)

        for _ in range(200):
            if time.time() > deadline - 3.0:
                break
            md, ni = min_dists_p(P)
            worst = np.argmax(md)
            ws = samples_p[worst]
            p_idx = ni[worst]
            v = P[p_idx]
            proj = np.dot(v, ws)
            if proj > 0.999999:
                continue
            grad_dir = ws - proj * v
            gn = np.linalg.norm(grad_dir)
            if gn < 1e-6:
                continue
            grad_dir = grad_dir / gn
            cur_max = np.max(md)
            best_step = 0.0
            best_new = cur_max
            best_v = v.copy()
            for ss in [0.05, 0.02, 0.01, 0.005]:
                new_v = _normalize((v + ss * grad_dir).reshape(1, 3))[0]
                P[p_idx] = new_v
                md_t, _ = min_dists_p(P)
                new_max = np.max(md_t)
                if new_max < best_new:
                    best_new = new_max
                    best_step = ss
                    best_v = new_v.copy()
                else:
                    P[p_idx] = v
            if best_step > 0:
                P[p_idx] = best_v
            else:
                P[p_idx] = v

        val = _covering_radius_deg(P)
        if val < best_val:
            best_val = val
            best_P = P.copy()

    return _normalize(best_P)


# EVOLVE-BLOCK-END
