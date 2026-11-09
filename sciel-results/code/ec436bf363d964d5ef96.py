# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(poly: np.ndarray) -> np.ndarray:
    c = poly.mean(axis=0)
    p = poly - c
    d = np.linalg.norm(p, axis=1)
    m = d.mean()
    if m <= 0.01:
        return p
    return p / m


def _vertex_score(p: np.ndarray) -> float:
    n = p.shape[0]
    i = 0
    # distances from vertex 0 to others
    diff = p[1:] - p[0]
    d = np.linalg.norm(diff, axis=1)
    d_sorted = np.sort(d)
    if d_sorted[0] < 0.05:
        return 0.0
    gaps = np.diff(d_sorted)
    k = int(np.argmin(gaps))
    D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = [0, 1, 2, 3]
    elif k == 17:
        idx = [15, 16, 17, 18]
    else:
        idx = [k - 1, k, k + 1, k + 2]
    four = d_sorted[idx]
    raw = np.mean(np.maximum(four / D, D / four))
    return min(1.0, 1.0 / raw) if raw > 0 else 0.0


def _score_polygon(p: np.ndarray) -> float:
    pn = _normalize(p)
    # strict convexity check
    n = pn.shape[0]
    for i in range(n):
        if np.any(np.allclose(pn[i], pn[(i + 1) % n], atol=1e-9)):
            return 0.0
    turns = []
    for i in range(n):
        e1 = pn[(i + 1) % n] - pn[i]
        e2 = pn[(i + 2) % n] - pn[(i + 1) % n]
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        turns.append(cross)
    turns = np.array(turns)
    if np.any(np.abs(turns) < 1e-12):
        return 0.0
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return 0.0
    total = np.sum(np.arctan2(turns, np.einsum('ij,ij->i',
        pn[1:] - pn[:-1], pn[2:] - pn[1:-1])))
    # simpler: sum of signed angles
    angles = []
    for i in range(n):
        e1 = pn[(i + 1) % n] - pn[i]
        e2 = pn[(i + 2) % n] - pn[(i + 1) % n]
        a1 = np.arctan2(e1[1], e1[0])
        a2 = np.arctan2(e2[1], e2[0])
        da = a2 - a1
        while da > np.pi:
            da -= 2 * np.pi
        while da < -np.pi:
            da += 2 * np.pi
        angles.append(da)
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 0.1:
        return 0.0
    scores = [_vertex_score(pn) for _ in range(n)]
    # actually compute for each vertex
    scores = []
    for i in range(n):
        # rotate so vertex i is at index 0
        pi = np.roll(pn, -i, axis=0)
        scores.append(_vertex_score(pi))
    return min(scores)


def _build_polygon(n: int, symmetry: int, angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Build polygon with given symmetry. angles and radii are for one period."""
    period = n // symmetry
    # angles and radii should have length `period`
    all_angles = np.tile(angles, symmetry)
    all_radii = np.tile(radii, symmetry)
    # offset angles by symmetry fraction
    base_offset = 2 * np.pi / n
    # Actually for cyclic symmetry of order s, we rotate by 2*pi/s
    # Let's rebuild: for symmetry s, the polygon has s copies of a period of n/s vertices
    # The period vertices are at angles theta_0..theta_{p-1} and radii r_0..r_{p-1}
    # Copy j is rotated by j * 2*pi/s
    p = period
    pts = []
    for j in range(symmetry):
        rot = j * 2 * np.pi / symmetry
        for i in range(p):
            a = angles[i] + rot
            r = radii[i]
            pts.append([r * np.cos(a), r * np.sin(a)])
    pts = np.array(pts)
    # sort by angle to ensure cyclic order
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(ang)
    return pts[order]


def _score_params(n: int, symmetry: int, angles: np.ndarray, radii: np.ndarray) -> float:
    try:
        p = _build_polygon(n, symmetry, angles, radii)
        if p.shape != (n, 2):
            return 0.0
        s = _score_polygon(p)
        return s
    except Exception:
        return 0.0


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    t0 = time.time()
    deadline = 600.0
    margin = 30.0

    best_score = 0.0
    best_poly = None

    # Regular polygon baseline
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii_reg = np.ones(n)
    p_reg = np.stack([np.cos(angles_reg), np.sin(angles_reg)], axis=1)
    best_poly = p_reg.copy()
    best_score = _score_polygon(p_reg)

    # Try different symmetries
    symmetries = [10, 5, 4, 2, 1]
    for sym in symmetries:
        if time.time() > deadline - margin:
            break
        period = n // sym
        # Initial guess: uniform within period
        base_angles = np.linspace(0, 2 * np.pi / sym, period, endpoint=False)
        base_radii = np.ones(period)

        # Local coordinate ascent
        current_angles = base_angles.copy()
        current_radii = base_radii.copy()
        current_score = _score_params(n, sym, current_angles, current_radii)

        step_a = 0.01
        step_r = 0.01
        max_iter = 5000
        iter_count = 0

        improved = True
        while improved and iter_count < max_iter:
            if time.time() > deadline - margin:
                break
            improved = False
            iter_count += 1
            # Try perturbing each angle and radius
            for i in range(period):
                # angle up
                trial_a = current_angles.copy()
                trial_a[i] += step_a
                s = _score_params(n, sym, trial_a, current_radii)
                if s > current_score + 1e-12:
                    current_angles = trial_a
                    current_score = s
                    improved = True
                    continue
                # angle down
                trial_a = current_angles.copy()
                trial_a[i] -= step_a
                s = _score_params(n, sym, trial_a, current_radii)
                if s > current_score + 1e-12:
                    current_angles = trial_a
                    current_score = s
                    improved = True
                    continue
                # radius up
                trial_r = current_radii.copy()
                trial_r[i] += step_r
                s = _score_params(n, sym, current_angles, trial_r)
                if s > current_score + 1e-12:
                    current_radii = trial_r
                    current_score = s
                    improved = True
                    continue
                # radius down
                trial_r = current_radii.copy()
                trial_r[i] -= step_r
                s = _score_params(n, sym, current_angles, trial_r)
                if s > current_score + 1e-12:
                    current_radii = trial_r
                    current_score = s
                    improved = True
                    continue
            # Reduce step size if no improvement in this pass
            if not improved:
                step_a *= 0.8
                step_r *= 0.8
                if step_a < 1e-5 or step_r < 1e-5:
                    break

        if current_score > best_score:
            best_score = current_score
            best_poly = _build_polygon(n, sym, current_angles, current_radii)

    # Also try no symmetry (full 20 params) with a few Nelder-Mead restarts
    if time.time() < deadline - margin:
        from scipy.optimize import minimize

        def _full_score(params: np.ndarray) -> float:
            angles = params[:n]
            radii = params[n:]
            p = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
            return -_score_polygon(p)

        # A few restarts
        for restart in range(5):
            if time.time() > deadline - margin:
                break
            if restart == 0:
                angles_init = np.linspace(0, 2 * np.pi, n, endpoint=False)
                radii_init = np.ones(n)
            else:
                noise_scale = 0.1 * restart
                angles_init = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.uniform(-noise_scale, noise_scale, n)
                radii_init = 1.0 + np.random.uniform(-noise_scale, noise_scale, n)

            x0 = np.concatenate([angles_init, radii_init])
            try:
                res = minimize(_full_score, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    angles_f = res.x[:n]
                    radii_f = res.x[n:]
                    best_poly = np.stack([radii_f * np.cos(angles_f), radii_f * np.sin(angles_f)], axis=1)
            except Exception:
                pass

    return best_poly


# EVOLVE-BLOCK-END
