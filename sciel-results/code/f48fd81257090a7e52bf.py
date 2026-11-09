import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # explicit safety margin before the 630s kill

    def elapsed():
        return time.time() - start_time

    def is_valid(centers, sides, tol=1e-7):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True

    def score(centers, sides):
        if not is_valid(centers, sides):
            return -np.inf
        return np.sum(sides)

    best_centers = None
    best_sides = None
    best_score = -np.inf

    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if centers is None or sides is None:
            return
        try:
            s = score(centers, sides)
        except Exception:
            return
        if s > best_score:
            best_score = s
            best_centers = np.array(centers, dtype=float).copy()
            best_sides = np.array(sides, dtype=float).copy()

    # ------------------------------------------------------------------
    # Seed configurations (explicit topologies, structural idea)
    # ------------------------------------------------------------------
    seeds = []

    # Topology A: 2 large (0.5) on top, 8 small (0.25) in 2x4 grid below. Sum = 3.0
    cA = np.zeros((n, 2)); sA = np.zeros(n)
    cA[0] = [0.25, 0.75]; sA[0] = 0.5
    cA[1] = [0.75, 0.75]; sA[1] = 0.5
    for r in range(2):
        for col in range(4):
            idx = 2 + r * 4 + col
            cA[idx] = [0.125 + col * 0.25, 0.125 + r * 0.25]
            sA[idx] = 0.25
    seeds.append((cA, sA))
    update_best(cA, sA)

    # Topology B: uniform 2x5 grid, side 0.2. Sum = 2.0 (weak but a different basin)
    cB = np.zeros((n, 2)); sB = np.zeros(n)
    for i in range(n):
        r = i // 5; c = i % 5
        cB[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        sB[i] = 0.2
    seeds.append((cB, sB))

    # Topology C: 1 large (0.5) + 9 small (0.25). Sum = 0.5 + 9*0.25 = 2.75
    cC = np.zeros((n, 2)); sC = np.zeros(n)
    cC[0] = [0.25, 0.75]; sC[0] = 0.5
    # right column 0.5 wide: 2x2 grid of 0.25
    for r in range(2):
        for c in range(2):
            idx = 1 + r * 2 + c
            cC[idx, 0] = 0.75 + c * 0.25 - 0.125
            cC[idx, 1] = 0.375 + r * 0.25
            sC[idx] = 0.25
    # bottom-left 0.5x0.5: 2x2 grid of 0.25
    for r in range(2):
        for c in range(2):
            idx = 5 + r * 2 + c
            cC[idx, 0] = 0.125 + c * 0.25
            cC[idx, 1] = 0.125 + r * 0.25
            sC[idx] = 0.25
    seeds.append((cC, sC))
    update_best(cC, sC)

    # ------------------------------------------------------------------
    # Penalty objective (DE / SLSQP), same family as current program
    # ------------------------------------------------------------------
    def penalty_objective(params):
        centers = params[:2 * n].reshape(n, 2)
        sides = params[2 * n:]
        if np.any(sides <= 0):
            return 1e6
        obj = -np.sum(sides)
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            v_l = max(0.0, -(cx - s / 2))
            v_r = max(0.0, (cx + s / 2) - 1)
            v_b = max(0.0, -(cy - s / 2))
            v_t = max(0.0, (cy + s / 2) - 1)
            obj += 1000.0 * (v_l * v_l + v_r * v_r + v_b * v_b + v_t * v_t)
        tol = 1e-7
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t <= 0:
                    continue
                ox = max(0.0, t - dx)
                oy = max(0.0, t - dy)
                if ox > 0 and oy > 0:
                    obj += 1000.0 * ox * oy
        return obj

    bounds = [(0.0, 1.0)] * (2 * n + n)

    def pack(centers, sides):
        p = np.empty(3 * n)
        p[:2 * n] = centers.reshape(-1)
        p[2 * n:] = sides
        return p

    def unpack(p):
        c = p[:2 * n].reshape(n, 2)
        s = np.maximum(p[2 * n:], 1e-6)
        return c, s

    # ------------------------------------------------------------------
    # Smooth-surrogate local polish (log-sum-exp epigraph of the max
    # overlap violation) -- refinement machinery. This is a smooth
    # relaxation of the hard penalty that lets a local solver nudge the
    # configuration across the nonsmooth "touching" boundary.
    # ------------------------------------------------------------------
    def smooth_polish(centers0, sides0):
        p0 = pack(centers0, sides0)
        beta = 1000.0  # sharpness of the log-sum-exp

        def smooth_obj(p):
            centers = p[:2 * n].reshape(n, 2)
            sides = np.maximum(p[2 * n:], 1e-6)
            val = -np.sum(sides)
            # boundary: soft max of the four violations
            v = np.zeros((n, 4))
            v[:, 0] = -(centers[:, 0] - sides / 2)
            v[:, 1] = (centers[:, 0] + sides / 2) - 1
            v[:, 2] = -(centers[:, 1] - sides / 2)
            v[:, 3] = (centers[:, 1] + sides / 2) - 1
            vmax = np.max(v, axis=1)
            soft = vmax + np.log(np.sum(np.exp(v - vmax[:, None]), axis=1) / 4.0) / beta
            val += 500.0 * np.sum(np.maximum(soft, 0.0))
            # overlap: soft max of (ox, oy) per pair
            tol = 1e-7
            for i in range(n):
                for j in range(i + 1, n):
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    t = (sides[i] + sides[j]) / 2.0 - tol
                    if t <= 0:
                        continue
                    ox = t - dx
                    oy = t - dy
                    m = max(ox, oy)
                    if m <= 0:
                        continue
                    softm = m + np.log(np.exp(ox - m) + np.exp(oy - m)) / beta
                    val += 500.0 * max(0.0, softm)
            return val

        try:
            res = minimize(smooth_obj, p0, method='SLSQP',
                           options={'maxiter': 300, 'ftol': 1e-13})
            c, s = unpack(res.x)
            update_best(c, s)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    try:
        res = differential_evolution(penalty_objective, bounds, maxiter=120,
                                     popsize=12, tol=1e-7, disp=False, seed=42)
        c_opt, s_opt = unpack(res.x)
        update_best(c_opt, s_opt)
    except Exception:
        pass

    # Polish the DE result and every seed with the smooth surrogate
    candidates = []
    try:
        candidates.append(unpack(res.x))
    except Exception:
        pass
    for c0, s0 in seeds:
        candidates.append((c0.copy(), s0.copy()))

    for c0, s0 in candidates:
        if elapsed() > deadline - 120:
            break
        smooth_polish(c0, s0)

    # Final hard-penalty SLSQP polish on the best found, then smooth polish
    if best_centers is not None and elapsed() < deadline - 60:
        p_best = pack(best_centers, best_sides)
        try:
            r2 = minimize(penalty_objective, p_best, method='SLSQP',
                          options={'maxiter': 400, 'ftol': 1e-13})
            c2, s2 = unpack(r2.x)
            update_best(c2, s2)
        except Exception:
            pass
        if elapsed() < deadline - 30:
            smooth_polish(best_centers, best_sides)

    if best_centers is None:
        best_centers, best_sides = cA, sA

    # Final clamp for safety
    best_sides = np.maximum(best_sides, 1e-9)
    return best_centers, best_sides
