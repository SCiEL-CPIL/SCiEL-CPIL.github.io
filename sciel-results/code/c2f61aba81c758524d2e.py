import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # explicit safety margin before the 630s kill

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

    # Topology D: 4 large (0.375) in corners + 6 small (0.25) in center strip. Sum = 4*0.375 + 6*0.25 = 1.5+1.5 = 3.0
    # Layout: corners have 0.375 squares, leaving a 0.25-wide cross in the middle
    # Center strip: 0.25 wide horizontally and vertically, can fit 6 squares of 0.25
    cD = np.zeros((n, 2)); sD = np.zeros(n)
    # 4 corner squares of side 0.375
    cD[0] = [0.1875, 0.1875]; sD[0] = 0.375  # bottom-left
    cD[1] = [0.8125, 0.1875]; sD[1] = 0.375  # bottom-right
    cD[2] = [0.1875, 0.8125]; sD[2] = 0.375  # top-left
    cD[3] = [0.8125, 0.8125]; sD[3] = 0.375  # top-right
    # 6 small squares of side 0.25 in the central region
    # The gap between corners: x from 0.375 to 0.625 (width 0.25), y from 0.375 to 0.625 (height 0.25)
    # That's only 1 square of 0.25. Need to extend the layout.
    # Better: use a 3x3 grid where corners are 0.375 and edges/center are 0.25
    # Actually, let's try: 4 squares of 0.375 in a 2x2 arrangement on left, 6 of 0.25 on right
    # Left half [0, 0.5]: 2x2 grid of 0.25 squares won't work with 0.375
    # Let me think differently: stack vertically
    # Top: 2 squares of 0.4 side by side (width 0.8), height 0.4 -> y: [0.6, 1.0]
    # Bottom: 8 squares... no that's too many
    # Try: 5 squares of 0.3 on top row, 5 squares of 0.3 on bottom row? Sum=3.0
    # But 5*0.3=1.5 > 1.0, doesn't fit
    # Try: 2 squares of 0.45 on top (width 0.9), 8 squares of 0.225 below? Sum = 0.9+1.8=2.7
    # Let's try the actual optimal: 4 of 0.375 + 6 of 0.25 = 3.0
    # Arrangement: 
    #   Row 1 (top): [0.375, 0.25, 0.375] but that's 1.0 wide, 3 squares
    #   Actually place 4 large in a 2x2 pattern taking up 0.75x0.75, and 6 small around
    # Let's just use a parameterized approach and let optimizer refine
    # Start with: 4 squares of 0.375 at corners of a 0.75x0.75 area, 6 of 0.25 filling gaps
    # Place large squares:
    #   BL: [0, 0.375] x [0, 0.375]
    #   BR: [0.625, 1.0] x [0, 0.375]  
    #   TL: [0, 0.375] x [0.625, 1.0]
    #   TR: [0.625, 1.0] x [0.625, 1.0]
    # Gap in middle: [0.375, 0.625] x [0.375, 0.625] = 0.25x0.25, fits 1 small
    # Gap left: [0.375, 0.625] x [0, 0.375] and [0, 0.375] x [0.375, 0.625] etc
    # This is getting complex. Let me just provide a reasonable starting point
    # and let the optimizer find the optimum.
    # Simple symmetric start: 10 squares in a pattern that the optimizer can refine
    # Use 2 rows of 5, each row with alternating large/small
    for i in range(n):
        r = i // 5
        c = i % 5
        if c % 2 == 0:  # large
            sD[i] = 0.375
        else:  # small
            sD[i] = 0.25
        cD[i, 0] = (c + 0.5) * 0.2
        cD[i, 1] = 0.25 + r * 0.5
    seeds.append((cD, sD))
    update_best(cD, sD)

    # Topology E: 2 large (0.45) + 8 small (0.225). Sum = 0.9 + 1.8 = 2.7
    cE = np.zeros((n, 2)); sE = np.zeros(n)
    cE[0] = [0.225, 0.775]; sE[0] = 0.45
    cE[1] = [0.775, 0.775]; sE[1] = 0.45
    for i in range(8):
        r = i // 4
        c = i % 4
        cE[2 + i, 0] = (c + 0.5) * 0.25
        cE[2 + i, 1] = 0.1125 + r * 0.225
        sE[2 + i] = 0.225
    seeds.append((cE, sE))
    update_best(cE, sE)

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
    def smooth_polish(centers0, sides0, beta=None):
        p0 = pack(centers0, sides0)
        if beta is None:
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
            # Numerically stable log-sum-exp
            exp_v = np.exp(v - vmax[:, None])
            soft = vmax + np.log(np.sum(exp_v, axis=1) / 4.0) / beta
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
                    # Numerically stable
                    softm = m + np.log(np.exp(ox - m) + np.exp(oy - m)) / beta
                    val += 500.0 * max(0.0, softm)
            return val

        try:
            # Try multiple beta values for robustness
            for b in [beta, beta * 2, beta * 5]:
                if elapsed() > deadline - 60:
                    break
                try:
                    res = minimize(smooth_obj, p0, method='SLSQP',
                                   options={'maxiter': 400, 'ftol': 1e-14})
                    c, s = unpack(res.x)
                    update_best(c, s)
                    # If we improved, use the new point as starting point for next beta
                    if best_centers is not None and best_score > score(centers0, sides0):
                        p0 = pack(best_centers, best_sides)
                except Exception:
                    continue
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    de_results = []
    # Run DE with multiple seeds for diversity
    de_seeds = [42, 123, 456, 789, 1000]
    for seed in de_seeds:
        if elapsed() > deadline - 180:
            break
        try:
            res = differential_evolution(penalty_objective, bounds, maxiter=100,
                                         popsize=12, tol=1e-7, disp=False, seed=seed)
            c_opt, s_opt = unpack(res.x)
            update_best(c_opt, s_opt)
            de_results.append((c_opt, s_opt))
        except Exception:
            pass

    # Polish the DE results and every seed with the smooth surrogate
    candidates = []
    for c0, s0 in de_results:
        candidates.append((c0.copy(), s0.copy()))
    for c0, s0 in seeds:
        candidates.append((c0.copy(), s0.copy()))

    for c0, s0 in candidates:
        if elapsed() > deadline - 150:
            break
        smooth_polish(c0, s0)

    # Additional DE runs if time permits, with tighter bounds around best solution
    if best_centers is not None and elapsed() < deadline - 240:
        # Local DE around the best solution
        local_bounds = []
        for i in range(n):
            local_bounds.append((max(0, best_centers[i,0] - 0.1), min(1, best_centers[i,0] + 0.1)))
            local_bounds.append((max(0, best_centers[i,1] - 0.1), min(1, best_centers[i,1] + 0.1)))
        for i in range(n):
            local_bounds.append((max(0.01, best_sides[i] - 0.05), min(1.0, best_sides[i] + 0.05)))
        try:
            res_local = differential_evolution(penalty_objective, local_bounds, maxiter=80,
                                                popsize=10, tol=1e-8, disp=False, seed=777)
            c_loc, s_loc = unpack(res_local.x)
            update_best(c_loc, s_loc)
            if elapsed() < deadline - 120:
                smooth_polish(c_loc, s_loc)
        except Exception:
            pass

    # Final hard-penalty SLSQP polish on the best found, then smooth polish
    if best_centers is not None and elapsed() < deadline - 90:
        p_best = pack(best_centers, best_sides)
        try:
            r2 = minimize(penalty_objective, p_best, method='SLSQP',
                          options={'maxiter': 500, 'ftol': 1e-14})
            c2, s2 = unpack(r2.x)
            update_best(c2, s2)
        except Exception:
            pass
        if elapsed() < deadline - 60:
            smooth_polish(best_centers, best_sides, beta=5000)
        
        # One more round of polishing with very high beta for final precision
        if elapsed() < deadline - 30:
            smooth_polish(best_centers, best_sides, beta=10000)

    if best_centers is None:
        best_centers, best_sides = cA, sA

    # Final clamp for safety
    best_sides = np.maximum(best_sides, 1e-9)
    # Ensure all values are finite and within bounds
    best_centers = np.clip(best_centers, 0.0, 1.0)
    return best_centers, best_sides
