# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square to maximize the sum
    of their side lengths, with no interior overlaps.

    Strategy:
      1. Build several diverse initial packings (grid, corner+center, random,
         hand-crafted asymmetric layouts).
      2. For each, run SLSQP with a smooth (log-sum-exp) penalty for overlaps
         and boundary violations, maximizing sum(sides).
      3. Keep the best feasible solution; do a final polish pass with a tighter
         penalty.
    Returns (centers, sides) with the required shapes and finite positive sides.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 600s kill

    # Decision variable x = [cx_0..cx_9, cy_0..cy_9, s_0..s_9]  (30 vars)
    # Objective: maximize sum(s) -> minimize -sum(s) + penalty
    # Overlap condition (forbidden): |dx| < t and |dy| < t, t=(s_i+s_j)/2 - tol
    # We penalize any pair that is "close" in both axes beyond a smooth margin.

    def overlap_penalty(centers, sides, beta=200.0):
        """Smooth penalty for overlaps using log-sum-exp of violations."""
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                t = 0.5 * (s[i] + s[j]) - tol
                if t <= 0:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # violation amount: how much the gap is below t
                # We want penalty when dx < t AND dy < t.
                # Use a smooth "both close" indicator via product of soft violations.
                # v_x = max(0, t - dx), v_y = max(0, t - dy)
                # Penalty ~ v_x * v_y (area of overlap region approx)
                # Smooth max via softplus: max(0,a) ~ log(1+exp(beta*a))/beta
                vx = np.logaddexp(0, beta * (t - dx)) / beta
                vy = np.logaddexp(0, beta * (t - dy)) / beta
                total += vx * vy
        return total

    def boundary_penalty(centers, sides, beta=200.0):
        """Penalty for squares extending outside [0,1]^2."""
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        # left: cx - s/2 < 0  -> violation = max(0, s/2 - cx)
        # right: cx + s/2 > 1 -> violation = max(0, cx + s/2 - 1)
        # similarly for y
        viol = np.zeros(n * 4)
        viol[0:n] = np.maximum(0, s / 2 - cx)
        viol[n:2 * n] = np.maximum(0, cx + s / 2 - 1)
        viol[2 * n:3 * n] = np.maximum(0, s / 2 - cy)
        viol[3 * n:4 * n] = np.maximum(0, cx + s / 2 - 1)
        # smooth sum
        return np.sum(np.logaddexp(0, beta * viol)) / beta

    def objective(x, beta=200.0, pen_weight=10.0):
        centers = x[:2 * n].reshape(n, 2)
        sides = x[2 * n:]
        obj = -np.sum(sides)
        obj += pen_weight * (overlap_penalty(centers, sides, beta) + boundary_penalty(centers, sides, beta))
        return obj

    def gradient(x, beta=200.0, pen_weight=10.0):
        # Numerical gradient (central difference) for reliability
        eps = 1e-6
        g = np.zeros_like(x)
        f0 = objective(x, beta, pen_weight)
        for k in range(len(x)):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            g[k] = (objective(xp, beta, pen_weight) - objective(xm, beta, pen_weight)) / (2 * eps)
        return g

    def make_initializations():
        """Return a list of initial x vectors (length 30)."""
        inits = []

        # 1. 3x3 grid + one extra (like baseline but as x vector)
        k = 3
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        sides = np.full(n, side)
        x = np.concatenate([cx, cy, sides])
        inits.append(x)

        # 2. Corner squares + center
        # Place 4 corners with side 0.3, and 6 smaller in remaining space
        corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        cx = np.array([c[0] for c in corners] + [0.5, 0.5, 0.5, 0.5, 0.3, 0.7])
        cy = np.array([c[1] for c in corners] + [0.3, 0.7, 0.5, 0.5, 0.5, 0.5])
        sides = np.array([0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        x = np.concatenate([cx, cy, sides])
        inits.append(x)

        # 3. Random feasible-ish start
        rng = np.random.default_rng(42)
        cx = rng.uniform(0.1, 0.9, n)
        cy = rng.uniform(0.1, 0.9, n)
        sides = rng.uniform(0.1, 0.3, n)
        x = np.concatenate([cx, cy, sides])
        inits.append(x)

        # 4. Two-column layout: left column 5 squares, right column 5 squares
        cx = np.array([0.15]*5 + [0.85]*5)
        cy = np.array([0.1, 0.3, 0.5, 0.7, 0.9] + [0.1, 0.3, 0.5, 0.7, 0.9])
        sides = np.full(n, 0.2)
        x = np.concatenate([cx, cy, sides])
        inits.append(x)

        # 5. Asymmetric: one large square + 9 small
        cx = np.array([0.5] + [0.15, 0.85, 0.15, 0.85, 0.15, 0.85, 0.5, 0.5, 0.5])
        cy = np.array([0.25] + [0.15, 0.15, 0.85, 0.85, 0.5, 0.5, 0.5, 0.75, 0.9])
        sides = np.array([0.5] + [0.15]*9)
        x = np.concatenate([cx, cy, sides])
        inits.append(x)

        return inits

    def is_feasible(centers, sides):
        """Check if solution is feasible within tol."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # boundary
        if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
            return False
        # overlap
        for i in range(n):
            for j in range(i + 1, n):
                t = 0.5 * (sides[i] + sides[j]) - tol
                if t <= 0:
                    continue
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def score(centers, sides):
        return np.sum(sides)

    best_x = None
    best_score = -np.inf

    inits = make_initializations()

    # Run optimization from each initialization
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline:
            break

        # Try a few beta values for the smooth penalty
        for beta in [100.0, 200.0, 400.0]:
            if time.time() > deadline:
                break

            res = minimize(
                objective,
                x0,
                args=(beta, 10.0),
                method='SLSQP',
                jac=gradient,
                bounds=[(0, 1)] * 2 * n + [(1e-6, 1.0)] * n,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )

            if res.success or res.fun < objective(x0, beta, 10.0):
                centers = res.x[:2 * n].reshape(n, 2)
                sides = res.x[2 * n:]
                if is_feasible(centers, sides):
                    s = score(centers, sides)
                    if s > best_score:
                        best_score = s
                        best_x = res.x.copy()

        # Also try a local search with perturbation from current best
        if best_x is not None:
            if time.time() > deadline:
                break
            rng_local = np.random.default_rng(init_idx + 100)
            for _ in range(3):
                x_pert = best_x + rng_local.normal(0, 0.02, len(best_x))
                x_pert = np.clip(x_pert, 0, 1)
                x_pert[2 * n:] = np.clip(x_pert[2 * n:], 1e-6, 1.0)
                res2 = minimize(
                    objective,
                    x_pert,
                    args=(400.0, 10.0),
                    method='SLSQP',
                    jac=gradient,
                    bounds=[(0, 1)] * 2 * n + [(1e-6, 1.0)] * n,
                    options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
                )
                if res2.fun < objective(x_pert, 400.0, 10.0):
                    centers = res2.x[:2 * n].reshape(n, 2)
                    sides = res2.x[2 * n:]
                    if is_feasible(centers, sides):
                        s = score(centers, sides)
                        if s > best_score:
                            best_score = s
                            best_x = res2.x.copy()

    # Final polish: tighten penalty weight and beta
    if best_x is not None:
        if time.time() < deadline:
            res_final = minimize(
                objective,
                best_x,
                args=(800.0, 50.0),
                method='SLSQP',
                jac=gradient,
                bounds=[(0, 1)] * 2 * n + [(1e-6, 1.0)] * n,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            centers = res_final.x[:2 * n].reshape(n, 2)
            sides = res_final.x[2 * n:]
            if is_feasible(centers, sides):
                s = score(centers, sides)
                if s > best_score:
                    best_score = s
                    best_x = res_final.x.copy()

    # Fallback: if optimization failed, return the grid baseline
    if best_x is None:
        k = 3
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    else:
        centers = best_x[:2 * n].reshape(n, 2).astype(np.float64)
        sides = best_x[2 * n:].astype(np.float64)

    # Ensure all sides are positive and finite
    sides = np.maximum(sides, 1e-8)
    centers = np.clip(centers, 0, 1)

    return centers, sides

# EVOLVE-BLOCK-END
