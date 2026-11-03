"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _pairwise(i, j):
    """Precomputed index arrays for i<j pairs."""
    ii = np.repeat(np.arange(N), N - 1)
    jj = np.tile(np.arange(N), N - 1)
    jj = jj[1:]  # shift to get i<j
    # Actually build properly
    ii_list = []
    jj_list = []
    for i in range(N):
        for j in range(i + 1, N):
            ii_list.append(i)
            jj_list.append(j)
    return np.array(ii_list, dtype=int), np.array(jj_list, dtype=int)

I_IDX, J_IDX = _pairwise(0, 1)
NPAIRS = len(I_IDX)


def _softplus(x, beta=20.0):
    """Numerically stable softplus: max(x,0) + log(1+exp(-beta*max(x,0)))... 
    Actually: softplus(x) = log(1+exp(beta*x))/beta, approximates max(x,0)."""
    # Use the stable form
    ax = np.abs(x)
    return np.where(x > 0, x, np.logaddexp(0.0, -x))  # log(1+exp(-x)) for x<=0, x for x>0


def _objective_and_grad(v, beta=15.0, mu=50.0):
    """
    v: (3N,) = [x_1..x_N, y_1..y_N, r_1..r_N]
    Returns (f, g) where f = -sum(r) + mu * (boundary_violation + overlap_violation)
    using softplus penalties (smooth).
    """
    x = v[0:N]
    y = v[N:2 * N]
    r = v[2 * N:3 * N]

    # Boundary violations: r - x, r - y, r - (1-x), r - (1-y)  (positive = violation)
    bv = np.concatenate([
        r - x,
        r - y,
        r - (1.0 - x),
        r - (1.0 - y),
    ])
    # Overlap violations for each pair: (r_i + r_j) - dist_ij
    ri = r[I_IDX]
    rj = r[J_IDX]
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    dist = np.sqrt(dx * dx + dy * dy)
    ov = (ri + rj) - dist
    # Avoid div by zero in gradient: guard dist
    dist_safe = np.where(dist > 1e-9, dist, 1e-9)

    # Smooth penalty via softplus
    penalty = np.sum(_softplus(bv, beta)) + np.sum(_softplus(ov, beta))

    f = -np.sum(r) + mu * penalty

    # Gradients
    # d(penalty)/d(bv_k) = sigmoid(beta * bv_k)  (derivative of softplus)
    sig_bv = 1.0 / (1.0 + np.exp(-beta * bv))
    sig_ov = 1.0 / (1.0 + np.exp(-beta * ov))

    g = np.zeros(3 * N)

    # -sum(r) gradient
    g[2 * N:3 * N] -= 1.0

    # Boundary gradient contributions
    # bv[0:N] = r - x  -> d/dx_i = -1, d/dr_i = +1
    g[0:N] += mu * sig_bv[0:N] * (-1.0)
    g[N:2 * N] += mu * sig_bv[N:2 * N] * (-1.0)
    g[2 * N:3 * N] += mu * (sig_bv[0:N] + sig_bv[N:2 * N] + sig_bv[2 * N:3 * N] + sig_bv[3 * N:4 * N])

    # Overlap gradient: ov_p = (r_i+r_j) - dist
    # d(ov_p)/dx_i = -(x_i - x_j)/dist = (x_j - x_i)/dist
    # d(ov_p)/dx_j = +(x_i - x_j)/dist
    # d(ov_p)/dr_i = 1, d(ov_p)/dr_j = 1
    for p in range(NPAIRS):
        i = I_IDX[p]
        j = J_IDX[p]
        s = mu * sig_ov[p]
        d = dist_safe[p]
        g[i] += s * (x[j] - x[i]) / d
        g[j] += s * (x[i] - x[j]) / d
        g[N + i] += s * (y[j] - y[i]) / d
        g[N + j] += s * (y[i] - y[j]) / d
        g[2 * N + i] += s
        g[2 * N + j] += s

    return f, g


def _check_valid(arr):
    """Return True if all circles are inside [0,1]^2 with disjoint interiors (strict)."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    eps = 1e-9
    if np.any(r <= 0):
        return False
    if np.any(x - r < -eps) or np.any(y - r < -eps):
        return False
    if np.any(x + r > 1.0 + eps) or np.any(y + r > 1.0 + eps):
        return False
    # pairwise
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < (r[i] + r[j]) - eps:
                return False
    return True


def _repair(arr):
    """Shrink radii slightly and nudge to satisfy constraints strictly. Returns valid array."""
    arr = arr.copy()
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    # Clamp positions to [r, 1-r] iteratively
    for _ in range(50):
        r_new = np.clip(r, 1e-6, None)
        # boundary
        x = np.clip(x, r_new, 1.0 - r_new)
        y = np.clip(y, r_new, 1.0 - r_new)
        # pairwise: if overlap, shrink the smaller radius and push apart
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                need = r[i] + r[j]
                if d < need - 1e-12:
                    # push apart along direction, shrink radii a bit
                    if d < 1e-12:
                        # random small nudge
                        ang = np.random.uniform(0, 2 * np.pi)
                        dx, dy = np.cos(ang), np.sin(ang)
                        d = 1e-12
                    ux, uy = dx / d, dy / d
                    gap = (need - d)
                    # move each by half the gap
                    x[i] += ux * gap * 0.5
                    y[i] += uy * gap * 0.5
                    x[j] -= ux * gap * 0.5
                    y[j] -= uy * gap * 0.5
                    # shrink radii to ensure separation
                    shrink = gap * 0.5 + 1e-10
                    r[i] = max(1e-6, r[i] - shrink)
                    r[j] = max(1e-6, r[j] - shrink)
                    changed = True
        arr[:, 0], arr[:, 1], arr[:, 2] = x, y, r
        if not changed:
            break
    # final strict clamp
    r = np.clip(r, 1e-6, None)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    arr[:, 0], arr[:, 1], arr[:, 2] = x, y, r
    return arr


def _make_initial(rng):
    """Generate a random feasible-ish initial configuration."""
    # Start with a grid-like base then jitter
    side = int(np.ceil(np.sqrt(N)))
    base = []
    for i in range(side):
        for j in range(side):
            base.append(((i + 0.5) / side, (j + 0.5) / side))
    base = base[:N]
    pts = np.array(base)
    # jitter
    pts = pts + rng.uniform(-0.03, 0.03, size=pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    # equal-ish radii
    r0 = 0.5 / side * 0.9
    radii = np.full(N, r0) + rng.uniform(-0.005, 0.005, N)
    radii = np.clip(radii, 1e-3, 0.3)
    arr = np.column_stack([pts[:, 0], pts[:, 1], radii])
    return arr


def construct():
    rng = np.random.default_rng(12345)
    best_arr = None
    best_val = -np.inf

    # A good baseline: grid
    side = int(np.ceil(np.sqrt(N)))
    grid = []
    for i in range(side):
        for j in range(side):
            grid.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    grid_arr = np.array(grid[:N])
    if _check_valid(grid_arr):
        best_arr = grid_arr.copy()
        best_val = float(np.sum(grid_arr[:, 2]))

    n_starts = 20
    for start in range(n_starts):
        if time.time() > DEADLINE - 20:
            break
        # initial config
        if start == 0:
            arr0 = _make_initial(rng)
        else:
            # perturb best so far
            if best_arr is not None:
                arr0 = best_arr.copy()
                arr0[:, 0] += rng.uniform(-0.05, 0.05, N)
                arr0[:, 1] += rng.uniform(-0.05, 0.05, N)
                arr0[:, 2] *= rng.uniform(0.85, 1.0, N)
            else:
                arr0 = _make_initial(rng)
        arr0 = _repair(arr0)

        v0 = arr0.reshape(-1)

        # Multi-stage beta/mu schedule
        for beta, mu, iters in [(10.0, 20.0, 80), (20.0, 50.0, 120), (40.0, 150.0, 200)]:
            if time.time() > DEADLINE - 15:
                break
            res = minimize(
                lambda v: _objective_and_grad(v, beta, mu),
                v0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': iters, 'maxls': 40, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            v0 = res.x

        arr = v0.reshape(N, 3)
        arr = _repair(arr)

        if _check_valid(arr):
            val = float(np.sum(arr[:, 2]))
            if val > best_val:
                best_val = val
                best_arr = arr.copy()

    if best_arr is None:
        # fallback to grid
        best_arr = grid_arr
    return best_arr


# Self-test on import (light)
if __name__ == '__main__':
    t0 = time.time()
    out = construct()
    print("sum r =", np.sum(out[:, 2]), "valid =", _check_valid(out), "time =", time.time() - t0)
