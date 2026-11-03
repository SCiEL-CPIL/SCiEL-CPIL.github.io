"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 150.0
_START = None


def _elapsed():
    return time.time() - _START


def _smooth_max(v, beta):
    """Smooth max via log-sum-exp; gradient-friendly."""
    m = np.max(v)
    if m < -50:
        return 0.0, np.zeros_like(v)
    exp = np.exp(beta * (v - m))
    s = np.sum(exp)
    sm = m + np.log(s) / beta
    g = exp / s
    return sm, g


def _make_objective(beta):
    """Return objective and gradient for sum radii - penalty."""

    def fun(x):
        pos = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        # boundary violations: circle must stay inside [0,1]^2
        # violations = max(0, r - pos) for left/bottom, max(0, r - (1-pos)) for right/top
        viol_l = r - pos[:, 0]
        viol_b = r - pos[:, 1]
        viol_r = r - (1.0 - pos[:, 0])
        viol_t = r - (1.0 - pos[:, 1])
        # overlap violations: for each pair, need dist >= r_i + r_j
        # violation = (r_i + r_j) - dist
        n = len(r)
        overlap_viol = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                v = r[i] + r[j] - d
                overlap_viol[i, j] = v
                overlap_viol[j, i] = v

        # smooth max over all violations
        all_viol = np.concatenate([viol_l, viol_b, viol_r, viol_t,
                                   overlap_viol[~np.eye(n, dtype=bool)].ravel()])
        sm, g = _smooth_max(all_viol, beta)
        obj = -np.sum(r) + sm
        return obj

    def grad(x):
        pos = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        n = len(r)

        viol_l = r - pos[:, 0]
        viol_b = r - pos[:, 1]
        viol_r = r - (1.0 - pos[:, 0])
        viol_t = r - (1.0 - pos[:, 1])

        overlap_viol = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                v = r[i] + r[j] - d
                overlap_viol[i, j] = v
                overlap_viol[j, i] = v

        all_viol = np.concatenate([viol_l, viol_b, viol_r, viol_t,
                                   overlap_viol[~np.eye(n, dtype=bool)].ravel()])
        _, g = _smooth_max(all_viol, beta)

        gx = np.zeros_like(x)
        # boundary gradients
        # d(viol_l)/d(pos_x) = -1, d/d(r)=1
        gx[0::2] += -g[:n] + g[2*n:3*n]  # from viol_l and viol_r
        gx[1::2] += -g[n:2*n] + g[3*n:4*n]  # from viol_b and viol_t
        # d(viol_l)/dr = 1, d(viol_b)/dr=1, d(viol_r)/dr=1, d(viol_t)/dr=1
        gr = g[:4*n].sum(axis=0) if False else (g[:n] + g[n:2*n] + g[2*n:3*n] + g[3*n:4*n])
        # overlap gradients
        for i in range(n):
            for j in range(i + 1, n):
                gi = g[4*n + i * (n - 1) + (j if j < i else j - 1)]
                gj = gi
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist > 1e-12:
                    gx[2*i] += gi * d[0] / dist
                    gx[2*i+1] += gi * d[1] / dist
                    gx[2*j] -= gj * d[0] / dist
                    gx[2*j+1] -= gj * d[1] / dist
                gx[2*n + i] += gi
                gx[2*n + j] += gj
        gr += np.zeros(n)
        # recompute overlap contribution to r
        for i in range(n):
            for j in range(i + 1, n):
                gi = g[4*n + i * (n - 1) + (j if j < i else j - 1)]
                gr[i] += gi
                gr[j] += gi

        gx[2*n:] = gr - 1.0  # objective is -sum(r) + sm, so d(-sum r)/dr = -1
        return gx

    return fun, grad


def _check_valid(arr):
    """Check if packing is valid (disjoint, inside unit square)."""
    pos = arr[:, :2]
    r = arr[:, 2]
    if np.any(r <= 0) or np.any(pos < 0) or np.any(pos > 1):
        return False
    if np.any(pos + r > 1 + 1e-9) or np.any(pos - r < -1e-9):
        return False
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < r[i] + r[j] - 1e-9:
                return False
    return True


def _score(arr):
    if not _check_valid(arr):
        return -np.inf
    return np.sum(arr[:, 2])


def _polish(arr, beta_start=50.0, max_iter=30):
    """Refine a valid packing by increasing beta (tighter smooth max)."""
    x = arr.ravel()
    best = arr.copy()
    best_score = _score(arr)
    for it in range(max_iter):
        beta = beta_start * (1.5 ** it)
        fun, grad = _make_objective(beta)
        try:
            res = minimize(fun, x, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success or res.fun < 0:
                new_arr = res.x.reshape(-1, 3)
                # clip radii to positive
                new_arr[:, 2] = np.maximum(new_arr[:, 2], 1e-6)
                new_score = _score(new_arr)
                if new_score > best_score:
                    best = new_arr
                    best_score = new_score
                    x = new_arr.ravel()
        except Exception:
            break
    return best


def _random_start(scale=1.0):
    """Generate a random initial packing."""
    pos = np.random.rand(N, 2) * 0.8 + 0.1
    r = np.random.rand(N) * 0.05 * scale + 0.01
    return np.column_stack([pos, r])


def _grid_start():
    """Grid-based initial packing."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.002
            cells.append([x, y, r])
    return np.array(cells[:N])


def construct():
    global _START
    _START = time.time()

    best_arr = _grid_start()
    best_score = _score(best_arr)

    # Try a few random starts
    rng = np.random.RandomState(42)
    n_restarts = 0
    max_restarts = 40

    for restart in range(max_restarts):
        if _elapsed() > _DEADLINE - 30:
            break

        # Mix of random and perturbed grid starts
        if restart % 3 == 0:
            arr = _grid_start()
            arr[:, :2] += rng.randn(N, 2) * 0.02
            arr[:, 2] = np.maximum(arr[:, 2] * (1 + rng.randn(N) * 0.1), 0.005)
        elif restart % 3 == 1:
            arr = _random_start()
        else:
            # clustered start: group circles into regions
            arr = np.zeros((N, 3))
            for i in range(N):
                cx = rng.rand() * 0.6 + 0.2
                cy = rng.rand() * 0.6 + 0.2
                arr[i, 0] = cx + rng.randn() * 0.03
                arr[i, 1] = cy + rng.randn() * 0.03
                arr[i, 2] = 0.03 + rng.rand() * 0.02
            # shuffle and spread
            idx = rng.permutation(N)
            for i in range(N):
                arr[i] = arr[idx[i]]
                arr[i, 0] = rng.rand()
                arr[i, 1] = rng.rand()

        # Optimize with moderate beta
        beta = 30.0
        fun, grad = _make_objective(beta)
        x0 = arr.ravel()
        try:
            res = minimize(fun, x0, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                new_arr = res.x.reshape(-1, 3)
                new_arr[:, 2] = np.maximum(new_arr[:, 2], 1e-6)
                # Polish
                polished = _polish(new_arr, beta_start=50.0, max_iter=20)
                s = _score(polished)
                if s > best_score:
                    best_score = s
                    best_arr = polished
        except Exception:
            pass

        n_restarts += 1

    # Final polish on best
    if _elapsed() < _DEADLINE - 15:
        final = _polish(best_arr, beta_start=100.0, max_iter=40)
        if _score(final) > best_score:
            best_arr = final
            best_score = _score(final)

    # Ensure validity
    if not _check_valid(best_arr):
        # Fallback to grid
        best_arr = _grid_start()

    return best_arr
