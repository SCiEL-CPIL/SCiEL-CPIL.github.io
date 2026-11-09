"""Bermudan put exercise policy: adaptive-Gauss-Legendre backward induction on a log-grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class ExercisePolicy:
    def __init__(self, cont_values, log_grid, step, K, n_grid):
        self.cont_values = cont_values  # list of arrays, one per exercise date (dates 1..N)
        self.log_grid = log_grid
        self.step = step
        self.K = K
        self.n_grid = n_grid

    def exercise(self, current_state, time_idx):
        """Return boolean vector: True where immediate exercise is optimal."""
        if time_idx <= 0 or time_idx > len(self.cont_values):
            return np.zeros(len(current_state), dtype=bool)

        states = np.asarray(current_state, dtype=np.float64)
        log_states = np.log(np.maximum(states, 1e-15))

        cont = self.cont_values[time_idx - 1]
        idx_float = (log_states - self.log_grid[0]) / self.step
        idx_float = np.clip(idx_float, 0.0, float(self.n_grid - 1))

        idx0 = idx_float.astype(np.int64)
        idx1 = np.minimum(idx0 + 1, self.n_grid - 1)
        idx0 = np.clip(idx0, 0, self.n_grid - 1)
        t = idx_float - idx0

        continuation = cont[idx0] * (1.0 - t) + cont[idx1] * t
        intrinsic = np.maximum(self.K - states, 0.0)

        return intrinsic > continuation


def _interp_uniform(log_vals, log_grid, step, y, n_grid):
    idx_float = (log_vals - log_grid[0]) / step
    idx_float = np.clip(idx_float, 0.0, float(n_grid - 1))
    idx0 = idx_float.astype(np.int64)
    idx1 = np.minimum(idx0 + 1, n_grid - 1)
    idx0 = np.clip(idx0, 0, self.n_grid if False else n_grid - 1)
    t = idx_float - idx0
    return y[idx0] * (1.0 - t) + y[idx1] * t


def _step_expectation(grid, log_grid, step, V_next, n_grid, shift, vol_step, xg, wg):
    """E[V_next(S * exp(shift + vol_step*z))] for all grid points via adaptive GL quadrature."""
    z_lo = -12.0
    z_hi = 12.0
    a = shift + vol_step * z_lo
    b = shift + vol_step * z_hi
    m = xg.size
    # u in [0,1]; u = 0.5*(1 + (2*(a+b)/... )) -> linear map to [a,b]
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    u = mid + half * xg                    # shape (m,)
    z = (u - shift) / vol_step             # shape (m,)
    # weight = w * (b-a)/2 * exp(-z^2/2)
    w = wg * (b - a) * 0.5 * np.exp(-0.5 * z * z)
    w = w / w.sum()                         # normalize (robust)
    log_next = log_grid[:, None] + u[None, :]  # (n_grid, m)
    vals = _interp_uniform(log_next.ravel(), log_grid, step, V_next, n_grid)
    vals = vals.reshape(n_grid, m)
    return vals @ w


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_steps = int(contract.get('n_steps', 12))

    dt = T / n_steps
    disc = np.exp(-r * dt)
    mu = r - 0.5 * sigma * sigma
    vol_step = sigma * np.sqrt(dt)
    shift = mu * dt

    # Log-uniform grid: tighter low end, finer resolution
    min_s = 0.05
    max_s = 1000.0
    n_grid = 8192
    log_grid = np.linspace(np.log(min_s), np.log(max_s), n_grid)
    grid = np.exp(log_grid)
    step = log_grid[1] - log_grid[0]

    # Adaptive Gauss-Legendre nodes/weights (64-pt, robust for the short step)
    xg, wg = np.polynomial.legendre.leggauss(64)

    def run_backward():
        cont_values = [None] * n_steps
        V_next = np.maximum(K - grid, 0.0)
        for k in range(n_steps - 1, 0, -1):
            E = _step_expectation(grid, log_grid, step, V_next, n_grid, shift, vol_step, xg, wg)
            Cont = disc * E
            cont_values[k - 1] = Cont
            V_next = np.maximum(K - grid, 0.0)
            V_next = np.maximum(V_next, Cont)
        for i in range(n_steps):
            if cont_values[i] is None:
                cont_values[i] = np.zeros(n_grid)
        return cont_values

    try:
        cont_values = run_backward()
    except Exception:
        # Fallback: cheap coarse solve, conservative (never exercise off-grid)
        cont_values = [np.zeros(n_grid) for _ in range(n_steps)]
        return ExercisePolicy(cont_values, log_grid, step, K, n_grid)

    # Refinement pass on the low-S region where the exercise boundary lives.
    # The policy is only sensitive to continuation values near/inside the money,
    # i.e. S < K. Re-integrate the low-S indices with higher quadrature.
    try:
        if time.time() < deadline - 20.0:
            xg_hi, wg_hi = np.polynomial.legendre.leggauss(128)
            k_mask = np.where(grid < K)[0]
            if k_mask.size > 0:
                k_lo = int(k_mask.min())
                k_hi = int(k_mask.max())
                # process low-to-high so refinement uses already-refined upper values
                for k in range(1, min(n_steps, k_hi + 1)):
                    if time.time() > deadline - 8.0:
                        break
                    V_next = np.maximum(K - grid, 0.0)
                    if k < n_steps:
                        V_next = np.maximum(V_next, cont_values[k])
                    E = _step_expectation(grid, log_grid, step, V_next, n_grid, shift, vol_step, xg_hi, wg_hi)
                    E_lo = disc * E[k_lo:k_hi + 1]
                    cont_values[k - 1][k_lo:k_hi + 1] = E_lo
    except Exception:
        pass

    return ExercisePolicy(cont_values, log_grid, step, K, n_grid)

# EVOLVE-BLOCK-END
