"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 118.0


def _time_left():
    return _DEADLINE - time.time()


def _build_cut_diag(n, edges):
    """Build diagonal of cost Hamiltonian (cut value for each computational basis state)."""
    dim = 1 << n
    diag = np.zeros(dim, dtype=np.float64)
    for state_idx in range(dim):
        cut_val = 0.0
        for u, v in edges:
            if ((state_idx >> u) & 1) != ((state_idx >> v) & 1):
                cut_val += 1.0
        diag[state_idx] = cut_val
    return diag


def _apply_mixer(state, beta, n):
    """Apply exp(-i beta sum X_i) to state vector."""
    c = np.cos(beta)
    s = np.sin(beta)
    # Rx = [[c, -is], [-is, c]]
    # For each qubit, new[0] = c*old[0] - is*old[1], new[1] = -is*old[0] + c*old[1]
    # Reshape to (2, 2, ..., 2) with n axes
    shape = [2] * n
    st = state.reshape(shape)
    for axis in range(n):
        st = np.moveaxis(st, axis, 0)
        # st shape: (2, 2, ..., 2) with first axis being the qubit
        old0 = st[0]
        old1 = st[1]
        new0 = c * old0 - 1j * s * old1
        new1 = -1j * s * old0 + c * old1
        st = np.stack([new0, new1], axis=0)
        st = np.moveaxis(st, 0, axis)
    return st.reshape(-1)


def _expected_cut(params, cut_diag, n):
    """Compute expected cut value for given QAOA parameters."""
    dim = 1 << n
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    g1, b1, g2, b2 = params
    # Layer 1
    state *= np.exp(-1j * g1 * cut_diag)
    state = _apply_mixer(state, b1, n)
    # Layer 2
    state *= np.exp(-1j * g2 * cut_diag)
    state = _apply_mixer(state, b2, n)
    probs = np.abs(state) ** 2
    return float(np.dot(probs, cut_diag))


def _neg_cut(params, cut_diag, n):
    return -_expected_cut(params, cut_diag, n)


def angles(n, edges, p):
    p = int(p)
    if _time_left() < 3.0:
        return np.tile([0.35, 0.25], (p, 1))

    cut_diag = _build_cut_diag(n, edges)
    dim = 1 << n

    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    best_val = _expected_cut(best_params, cut_diag, n)

    # Coarse grid search
    n_grid = 10
    grid = np.linspace(0.0, np.pi, n_grid)

    for g1 in grid:
        if _time_left() < 8.0:
            break
        for b1 in grid:
            if _time_left() < 8.0:
                break
            for g2 in grid:
                if _time_left() < 8.0:
                    break
                for b2 in grid:
                    if _time_left() < 8.0:
                        break
                    val = _expected_cut([g1, b1, g2, b2], cut_diag, n)
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])

    # Local refinement with Powell
    if _time_left() > 20.0:
        try:
            res = minimize(_neg_cut, best_params.copy(), args=(cut_diag, n),
                           method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 1500})
            if -res.fun > best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass

    # Additional restarts with L-BFGS-B
    if _time_left() > 30.0:
        rng = np.random.default_rng(12345)
        n_restarts = 5
        for _ in range(n_restarts):
            if _time_left() < 12.0:
                break
            init = rng.uniform(0.05, np.pi - 0.05, size=4)
            try:
                res2 = minimize(_neg_cut, init, args=(cut_diag, n),
                                method='L-BFGS-B',
                                bounds=[(0.001, np.pi - 0.001)] * 4,
                                options={'maxiter': 150, 'gtol': 1e-8, 'maxfun': 400})
                if -res2.fun > best_val:
                    best_val = -res2.fun
                    best_params = res2.x
            except Exception:
                pass

    # Final polish with Nelder-Mead
    if _time_left() > 15.0:
        try:
            res3 = minimize(_neg_cut, best_params.copy(), args=(cut_diag, n),
                            method='Nelder-Mead',
                            options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10,
                                     'adaptive': True, 'initial_simplex': 0.03})
            if -res3.fun > best_val:
                best_val = -res3.fun
                best_params = res3.x
        except Exception:
            pass

    best_params = np.asarray(best_params, dtype=float)
    best_params = np.clip(best_params, 0.0, np.pi)

    if p == 2:
        return best_params.reshape(2, 2)
    if p <= 2:
        return best_params[:2 * p].reshape(p, 2)
    return np.tile(best_params.reshape(2, 2), ((p + 1) // 2, 1))[:p]
