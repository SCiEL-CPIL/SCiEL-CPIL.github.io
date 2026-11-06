"""Model-and-optimize: parameterize the 3-layer Ry/CZ circuit and directly
minimize the exact mean energy gap per spin against the exact ground state."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize

# Explicit global wall-clock deadline guard (evaluator kills at 150 s).
_GLOBAL_DEADLINE = time.time() + 140.0
_CACHE = {}


def _apply_ry(psi, n, qubit, theta):
    """Apply single-qubit Ry(theta) on qubit (bit qubit). Returns new vector."""
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)
    mask = 1 << qubit
    dim = len(psi)
    out = psi.copy()
    # Vectorized: find states where qubit is 0
    idx = np.arange(dim)
    zero_mask = (idx & mask) == 0
    zero_idx = idx[zero_mask]
    one_idx = zero_idx | mask
    a0 = psi[zero_idx]
    a1 = psi[one_idx]
    out[zero_idx] = c * a0 - s * a1
    out[one_idx] = s * a0 + c * a1
    return out


def _apply_cz(psi, n, a, b):
    """Apply controlled-Z on qubits (a, b). Returns new vector."""
    mask = (1 << a) | (1 << b)
    dim = len(psi)
    out = psi.copy()
    idx = np.arange(dim)
    both_one = (idx & mask) == mask
    out[both_one] = -out[both_one]
    return out


def _circuit_state(theta, n, layers):
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for l in range(layers):
        for i in range(n):
            psi = _apply_ry(psi, n, i, theta[l, i])
        if l < layers - 1:
            for i in range(n - 1):
                psi = _apply_cz(psi, n, i, i + 1)
    return psi


def _build_H(n, coupling, field):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    # ZZ (Ising) terms: -coupling * sum_i Z_i Z_{i+1}
    for i in range(n - 1):
        ma = 1 << i
        mb = 1 << (i + 1)
        for state in range(dim):
            za = 1.0 if ((state & ma) == 0) else -1.0
            zb = 1.0 if ((state & mb) == 0) else -1.0
            H[state, state] += -coupling * za * zb
    # Transverse-field terms: -field * sum_i X_i
    for i in range(n):
        m = 1 << i
        for state in range(dim):
            H[state, state ^ m] += -field
    return H


def _ground_energy(n, coupling, field):
    H = _build_H(n, coupling, field)
    return float(np.min(np.linalg.eigvalsh(H))), H


def _optimize_case(n, coupling, field, layers, time_budget):
    E0, H = _ground_energy(n, coupling, field)
    dim = 2 ** n

    def energy(theta):
        psi = _circuit_state(theta, n, layers)
        return float(np.real(np.vdot(psi, H @ psi)))

    def obj(theta_flat):
        theta = theta_flat.reshape(layers, n)
        return (energy(theta) - E0) / n

    bounds = [(-np.pi, np.pi)] * (layers * n)
    rng = np.random.default_rng(12345)

    starts = [np.zeros((layers, n))]
    n_starts = 20
    for _ in range(n_starts):
        starts.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))

    best_theta = np.zeros((layers, n))
    best_val = np.inf
    t0 = time.time()

    for s in starts:
        if time.time() - t0 > time_budget - 3.0:
            break
        try:
            res = scipy.optimize.minimize(
                obj, s.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 150, "ftol": 1e-14, "gtol": 1e-10},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Final high-precision polish pass with tighter tolerance
    if time.time() - t0 < time_budget - 1.0 and best_val < np.inf:
        try:
            res_polish = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-16, "gtol": 1e-12},
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_theta = res_polish.x.reshape(layers, n)
        except Exception:
            pass

    # Guard against non-finite results.
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((layers, n))
    return best_theta


def angles(n, coupling, field, layers):
    key = (int(n), float(coupling), float(field), int(layers))
    if key in _CACHE:
        return _CACHE[key].copy()

    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining <= 3.0:
        # Not enough time: return a sane product-state default (|+>^n on last layer).
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _CACHE[key] = theta
        return theta.copy()

    # Per-call budget: keep total under the deadline with a safety margin.
    budget = min(remaining * 0.85, 18.0)

    try:
        theta = _optimize_case(n, coupling, field, layers, budget)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    if not np.all(np.isfinite(theta)):
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    _CACHE[key] = theta
    return theta.copy()
# EVOLVE-BLOCK-END
