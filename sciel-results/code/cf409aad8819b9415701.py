"""Shallow-circuit Ising ground-state approximation: multi-start L-BFGS-B with vectorized circuit eval."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE_S = 115.0
_START_TIME = None

_H_cache = {}
_E0_cache = {}
_GATE_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for s in range(dim):
            bi = (s >> i) & 1
            bj = (s >> (i + 1)) & 1
            zz = 1.0 if bi == bj else -1.0
            H[s, s] += -coupling * zz
    for i in range(n):
        mask = 1 << i
        for s in range(dim):
            H[s, s ^ mask] += -field
    _H_cache[key] = H
    return H


def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _E0_cache:
        return _E0_cache[key]
    e0 = np.linalg.eigvalsh(_build_H(n, coupling, field))[0]
    _E0_cache[key] = e0
    return e0


def _build_gates(n, layers):
    key = (n, layers)
    if key in _GATE_cache:
        return _GATE_cache[key]
    c = np.cos(np.pi / 4)
    s = np.sin(np.pi / 4)
    Ry = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex128)
    _GATE_cache[key] = (Ry, CZ)
    return Ry, CZ


def _apply_single_gate(psi, n, i, gate):
    """Apply a 2x2 gate on qubit i. psi is flat vector of length 2**n (qubit i is bit i)."""
    dim = 1 << n
    # reshape so that qubit i is the last axis
    axes = [j for j in range(n) if j != i] + [i]
    psi_t = psi.reshape([2] * n).transpose(axes).reshape(-1, 2)
    psi_t = gate @ psi_t
    # transpose back
    psi_t = psi_t.reshape([2] * n)
    inv_axes = [0] * n
    for pos, q in enumerate(axes):
        inv_axes[q] = pos
    psi_t = psi_t.transpose(inv_axes)
    return psi_t.reshape(dim)


def _apply_cz_gate(psi, n, i, j, gate):
    """Apply a 4x4 gate on qubits i, j. psi is flat vector of length 2**n."""
    dim = 1 << n
    # reshape so that qubits i, j are the last two axes
    axes = [k for k in range(n) if k not in (i, j)] + [i, j]
    psi_t = psi.reshape([2] * n).transpose(axes).reshape(-1, 4)
    psi_t = gate @ psi_t
    psi_t = psi_t.reshape([2] * n)
    inv_axes = [0] * n
    for pos, q in enumerate(axes):
        inv_axes[q] = pos
    psi_t = psi_t.transpose(inv_axes)
    return psi_t.reshape(dim)


def _circuit_state_vec(params, n, layers, Ry, CZ):
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    for l in range(layers):
        for i in range(n):
            psi = _apply_single_gate(psi, n, i, Ry)
        if l < layers - 1:
            for i in range(n - 1):
                psi = _apply_cz_gate(psi, n, i, i + 1, CZ)
    return psi


def _circuit_energy(params, n, coupling, field, Ry, CZ, H, e0):
    psi = _circuit_state_vec(params, n, layers, Ry, CZ)
    E = np.vdot(psi, H @ psi).real
    return (E - e0) / n


def angles(n, coupling, field, layers):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    Ry, CZ = _build_gates(n, layers)
    H = _build_H(n, coupling, field)
    e0 = _exact_ground_energy(n, coupling, field)

    def objective(flat):
        params = flat.reshape(layers, n)
        return _circuit_energy(params, n, coupling, field, Ry, CZ, H, e0)

    def time_left():
        return _DEADLINE_S - (time.time() - _START_TIME)

    bounds = [(-np.pi, np.pi)] * (layers * n)

    best_params = None
    best_val = np.inf

    rng = np.random.default_rng(0)
    n_restarts = 24
    for r in range(n_restarts):
        if time_left() < 4.0:
            break
        if r == 0:
            x0 = np.zeros(layers * n)
        else:
            x0 = rng.uniform(-np.pi, np.pi, size=layers * n)
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is not None and time_left() > 5.0:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        out = np.zeros((layers, n))
        out[-1] = np.pi / 2
        return out

    return best_params.reshape(layers, n)
# EVOLVE-BLOCK-END
