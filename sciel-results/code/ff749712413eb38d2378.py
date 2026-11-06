"""Model-and-optimize: fast bit-manipulation circuit + physics-informed multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_TIME_LIMIT = 148.0  # explicit safety margin before the 150s kill
_cache = {}


def _build_hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h = 0.0
        for i in range(n - 1):
            bi = (idx >> i) & 1
            bp = (idx >> (i + 1)) & 1
            zi = 1.0 if bi == 0 else -1.0
            zp = 1.0 if bp == 0 else -1.0
            h += -coupling * zi * zp
        H[idx, idx] = h
        for i in range(n):
            H[idx, idx ^ (1 << i)] += -field
    return H


def _get_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    w = np.linalg.eigvalsh(H)
    return float(w[0])


def _apply_ry(state, q, theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    if abs(theta) < 1e-15:
        return state
    shape = [2] * state.ndim
    sr = state.reshape(shape)
    m = np.moveaxis(sr, q, 0)
    out = np.empty_like(m)
    out[0] = c * m[0] - s * m[1]
    out[1] = s * m[0] + c * m[1]
    return np.moveaxis(out, 0, q).reshape(-1)


def _circuit_forward(params, n, layers):
    thetas = params.reshape(layers, n)
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry(state, q, thetas[layer, q])
        if layer < layers - 1:
            for q in range(n - 1):
                shape = [2] * n
                sr = state.reshape(shape)
                m = np.moveaxis(sr, [q, q + 1], [-2, -1])
                idx_11 = (slice(None),) * (n - 2) + (1, 1)
                m[idx_11] *= -1.0
                state = np.moveaxis(m, [-2, -1], [q, q + 1]).reshape(-1)
    return state


def _energy_and_grad(params, n, coupling, field, layers, H, e0):
    thetas = params.reshape(layers, n)
    dim = 1 << n
    states_before_ry = []
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    for layer in range(layers):
        for q in range(n):
            states_before_ry.append(state)  # reference, no copy needed (forward only)
            state = _apply_ry(state, q, thetas[layer, q])
        if layer < layers - 1:
            for q in range(n - 1):
                shape = [2] * n
                sr = state.reshape(shape)
                m = np.moveaxis(sr, [q, q + 1], [-2, -1])
                idx_11 = (slice(None),) * (n - 2) + (1, 1)
                m[idx_11] *= -1.0
                state = np.moveaxis(m, [-2, -1], [q, q + 1]).reshape(-1)
    psi_c = state
    Hpsi = H @ psi_c
    energy = np.vdot(psi_c, Hpsi).real
    obj_val = (energy - e0) / n

    grad = np.zeros((layers, n), dtype=np.float64)
    adjoint = Hpsi.copy()
    gate_idx = len(states_before_ry) - 1
    for layer in range(layers - 1, -1, -1):
        if layer < layers - 1:
            for q in range(n - 2, -1, -1):
                shape = [2] * n
                sr = adjoint.reshape(shape)
                m = np.moveaxis(sr, [q, q + 1], [-2, -1])
                idx_11 = (slice(None),) * (n - 2) + (1, 1)
                m[idx_11] *= -1.0
                adjoint = np.moveaxis(m, [-2, -1], [q, q + 1]).reshape(-1)
        for q in range(n - 1, -1, -1):
            theta = thetas[layer, q]
            psi_b = states_before_ry[gate_idx]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            v_moved = np.moveaxis(adjoint.reshape([2] * n), q, 0)
            psi_moved = np.moveaxis(psi_b.reshape([2] * n), q, 0)
            dp0 = 0.5 * (-s * psi_moved[0] - c * psi_moved[1])
            dp1 = 0.5 * (c * psi_moved[0] - s * psi_moved[1])
            inner = np.vdot(v_moved, np.stack([dp0, dp1], axis=0))
            grad[layer, q] = 2.0 * inner.real
            v_new = np.empty_like(v_moved)
            v_new[0] = c * v_moved[0] + s * v_moved[1]
            v_new[1] = -s * v_moved[0] + c * v_moved[1]
            adjoint = np.moveaxis(v_new, 0, q).reshape(-1)
            gate_idx -= 1
    grad_flat = (grad / n).flatten()
    return obj_val, grad_flat


def _physics_informed_init(n, layers, field, coupling):
    """Start near the X-field alignment angle, with a ZZ-correlation tilt per layer."""
    # X-field wants each qubit toward |->: last-layer angle ~ -pi/2 (or pi/2 up to phase).
    base = -np.pi / 2.0
    # Low field -> strong ZZ (ferro) correlations; high field -> more product-like.
    tilt = 0.15 * coupling / (1.0 + field)
    x = np.zeros(layers * n)
    for l in range(layers):
        # alternate small sign to encourage staggered alignment
        sign = 1.0 if (l % 2 == 0) else -1.0
        for q in range(n):
            x[l * n + q] = base + sign * tilt * (1.0 if (q % 2 == 0) else -1.0)
    return x


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    e0 = _get_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    np_params = layers * n

    def obj(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)[0]

    def jac(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)[1]

    def objgrad(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)

    # Build a diverse, physics-informed restart set.
    restarts = []
    restarts.append(_physics_informed_init(n, layers, field, coupling))
    restarts.append(_physics_informed_init(n, layers, field, coupling) * -1.0)
    for val in [0.0, np.pi / 4, np.pi / 2, -np.pi / 2, np.pi]:
        restarts.append(np.full(np_params, val))
    for l in range(layers):
        for val in [np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 2]:
            x = np.zeros(np_params)
            x[l * n:(l + 1) * n] = val
            restarts.append(x)
    rng = np.random.RandomState(2024)
    for _ in range(6):
        restarts.append(rng.randn(np_params) * (np.pi * 0.5))
    for _ in range(4):
        restarts.append(rng.uniform(-np.pi, np.pi, np_params))

    best_val = np.inf
    best_x = restarts[0].copy()

    for x0 in restarts:
        if time.time() - _start_time > _TIME_LIMIT - 12.0:
            break
        x0 = np.clip(x0, -np.pi, np.pi)
        try:
            res = opt.minimize(
                obj, x0, jac=jac, method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 40},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final tight polish.
    if time.time() - _start_time < _TIME_LIMIT - 6.0:
        try:
            res2 = opt.minimize(
                obj, best_x, jac=jac, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 30},
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass

    theta = best_x.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
