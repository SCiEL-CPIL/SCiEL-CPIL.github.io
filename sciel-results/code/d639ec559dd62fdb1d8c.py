import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 142.0
_START = None
_best_cache = {}


def _init():
    global _START
    if _START is None:
        _START = time.time()
    return _START


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _circuit_state(params, n):
    layers = params.shape[0]
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0 + 0j
    for l in range(layers):
        theta = params[l]
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        for i in range(n):
            bit = 1 << i
            state = c * state + 1j * s * (state << bit)
        if l < layers - 1:
            for i in range(n - 1):
                b1 = 1 << i
                b2 = 1 << (i + 1)
                mask = b1 | b2
                state &= (state ^ (state << b1)) & (state ^ (state << b2)) & (state ^ mask)
    return state


def _energy(params, n, coupling, field):
    state = _circuit_state(params, n)
    norm = np.vdot(state, state).real
    if norm <= 0:
        return 1e9
    e = 0.0
    for i in range(n - 1):
        b1 = 1 << i
        b2 = 1 << (i + 1)
        e -= coupling * (np.vdot(state, state << b1) + np.vdot(state, state << b2)).real
    for i in range(n):
        b = 1 << i
        e -= field * np.vdot(state, state << b).real
    return e / norm


def _grad(params, n, coupling, field):
    layers = params.shape[0]
    g = np.zeros_like(params)
    for l in range(layers):
        for i in range(n):
            eps = 1e-7
            p = params.copy()
            p[l, i] += eps
            e1 = _energy(p, n, coupling, field)
            p[l, i] -= 2.0 * eps
            e0 = _energy(p, n, coupling, field)
            g[l, i] = (e1 - e0) / (2.0 * eps)
    return g


def _polish(params, n, coupling, field, budget):
    t0 = time.time()
    res = minimize(
        _energy, params, args=(n, coupling, field),
        jac=_grad, method="L-BFGS-B",
        options={"maxiter": 200, "maxls": 40, "ftol": 1e-12, "gtol": 1e-8},
    )
    if time.time() - t0 > budget or not np.all(np.isfinite(res.x)):
        return params
    return np.clip(res.x, -np.pi, np.pi)


def _opt_case(n, coupling, field, layers, budget):
    best = None
    best_e = np.inf
    t0 = time.time()
    rng = np.random.default_rng(12345 + 17 * n + int(round(field * 100)))
    while time.time() - t0 < budget and _remaining() > 15.0:
        params = rng.uniform(-np.pi, np.pi, size=(layers, n))
        e0 = _energy(params, n, coupling, field)
        if e0 < best_e:
            best_e, best = e0, params.copy()
        params = _polish(params, n, coupling, field, budget=30.0)
        e1 = _energy(params, n, coupling, field)
        if e1 < best_e:
            best_e, best = e1, params.copy()
    return best


def angles(n, coupling, field, layers):
    _init()
    key = (n, coupling, field, layers)
    if key in _best_cache:
        return _best_cache[key].copy()

    if _remaining() < 25.0:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _best_cache[key] = theta
        return theta.copy()

    params = _opt_case(n, coupling, field, layers, budget=30.0)
    if params is None:
        params = np.zeros((layers, n))
        params[-1] = np.pi / 2
    _best_cache[key] = params
    return params.copy()
