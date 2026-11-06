import numpy as np
from scipy.optimize import minimize
import time


def _simulate_circuit(theta, n, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0

    for l in range(layers):
        for i in range(n):
            half = theta[l, i] / 2.0
            c = np.cos(half)
            s = np.sin(half)
            mask = 1 << i
            for idx in range(dim):
                if idx & mask:
                    state[idx] *= -s
                else:
                    state[idx] *= c
        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state /= norm

        if l < layers - 1:
            for i in range(n - 1):
                mask1 = 1 << i
                mask2 = 1 << (i + 1)
                for idx in range(dim):
                    if (idx & mask1) and (idx & mask2):
                        state[idx] *= -1.0

    return state


def _compute_energy(state, n, coupling, field):
    dim = 1 << n
    probs = state * state
    E = 0.0

    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        zz = 0.0
        for idx in range(dim):
            bit1 = 1 if (idx & mask1) else 0
            bit2 = 1 if (idx & mask2) else 0
            z1 = 1 - 2 * bit1
            z2 = 1 - 2 * bit2
            zz += z1 * z2 * probs[idx]
        E -= coupling * zz

    for i in range(n):
        mask = 1 << i
        x = 0.0
        for idx in range(dim):
            if not (idx & mask):
                partner = idx | mask
                x += 2.0 * state[idx] * state[partner]
        E -= field * x

    return E


def _exact_ground_energy(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for idx in range(dim):
        diag = 0.0
        for i in range(n - 1):
            mask1 = 1 << i
            mask2 = 1 << (i + 1)
            bit1 = 1 if (idx & mask1) else 0
            bit2 = 1 if (idx & mask2) else 0
            z1 = 1 - 2 * bit1
            z2 = 1 - 2 * bit2
            diag -= coupling * z1 * z2
        H[idx, idx] += diag

        for i in range(n):
            mask = 1 << i
            if not (idx & mask):
                partner = idx | mask
                H[idx, partner] -= field
                H[partner, idx] -= field

    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _objective(x, n, coupling, field, layers, e0):
    theta = x.reshape(layers, n)
    state = _simulate_circuit(theta, n, layers)
    e = _compute_energy(state, n, coupling, field)
    return (e - e0) / n


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 145.0

    e0 = _exact_ground_energy(n, coupling, field)

    def obj(x):
        return _objective(x, n, coupling, field, layers, e0)

    best_theta = None
    best_val = np.inf

    inits = []
    inits.append(np.zeros(layers * n))
    inits.append(np.full(layers * n, np.pi / 2))

    t3 = np.zeros(layers * n)
    t3[(layers - 1) * n:layers * n] = np.pi / 2
    inits.append(t3)

    t1 = np.zeros(layers * n)
    t1[0:n] = np.pi / 2
    inits.append(t1)

    inits.append(np.full(layers * n, np.pi / 2))

    rng = np.random.default_rng(12345)
    for _ in range(8):
        inits.append(rng.uniform(-np.pi, np.pi, layers * n))

    for val in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        inits.append(np.full(layers * n, val))

    for val in [0.4, 0.8, 1.2, 1.6]:
        t = np.zeros(layers * n)
        for layer in range(layers):
            for i in range(n):
                t[layer * n + i] = val if i % 2 == 0 else -val
        inits.append(t)

    for base in [0.3, 0.5, 0.8]:
        t = np.zeros(layers * n)
        for layer in range(layers):
            for i in range(n):
                t[layer * n + i] = base * (layer + 1) / layers
        inits.append(t)

    for init in inits:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(obj, init, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            val = res.fun
            if val < best_val:
                best_val = val
                best_theta = res.x.copy()
        except Exception:
            continue

    if best_theta is not None and time.time() - start_time < deadline - 5:
        try:
            res = minimize(obj, best_theta, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass

    if best_theta is not None and time.time() - start_time < deadline - 2:
        try:
            res = minimize(obj, best_theta, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass

    if best_theta is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    return best_theta.reshape(layers, n)
