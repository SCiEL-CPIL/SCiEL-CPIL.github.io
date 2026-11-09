import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigvalsh


def _build_hamiltonian(n, coupling, field):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        op = np.array([[1]], dtype=complex)
        for j in range(n):
            if j == i or j == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I2)
        H -= coupling * op

    # X terms: -field * X_i
    for i in range(n):
        op = np.array([[1]], dtype=complex)
        for j in range(n):
            if j == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I2)
        H -= field * op

    return H


def _ground_energy(H):
    return float(np.min(eigvalsh(H)))


def _circuit_state(params, n, layers):
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            new_state = np.zeros(dim, dtype=complex)
            for idx in range(dim):
                if (idx >> q) & 1 == 0:
                    idx1 = idx | (1 << q)
                    new_state[idx] = c * state[idx] - s * state[idx1]
                else:
                    idx0 = idx & ~(1 << q)
                    new_state[idx] = s * state[idx0] + c * state[idx]
            state = new_state

        # Apply CZ between adjacent pairs (not after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                new_state = state.copy()
                for idx in range(dim):
                    if (idx & mask1) and (idx & mask2):
                        new_state[idx] = -state[idx]
                state = new_state

    return state


def angles(n, coupling, field, layers):
    t_start = time.time()
    deadline = t_start + 110  # Safety margin before 120s limit

    H = _build_hamiltonian(n, coupling, field)
    e0 = _ground_energy(H)

    def objective(x):
        params = x.reshape((layers, n))
        state = _circuit_state(params, n, layers)
        e = np.real(np.vdot(state, H @ state))
        return (e - e0) / n

    best_val = np.inf
    best_params = None

    rng = np.random.default_rng(123)
    n_inits = 10

    for init_i in range(n_inits):
        if time.time() > deadline:
            break

        if init_i == 0:
            x0 = np.zeros((layers, n))
            x0[-1] = np.pi / 2
        elif init_i == 1:
            x0 = np.zeros((layers, n))
        elif init_i == 2:
            x0 = rng.uniform(-0.5, 0.5, (layers, n))
        elif init_i == 3:
            x0 = rng.uniform(-1.0, 1.0, (layers, n))
        elif init_i == 4:
            x0 = np.tile(rng.uniform(-0.3, 0.3, (layers, 1)), (1, n))
        elif init_i == 5:
            x0 = np.full((layers, n), np.pi / 4)
            x0[0] = 0
        elif init_i == 6:
            x0 = rng.uniform(-np.pi / 2, np.pi / 2, (layers, n))
        elif init_i == 7:
            x0 = rng.uniform(-np.pi, np.pi, (layers, n))
        elif init_i == 8:
            x0 = np.full((layers, n), np.pi / 3)
        else:
            x0 = rng.uniform(-np.pi / 4, np.pi / 4, (layers, n))

        try:
            result = minimize(
                objective,
                x0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            val = float(result.fun)
            if val < best_val:
                best_val = val
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass

    if best_params is None:
        best_params = np.zeros((layers, n))
        best_params[-1] = np.pi / 2

    return best_params
