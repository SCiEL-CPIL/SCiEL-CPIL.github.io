import numpy as np
import time
from scipy.optimize import minimize


def _build_hamiltonian(n, coupling, field):
    """H = -coupling * sum Z_i Z_{i+1} - field * sum X_i, little-endian."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)

    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            if bi == bj:
                H[b, b] -= coupling
            else:
                H[b, b] += coupling

    # X terms: off-diagonal (symmetric)
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:
                H[b, b_flip] -= field
                H[b_flip, b] -= field

    return H


def _ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _circuit_state(params, n, layers):
    """Apply the 3-layer Ry/CZ circuit to |0>^n. params: (layers, n)."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for l in range(layers):
        # Ry on every qubit (in-place via index masks)
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            idx0 = np.array([i for i in range(dim) if (i & mask) == 0])
            idx1 = idx0 | mask
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state

        # CZ on adjacent pairs (not after final layer)
        if l < layers - 1:
            for q in range(n - 1):
                m1 = 1 << q
                m2 = 1 << (q + 1)
                both = np.array([i for i in range(dim) if (i & m1) and (i & m2)])
                state[both] *= -1

    return state


def _energy(params, n, layers, H, E0):
    psi = _circuit_state(params, n, layers)
    e = np.vdot(psi, H @ psi).real
    return e - E0


def _energy_and_grad(params, n, layers, H, E0):
    """Energy (minus E0) and analytic gradient via backpropagation."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    states = [state.copy()]
    ops = []

    for l in range(layers):
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            idx0 = np.array([i for i in range(dim) if (i & mask) == 0])
            idx1 = idx0 | mask
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
            states.append(state.copy())
            ops.append(('ry', l, q))

        if l < layers - 1:
            for q in range(n - 1):
                m1 = 1 << q
                m2 = 1 << (q + 1)
                both = np.array([i for i in range(dim) if (i & m1) and (i & m2)])
                state[both] *= -1
                states.append(state.copy())
                ops.append(('cz', q))

    psi = states[-1]
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real

    # Backward pass
    v = Hpsi.copy()
    grad = np.zeros((layers, n))

    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        psi_before = states[i]

        if op[0] == 'ry':
            l, q = op[1], op[2]
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            # dU/dtheta = 0.5*[[ -s, -c],[ c, -s ]]; (dU/dtheta)^dag = 0.5*[[ -s, c],[-c, -s]]
            mask = 1 << q
            idx0 = np.array([i for i in range(dim) if (i & mask) == 0])
            idx1 = idx0 | mask

            # gradient contribution: 2 Re <psi_before| (dU/dtheta)^dag v |psi_after>
            # (dU/dtheta)^dag v:
            A00, A01 = -0.5 * s, 0.5 * c
            A10, A11 = -0.5 * c, -0.5 * s
            v0 = v[idx0].copy()
            v1 = v[idx1].copy()
            w0 = A00 * v0 + A01 * v1
            w1 = A10 * v0 + A11 * v1
            w = np.zeros(dim, dtype=np.complex128)
            w[idx0] = w0
            w[idx1] = w1
            g = 2.0 * np.vdot(psi_before, w).real
            grad[l, q] = g

            # Propagate v backwards: v_before = U^dag v_after, U^dag = [[c, s],[-s, c]]
            v0 = v[idx0].copy()
            v1 = v[idx1].copy()
            b0 = c * v0 + s * v1
            b1 = -s * v0 + c * v1
            v = np.zeros(dim, dtype=np.complex128)
            v[idx0] = b0
            v[idx1] = b1

        elif op[0] == 'cz':
            q = op[1]
            m1 = 1 << q
            m2 = 1 << (q + 1)
            both = np.array([i for i in range(dim) if (i & m1) and (i & m2)])
            v[both] *= -1

    return energy - E0, grad


def _structured_guesses(n, coupling, field, layers, seed=42):
    """Build a diverse set of initial guesses."""
    rng = np.random.default_rng(seed)
    guesses = []

    # Mean-field single-qubit angle
    n_neighbors = np.zeros(n)
    for i in range(n):
        if i > 0:
            n_neighbors[i] += 1
        if i < n - 1:
            n_neighbors[i] += 1
    n_neighbors[n_neighbors == 0] = 1
    J_eff = coupling * n_neighbors
    h_eff = field
    theta_single = 2.0 * np.arctan2(h_eff, J_eff)

    # Guess 1: mean-field distributed across layers (50/30/20)
    g = np.zeros((layers, n))
    fracs = [0.5, 0.3, 0.2]
    for l in range(layers):
        for q in range(n):
            g[l, q] = fracs[l] * theta_single[q]
    guesses.append(g)

    # Guess 2: all equal base angle
    base = float(np.mean(theta_single))
    guesses.append(np.full((layers, n), base))

    # Guess 3: alternating pattern
    g = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            g[l, q] = base if q % 2 == 0 else -base
    guesses.append(g)

    # Guess 4: all zeros
    guesses.append(np.zeros((layers, n)))

    # Guess 5: all pi/2
    guesses.append(np.full((layers, n), np.pi / 2))

    # Guess 6: uniform small
    guesses.append(np.full((layers, n), 0.3))

    # Random perturbations around mean-field
    for _ in range(6):
        g = g.copy() if False else np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                g[l, q] = fracs[l] * theta_single[q] + rng.normal(0, 0.25)
        guesses.append(g)

    # Fully random starts
    for _ in range(4):
        guesses.append(rng.uniform(-np.pi, np.pi, (layers, n)))

    return guesses


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    E0 = _ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)

    def objective(flat):
        if time.time() > deadline:
            return 1e10
        p = flat.reshape((layers, n))
        try:
            e, _ = _energy_and_grad(p, n, layers, H, E0)
            return e
        except Exception:
            return 1e10

    def grad_func(flat):
        if time.time() > deadline:
            return np.zeros_like(flat)
        p = flat.reshape((layers, n))
        try:
            _, g = _energy_and_grad(p, n, layers, H, E0)
            return g.flatten()
        except Exception:
            return np.zeros_like(flat)

    guesses = _structured_guesses(n, coupling, field, layers)

    best_energy = np.inf
    best_result = guesses[0].flatten().copy()
    for g in guesses:
        v = objective(g.flatten())
        if v < best_energy:
            best_energy = v
            best_result = g.flatten().copy()

    bounds = [(-np.pi, np.pi)] * (layers * n)

    for g in guesses:
        if time.time() > deadline - 20.0:
            break
        x0 = g.flatten()

        # Stage 1: L-BFGS-B
        try:
            res = minimize(objective, x0, jac=grad_func, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 120, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_energy:
                best_energy = res.fun
                best_result = res.x.copy()
        except Exception:
            pass

        # Stage 2: tight L-BFGS-B polish from the best so far
        if time.time() > deadline - 12.0:
            continue
        try:
            res2 = minimize(objective, best_result, jac=grad_func, method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
            if res2.fun < best_energy:
                best_energy = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass

        # Stage 3: Nelder-Mead final polish
        if time.time() > deadline - 6.0:
            continue
        try:
            res3 = minimize(objective, best_result, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14})
            if res3.fun < best_energy:
                best_energy = res3.fun
                best_result = res3.x.copy()
        except Exception:
            pass

    if best_result is None:
        return guesses[0]

    return best_result.reshape((layers, n))
