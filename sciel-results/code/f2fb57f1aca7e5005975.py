"""Rewritten program: faster vectorized circuit + more restarts within time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _ry_matrix(theta):
    """Return 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _ry_derivative_matrix(theta):
    """Return 2x2 derivative of Ry rotation matrix w.r.t theta."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return 0.5 * np.array([[-s, -c], [c, -s]], dtype=np.complex128)


def _apply_ry_layer(state, thetas, n, derivative_index=-1):
    """
    Apply Ry(theta_q) to each qubit q in one layer.
    thetas: shape (n,)
    If derivative_index >= 0 (global param index within this layer, i.e. qubit),
    the gate at that qubit is replaced by its derivative.
    Returns new state (2^n,).
    """
    shape = [2] * n
    state_r = state.reshape(shape)

    # We'll build the output tensor by applying each qubit's gate.
    # To vectorize across qubits, we process each qubit but use broadcasting
    # over the other axes. For n<=6 this is fine.
    out = state_r
    for q in range(n):
        theta = thetas[q]
        if derivative_index == q:
            gate = _ry_derivative_matrix(theta)
        else:
            if abs(theta) < 1e-15:
                continue
            gate = _ry_matrix(theta)
        # Move qubit axis to front
        moved = np.moveaxis(out, q, 0)
        new = np.tensordot(gate, moved, axes=([1], [0]))
        out = np.moveaxis(new, 0, q)
    return out.reshape(-1)


def _apply_cz_layer(state, n):
    """Apply CZ on every adjacent pair (q, q+1) in ascending order."""
    shape = [2] * n
    state_r = state.reshape(shape)
    out = state_r
    for q in range(n - 1):
        moved = np.moveaxis(out, [q, q + 1], [-2, -1])
        # factor: -1 on |11>
        factor = np.ones(moved.shape, dtype=np.complex128)
        idx11 = (slice(None),) * (n - 2) + (1, 1)
        factor[idx11] = -1.0
        moved = moved * factor
        out = np.moveaxis(moved, [-2, -1], [q, q + 1])
    return out.reshape(-1)


def _circuit_state(params, n, layers, derivative_index=-1):
    """
    Compute the state vector produced by the circuit.
    params: flattened (layers*n,)
    derivative_index: global index of the Ry gate to replace with derivative, or -1.
    """
    thetas = params.reshape(layers, n)
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0  # |0>^n

    for layer in range(layers):
        # Determine which qubit (if any) in this layer is the derivative
        local_deriv = -1
        if derivative_index >= 0:
            g = derivative_index
            l = g // n
            q = g % n
            if l == layer:
                local_deriv = q
        state = _apply_ry_layer(state, thetas[layer], n, derivative_index=local_deriv)
        if layer < layers - 1:
            state = _apply_cz_layer(state, n)

    return state


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Diagonal: Z_i Z_{i+1}
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag

    # Off-diagonal: X_i
    for i in range(n):
        for idx in range(dim):
            other = idx ^ (1 << i)
            H[idx, other] += -field

    return H


def _exact_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _energy_and_grad(params, n, coupling, field, layers, H, e0):
    """Compute (E - E0)/n and its gradient."""
    psi = _circuit_state(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n

    num_params = params.size
    grad = np.zeros(num_params, dtype=np.float64)
    for k in range(num_params):
        psi_k = _circuit_state(params, n, layers, derivative_index=k)
        val = np.vdot(psi_k, Hpsi)
        grad[k] = 2.0 * val.real / n

    return obj_val, grad


def _initial_guesses(n, coupling, field, layers):
    """Generate a list of initial parameter arrays."""
    guesses = []
    # 1. All zeros (|0>^n)
    guesses.append(np.zeros(layers * n))
    # 2. All pi/2 (|+>^n)
    guesses.append(np.full(layers * n, np.pi / 2))
    # 3. All pi/4
    guesses.append(np.full(layers * n, np.pi / 4))
    # 4. Physics-informed: small field -> closer to |0>, large field -> closer to |+>
    # Interpolate theta per layer
    t_small = 0.0
    t_large = np.pi / 2
    # field range roughly 0.3..1.65; map to 0..1
    f = field
    frac = np.clip((f - 0.3) / (1.65 - 0.3), 0.0, 1.0)
    t_phys = t_small + frac * (t_large - t_small)
    guesses.append(np.full(layers * n, t_phys))
    # 5. Alternating pattern for antiferro-like
    alt = np.zeros(layers * n)
    for l in range(layers):
        for q in range(n):
            alt[l * n + q] = (np.pi / 4) if (q % 2 == 0) else (-np.pi / 4)
    guesses.append(alt)
    # 6-... Random restarts
    np.random.seed(12345)
    n_random = 8
    for _ in range(n_random):
        g = np.random.randn(layers * n) * 0.7
        guesses.append(g)
    # 7. Small perturbation around physics guess
    g = np.full(layers * n, t_phys) + np.random.randn(layers * n) * 0.1
    guesses.append(g)
    return guesses


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)

    def objective_and_grad(params):
        return _energy_and_grad(params, n, coupling, field, layers, H, e0)

    def objective_only(params):
        val, _ = _energy_and_grad(params, n, coupling, field, layers, H, e0)
        return val

    def jac(params):
        _, g = _energy_and_grad(params, n, coupling, field, layers, H, e0)
        return g

    guesses = _initial_guesses(n, coupling, field, layers)

    best_result = None
    best_val = np.inf

    # Adaptive time allocation: spend more time on earlier (better) guesses
    # but ensure at least a quick pass over all.
    n_guesses = len(guesses)
    # Reserve 15s for finalization/safety
    available = max(1.0, _TIME_LIMIT - 15.0 - (time.time() - _start_time))
    # Weighted allocation: first few guesses get more
    weights = []
    for i in range(n_guesses):
        if i < 3:
            weights.append(2.0)
        elif i < 6:
            weights.append(1.5)
        else:
            weights.append(1.0)
    total_w = sum(weights)
    time_per = [available * w / total_w for w in weights]

    for i, x_init in enumerate(guesses):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        x_init = np.clip(x_init, -np.pi, np.pi)
        t_start = time.time()
        # Max iterations scaled by allocated time (rough heuristic: ~0.5s per iter for n=6)
        max_iter = max(50, int(time_per[i] / 0.3))
        max_iter = min(max_iter, 400)
        try:
            res = opt.minimize(
                objective_only,
                x_init,
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 20}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
        except Exception:
            pass

    if best_result is None:
        best_result = np.zeros(layers * n)

    theta = best_result.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: |+> state (last layer pi/2)
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
