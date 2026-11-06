"""Structural change: direct vectorized energy evaluation with finite-difference-free
analytic gradients via automatic differentiation of the circuit, plus a
multi-start BFGS with physics-informed initializations and adaptive time budgeting."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 145.0


def _circuit_state_and_grads(params, n, layers):
    """
    Compute the state vector AND the derivative of the state w.r.t. each parameter.
    params: shape (layers, n)
    Returns:
        psi: shape (2^n,)
        grads: list of length layers*n, each shape (2^n,)
    """
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0

    # Store intermediate states after each layer (before CZ)
    # and the derivatives of psi w.r.t. each parameter
    # We'll use a forward-mode approach: for each parameter, track how the state changes.
    # But that's O(layers*n) state vectors. For n=6, dim=64, layers=3, n=6: 18 vectors of size 64.
    # That's fine.

    # Actually, let's do it differently. We'll compute psi and all d_psi/d_theta_k
    # by propagating both the state and its derivatives through the circuit.

    # State: psi
    # Derivatives: list of d_psi/d_theta_{l,q} for each layer l and qubit q

    # We'll store derivatives as a 2D array: shape (layers*n, dim)
    num_params = layers * n
    d_psi = np.zeros((num_params, dim), dtype=np.complex128)

    shape = [2] * n

    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            # Ry = [[c, -s], [s, c]]
            # dRy/dtheta = 0.5 * [[-s, -c], [c, -s]]

            # Apply to psi: for each basis state, update the q-th qubit
            psi_r = psi.reshape(shape)
            d_psi_r = d_psi.reshape(num_params, *[2] * n)

            # Move qubit q to axis 0
            psi_m = np.moveaxis(psi_r, q, 0)
            # New psi: new[0] = c*old[0] - s*old[1], new[1] = s*old[0] + c*old[1]
            psi_new = np.empty_like(psi_m)
            psi_new[0] = c * psi_m[0] - s * psi_m[1]
            psi_new[1] = s * psi_m[0] + c * psi_m[1]
            psi = np.moveaxis(psi_new, 0, q).reshape(dim)

            # For derivatives: d(psi)/dtheta_k for k != (layer,q) just gets the Ry applied
            # d(psi)/dtheta_{layer,q} gets the dRy applied to the pre-Ry state
            # But we need to be careful: the derivative of psi w.r.t. theta_{layer,q}
            # at this point is: dRy/dtheta * (pre-Ry state) + Ry * (d(pre-Ry state)/dtheta_{layer,q})
            # The second term is zero for this parameter (it's the first time it's applied)
            # but non-zero for other parameters that were applied in previous layers.

            # Let's handle this more carefully.
            # Before applying Ry_{l,q}, we have psi and all d_psi/dtheta_k.
            # After applying:
            #   psi_new = Ry * psi_old
            #   d_psi_new/dtheta_k = Ry * d_psi_old/dtheta_k  for k != (l,q)
            #   d_psi_new/dtheta_{l,q} = dRy/dtheta * psi_old + Ry * d_psi_old/dtheta_{l,q}
            #   But d_psi_old/dtheta_{l,q} = 0 (first time this parameter is used)
            #   So d_psi_new/dtheta_{l,q} = dRy/dtheta * psi_old

            # Apply Ry to all existing derivatives
            for k in range(num_params):
                if k == layer * n + q:
                    continue
                dpsi_m = np.moveaxis(d_psi_r[k], q, 0)
                dpsi_new = np.empty_like(dpsi_m)
                dpsi_new[0] = c * dpsi_m[0] - s * dpsi_m[1]
                dpsi_new[1] = s * dpsi_m[0] + c * dpsi_m[1]
                d_psi_r[k] = np.moveaxis(dpsi_new, 0, q)

            # For the current parameter, d_psi/dtheta = dRy/dtheta * psi_old
            # But we already overwrote psi... we need psi_old.
            # Let me restructure: compute psi_old first, then update.

            # Actually, let me redo this. I'll keep a copy of the pre-Ry state.
            # This is getting complex. Let me use a simpler approach:
            # Compute the full circuit state, then use finite differences for gradients.
            # For n=6, dim=64, and 18 params, that's 19 circuit evaluations per objective call.
            # Each circuit eval is O(n * dim) ~ 6*64 = 384 ops. Total: 19*384 ~ 7300 ops.
            # With BFGS doing ~100 iterations, that's 730K ops. Fast enough.

            # Let me restart with a cleaner approach.
            break
        break

    # Clean approach: just compute psi, then finite-difference gradients
    # But that's slow. Let me do the analytic gradient properly.

    # Actually, let me just use a clean implementation:
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    d_psi = np.zeros((num_params, dim), dtype=np.complex128)

    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            psi_r = psi.reshape(shape)
            psi_m = np.moveaxis(psi_r, q, 0)

            # Save old state for derivative computation
            psi_old = psi_m.copy()

            # Update psi
            psi_new = np.empty_like(psi_m)
            psi_new[0] = c * psi_m[0] - s * psi_m[1]
            psi_new[1] = s * psi_m[0] + c * psi_m[1]
            psi = np.moveaxis(psi_new, 0, q).reshape(dim)

            # Update derivatives for all other params
            for k in range(num_params):
                if k == layer * n + q:
                    continue
                dpsi_r = d_psi[k].reshape(shape)
                dpsi_m = np.moveaxis(dpsi_r, q, 0)
                dpsi_new = np.empty_like(dpsi_m)
                dpsi_new[0] = c * dpsi_m[0] - s * dpsi_m[1]
                dpsi_new[1] = s * dpsi_m[0] + c * dpsi_m[1]
                d_psi[k] = np.moveaxis(dpsi_new, 0, q).reshape(dim)

            # For current param: d_psi/dtheta = dRy/dtheta * psi_old
            # dRy/dtheta = 0.5 * [[-s, -c], [c, -s]]
            k_cur = layer * n + q
            dpsi_cur = np.empty_like(psi_old)
            dpsi_cur[0] = 0.5 * (-s * psi_old[0] - c * psi_old[1])
            dpsi_cur[1] = 0.5 * (c * psi_old[0] - s * psi_old[1])
            d_psi[k_cur] = np.moveaxis(dpsi_cur, 0, q).reshape(dim)

        # Apply CZ layer (if not last)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ: |11> -> -|11>
                # Apply to psi
                psi_r = psi.reshape(shape)
                psi_m = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
                psi_m[1, 1] *= -1.0
                psi = np.moveaxis(psi_m, [-2, -1], [q, q + 1]).reshape(dim)

                # Apply to all derivatives
                for k in range(num_params):
                    dpsi_r = d_psi[k].reshape(shape)
                    dpsi_m = np.moveaxis(dpsi_r, [q, q + 1], [-2, -1])
                    dpsi_m[1, 1] *= -1.0
                    d_psi[k] = np.moveaxis(dpsi_m, [-2, -1], [q, q + 1]).reshape(dim)

    return psi, d_psi


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)

    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag

    for i in range(n):
        for idx in range(dim):
            other = idx ^ (1 << i)
            H[idx, other] += -field

    return H


def _exact_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _objective_and_grad(params, n, coupling, field, layers, H, e0):
    """Compute (E - E0)/n and analytic gradient."""
    psi, d_psi = _circuit_state_and_grads(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n

    num_params = params.size
    grad = np.zeros(num_params, dtype=np.float64)
    for k in range(num_params):
        val = np.vdot(d_psi[k], Hpsi)
        grad[k] = 2.0 * val.real / n

    return obj_val, grad


def _initial_guesses(n, coupling, field, layers):
    guesses = []
    # Physics-informed: interpolate between |0> (theta=0) and |+> (theta=pi/2)
    frac = np.clip((field - 0.3) / (1.65 - 0.3), 0.0, 1.0)
    t_phys = frac * np.pi / 2

    guesses.append(np.zeros(layers * n))
    guesses.append(np.full(layers * n, np.pi / 2))
    guesses.append(np.full(layers * n, t_phys))
    guesses.append(np.full(layers * n, t_phys * 0.5))
    guesses.append(np.full(layers * n, t_phys * 1.5))

    # Alternating
    alt = np.zeros(layers * n)
    for l in range(layers):
        for q in range(n):
            alt[l * n + q] = (np.pi / 4) if (q % 2 == 0) else (-np.pi / 4)
    guesses.append(alt)

    # Random restarts with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(12):
        g = rng.randn(layers * n) * 0.8
        guesses.append(g)

    # Small perturbations around physics guess
    for scale in [0.05, 0.1, 0.2]:
        g = np.full(layers * n, t_phys) + rng.randn(layers * n) * scale
        guesses.append(g)

    return guesses


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)

    def objective(params):
        val, _ = _objective_and_grad(params, n, coupling, field, layers, H, e0)
        return val

    def jac(params):
        _, g = _objective_and_grad(params, n, coupling, field, layers, H, e0)
        return g

    guesses = _initial_guesses(n, coupling, field, layers)

    best_val = np.inf
    best_result = None

    elapsed = time.time() - _start_time
    available = max(1.0, _TIME_LIMIT - 10.0 - elapsed)

    n_guesses = len(guesses)
    # Allocate time: more to first few
    weights = [2.0 if i < 4 else (1.5 if i < 8 else 1.0) for i in range(n_guesses)]
    total_w = sum(weights)
    time_per = [available * w / total_w for w in weights]

    for i, x_init in enumerate(guesses):
        if time.time() - _start_time > _TIME_LIMIT - 8:
            break

        x_init = np.clip(x_init, -np.pi, np.pi)
        t_start = time.time()
        max_iter = max(30, int(time_per[i] / 0.15))
        max_iter = min(max_iter, 300)

        try:
            res = opt.minimize(
                objective,
                x_init,
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
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
