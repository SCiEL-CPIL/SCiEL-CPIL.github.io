"""Model-and-optimize: Fast bit-manipulation circuit with physics-informed optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_TIME_LIMIT = 148.0  # Tight safety margin before 150s kill
_cache = {}


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix using bit manipulation."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bi = (idx >> i) & 1
            bp = (idx >> (i + 1)) & 1
            zi = 1.0 if bi == 0 else -1.0
            zp = 1.0 if bp == 0 else -1.0
            h_diag += -coupling * zi * zp
        H[idx, idx] = h_diag
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _get_ground_energy(n, coupling, field):
    """Compute the exact ground energy only."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _circuit_forward_fast(params, n, layers):
    """
    Fast forward pass using bit-manipulation instead of moveaxis.
    State is stored as complex array indexed by bitstring.
    Qubit i corresponds to bit i (LSB = qubit 0).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n

    thetas = params.reshape(layers, n)

    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            if abs(theta) < 1e-15:
                continue
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # For each basis state, if bit q is 0: new[0] = c*old[0] - s*old[1]
            # if bit q is 1: new[1] = s*old[0] + c*old[1]
            # We need to pair states that differ only in bit q
            bit_mask = 1 << q
            new_state = np.empty_like(state)
            for idx in range(dim):
                if (idx & bit_mask) == 0:
                    partner = idx | bit_mask
                    old_0 = state[idx]
                    old_1 = state[partner]
                    new_state[idx] = c * old_0 - s * old_1
                    new_state[partner] = s * old_0 + c * old_1
            state = new_state

        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                bit_q = 1 << q
                bit_qp1 = 1 << (q + 1)
                # CZ: |11> -> -|11>
                # Find all states where both bit q and bit q+1 are 1
                for idx in range(dim):
                    if (idx & bit_q) and (idx & bit_qp1):
                        state[idx] *= -1.0

    return state


def _energy_and_grad_fast(params, n, coupling, field, layers, H, e0):
    """
    Compute energy and gradient using adjoint method with fast circuit simulation.
    Stores states before each Ry gate for backward pass.
    """
    dim = 1 << n
    thetas = params.reshape(layers, n)

    # Forward pass, storing states before each Ry
    states_before_ry = []
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for layer in range(layers):
        for q in range(n):
            states_before_ry.append(state.copy())
            theta = thetas[layer, q]
            if abs(theta) < 1e-15:
                continue
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            bit_mask = 1 << q
            new_state = np.empty_like(state)
            for idx in range(dim):
                if (idx & bit_mask) == 0:
                    partner = idx | bit_mask
                    old_0 = state[idx]
                    old_1 = state[partner]
                    new_state[idx] = c * old_0 - s * old_1
                    new_state[partner] = s * old_0 + c * old_1
            state = new_state

        if layer < layers - 1:
            for q in range(n - 1):
                bit_q = 1 << q
                bit_qp1 = 1 << (q + 1)
                for idx in range(dim):
                    if (idx & bit_q) and (idx & bit_qp1):
                        state[idx] *= -1.0

    psi_c = state
    Hpsi = H @ psi_c
    energy = np.vdot(psi_c, Hpsi).real
    obj_val = (energy - e0) / n

    # Backward pass
    grad = np.zeros((layers, n), dtype=np.float64)
    adjoint = Hpsi.copy()
    gate_idx = len(states_before_ry) - 1

    for layer in range(layers - 1, -1, -1):
        # Undo CZ gates (CZ is self-adjoint)
        if layer < layers - 1:
            for q in range(n - 2, -1, -1):
                bit_q = 1 << q
                bit_qp1 = 1 << (q + 1)
                for idx in range(dim):
                    if (idx & bit_q) and (idx & bit_qp1):
                        adjoint[idx] *= -1.0

        # Undo Ry gates in reverse order
        for q in range(n - 1, -1, -1):
            theta = thetas[layer, q]
            psi_b = states_before_ry[gate_idx]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            bit_mask = 1 << q

            # Compute gradient: 2 * Re(<adjoint | dRy/dtheta | psi_b>)
            # dRy/dtheta = 0.5 * [[-s, -c], [c, -s]]
            # For each pair of states differing in bit q:
            # dpsi_0/dtheta = 0.5 * (-s * psi_b_0 - c * psi_b_1)
            # dpsi_1/dtheta = 0.5 * (c * psi_b_0 - s * psi_b_1)
            grad_sum = 0.0
            for idx in range(dim):
                if (idx & bit_mask) == 0:
                    partner = idx | bit_mask
                    adj_0 = adjoint[idx]
                    adj_1 = adjoint[partner]
                    psi_0 = psi_b[idx]
                    psi_1 = psi_b[partner]

                    dpsi_0 = 0.5 * (-s * psi_0 - c * psi_1)
                    dpsi_1 = 0.5 * (c * psi_0 - s * psi_1)

                    # <adj | dpsi> = conj(adj_0)*dpsi_0 + conj(adj_1)*dpsi_1
                    inner = np.conj(adj_0) * dpsi_0 + np.conj(adj_1) * dpsi_1
                    grad_sum += inner.real

            grad[layer, q] = 2.0 * grad_sum

            # Propagate adjoint backward: adjoint = Ry^dag @ adjoint
            # Ry^dag = [[c, s], [-s, c]]
            new_adj = np.empty_like(adjoint)
            for idx in range(dim):
                if (idx & bit_mask) == 0:
                    partner = idx | bit_mask
                    adj_0 = adjoint[idx]
                    adj_1 = adjoint[partner]
                    new_adj[idx] = c * adj_0 + s * adj_1
                    new_adj[partner] = -s * adj_0 + c * adj_1
            adjoint = new_adj
            gate_idx -= 1

    grad_flat = (grad / n).flatten()
    return obj_val, grad_flat


def _physics_informed_init(n, coupling, field, layers):
    """
    Generate physics-informed initial parameters.
    For transverse-field Ising model:
    - Strong field: qubits tend to be in |-> (X-basis), so Ry(pi/2) creates |->
    - Strong coupling: favors aligned Z, so smaller angles
    """
    inits = []

    # Base angle from field strength (heuristic)
    # field=0.45: weak field, mostly |0> -> small angle
    # field=1.0: moderate -> ~pi/4
    # field=1.65: strong field -> ~pi/2
    base_angle = min(np.pi / 2, field * 0.8)

    # Pattern 1: All layers same base angle
    x = np.full((layers, n), base_angle)
    inits.append(x.flatten())

    # Pattern 2: Alternating signs (staggered)
    x = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            sign = 1 if (q % 2 == 0) else -1
            x[l, q] = sign * base_angle
    inits.append(x.flatten())

    # Pattern 3: Layer-dependent (deeper layers larger angle)
    x = np.zeros((layers, n))
    for l in range(layers):
        x[l, :] = base_angle * (l + 1) / layers
    inits.append(x.flatten())

    # Pattern 4: Middle layer larger
    x = np.full((layers, n), base_angle * 0.5)
    x[1, :] = base_angle
    inits.append(x.flatten())

    # Pattern 5: Near pi/2 for strong field
    if field > 1.2:
        x = np.full((layers, n), np.pi / 2 - 0.1)
        inits.append(x.flatten())

    # Pattern 6: Small angles for weak field
    if field < 0.6:
        x = np.full((layers, n), 0.2)
        inits.append(x.flatten())

    # Random perturbations of base
    np.random.seed(42)
    for _ in range(8):
        x = base_angle + np.random.randn(layers, n) * 0.3
        x = np.clip(x, -np.pi, np.pi)
        inits.append(x.flatten())

    return inits


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    e0 = _get_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    np_params = layers * n

    def obj_only(x):
        return _energy_and_grad_fast(x, n, coupling, field, layers, H, e0)[0]

    def jac_only(x):
        return _energy_and_grad_fast(x, n, coupling, field, layers, H, e0)[1]

    # Generate physics-informed restarts
    restarts = _physics_informed_init(n, coupling, field, layers)

    best_val = np.inf
    best_x = np.zeros(np_params)

    # Evaluate all restarts first to find best starting point
    for x0 in restarts:
        if time.time() - _start_time > _TIME_LIMIT - 15:
            break
        try:
            val = obj_only(x0)
            if val < best_val:
                best_val = val
                best_x = x0.copy()
        except Exception:
            pass

    # L-BFGS-B from best restart
    if time.time() - _start_time < _TIME_LIMIT - 10:
        try:
            res = opt.minimize(
                obj_only, best_x, jac=jac_only,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Additional restarts from perturbed best solution if time allows
    if time.time() - _start_time < _TIME_LIMIT - 20:
        np.random.seed(99)
        for _ in range(5):
            if time.time() - _start_time > _TIME_LIMIT - 15:
                break
            x0 = best_x + np.random.randn(np_params) * 0.2
            x0 = np.clip(x0, -np.pi, np.pi)
            try:
                res = opt.minimize(
                    obj_only, x0, jac=jac_only,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

    # Final polish
    if time.time() - _start_time < _TIME_LIMIT - 5:
        try:
            res2 = opt.minimize(
                obj_only, best_x, jac=jac_only,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 50}
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
