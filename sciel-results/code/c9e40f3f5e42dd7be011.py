"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_idx_cache = {}
_psi0_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms: -field * sum X_i
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H


def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = float(eigvals[0])
    _e0_cache[key] = e0
    return e0


def _get_indices(n):
    """Pre-compute index arrays for each qubit's 0 and 1 subspaces, and CZ masks."""
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    qubit_indices = []
    for q in range(n):
        mask = 1 << q
        idx_0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
        idx_1 = idx_0 + mask
        qubit_indices.append((idx_0, idx_1))
    # Pre-compute CZ masks for adjacent pairs
    cz_masks = []
    all_idx = np.arange(dim)
    for i in range(n - 1):
        m1 = 1 << i
        m2 = 1 << (i + 1)
        both_set = ((all_idx & m1) != 0) & ((all_idx & m2) != 0)
        cz_masks.append(both_set)
    _idx_cache[n] = (qubit_indices, cz_masks)
    return _idx_cache[n]


def _get_psi0(n):
    """Get the |0...0> state vector."""
    if n not in _psi0_cache:
        dim = 1 << n
        psi0 = np.zeros(dim, dtype=np.float64)
        psi0[0] = 1.0
        _psi0_cache[n] = psi0
    return _psi0_cache[n]


def _apply_ry_vec(state, q, theta, qubit_indices):
    """Apply Ry(theta) to qubit q in-place on state vector."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    idx_0, idx_1 = qubit_indices[q]
    v0 = state[idx_0].copy()
    v1 = state[idx_1].copy()
    state[idx_0] = c * v0 - s * v1
    state[idx_1] = s * v0 + c * v1


def _apply_cz_vec(state, cz_masks):
    """Apply all CZ gates in-place."""
    for mask in cz_masks:
        state[mask] *= -1.0


def _apply_cz_inv_vec(state, cz_masks):
    """Apply inverse of all CZ gates (same as CZ since self-inverse)."""
    for mask in cz_masks:
        state[mask] *= -1.0


def _apply_ry_inv_vec(state, q, theta, qubit_indices):
    """Apply Ry(theta)^dagger to qubit q in-place."""
    # Ry(theta)^dagger = Ry(-theta)
    _apply_ry_vec(state, q, -theta, qubit_indices)


def _circuit_state_fwd(n, thetas, layers, start_state=None):
    """Forward circuit application. thetas: (layers, n)."""
    dim = 1 << n
    qubit_indices, cz_masks = _get_indices(n)
    if start_state is None:
        state = _get_psi0(n).copy()
    else:
        state = start_state.copy()
    for layer in range(layers):
        for q in range(n):
            _apply_ry_vec(state, q, thetas[layer, q], qubit_indices)
        if layer < layers - 1:
            _apply_cz_vec(state, cz_masks)
    return state


def _circuit_state_bwd(n, thetas, layers, target_state):
    """
    Backward circuit: apply inverse of full circuit to target_state.
    This gives us the state that, when passed through the forward circuit, yields target_state.
    thetas: (layers, n).
    """
    dim = 1 << n
    qubit_indices, cz_masks = _get_indices(n)
    state = target_state.copy()
    for layer in range(layers - 1, -1, -1):
        if layer < layers - 1:
            _apply_cz_inv_vec(state, cz_masks)
        for q in range(n):
            _apply_ry_inv_vec(state, q, thetas[layer, q], qubit_indices)
    return state


def _circuit_state(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    return _circuit_state_fwd(n, thetas, layers)


def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fwd(n, thetas, layers)
    return float(np.dot(psi, H @ psi))


def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n


def _layer_wise_projection(n, coupling, field, layers, initial_thetas, num_iters=10):
    """
    Layer-wise back-to-front exact state projection.
    For each layer l (back to front), fix subsequent layers and solve for
    current layer's Ry angles by minimizing distance to target state
    transformed by inverse of subsequent circuit.
    """
    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_H(n, coupling, field)
    qubit_indices, cz_masks = _get_indices(n)
    dim = 1 << n

    # Get exact ground state
    eigvals, eigvecs = np.linalg.eigh(H)
    psi_target = eigvecs[:, 0]
    # Ensure real (should be real for this Hamiltonian)
    psi_target = np.real(psi_target)
    psi_target /= np.linalg.norm(psi_target)

    thetas = initial_thetas.copy()

    for iteration in range(num_iters):
        if time.time() > _deadline:
            break

        # Sweep layers back to front
        for l in range(layers - 1, -1, -1):
            if time.time() > _deadline:
                break

            # Build the "effective target" for layer l:
            # Apply inverse of layers l+1..end to psi_target
            # Circuit: Ry(θ0) -> CZ -> Ry(θ1) -> CZ -> Ry(θ2)
            # For layer l, the subsequent circuit is:
            #   (if l < layers-1) CZ then Ry(θ_{l+1}) ... CZ ... Ry(θ_{layers-1})
            # Inverse: Ry(θ_{layers-1})^dag ... CZ^dag ... Ry(θ_{l+1})^dag ... CZ^dag

            # Compute phi = (subsequent_circuit)^dag @ psi_target
            phi = psi_target.copy()
            for sl in range(layers - 1, l, -1):
                # Apply Ry(θ_sl)^dag to each qubit
                for q in range(n):
                    _apply_ry_inv_vec(phi, q, thetas[sl, q], qubit_indices)
                if sl > l:
                    # Apply CZ^dag (same as CZ)
                    _apply_cz_inv_vec(phi, cz_masks)
                # Note: CZ is between layers, so after Ry(sl) there's a CZ if sl < layers-1
                # Let me re-think the circuit structure...

            # Actually let me redo this more carefully.
            # Forward: |0> -> Ry(θ0) -> CZ -> Ry(θ1) -> CZ -> Ry(θ2)
            # So the full operator is: Ry(θ2) @ CZ @ Ry(θ1) @ CZ @ Ry(θ0)
            # For layer l, subsequent layers l+1..L-1 form:
            #   Ry(θ_{L-1}) @ CZ @ ... @ Ry(θ_{l+1}) @ CZ
            # (with CZ after each Ry except the last layer)
            # Inverse: CZ^dag @ Ry(θ_{l+1})^dag @ ... @ CZ^dag @ Ry(θ_{L-1})^dag
            # Since CZ is self-inverse:
            #   CZ @ Ry(θ_{l+1})^dag @ ... @ CZ @ Ry(θ_{L-1})^dag

            # Let me just recompute phi properly
            phi = psi_target.copy()
            for sl in range(layers - 1, l, -1):
                # First undo Ry(θ_sl)
                for q in range(n):
                    _apply_ry_inv_vec(phi, q, thetas[sl, q], qubit_indices)
                # Then undo CZ (if it exists between sl-1 and sl)
                # CZ is applied AFTER Ry of layer sl-1, BEFORE Ry of layer sl
                # So between layer l and l+1, there's a CZ
                # When going backward from layer sl, we undo CZ that was before Ry(sl)
                if sl > l:
                    _apply_cz_inv_vec(phi, cz_masks)

            # Now we need to find θ_l such that Ry(θ_l) @ |0> approximates phi
            # Ry(θ_l) @ |0> = prod_q Ry_q(θ_l[q]) |0>
            # This is a product state: ⊗_q [cos(θ_q/2)|0> + sin(θ_q/2)|1>]
            # We want to minimize ||prod_q Ry_q(θ_q)|0> - phi||^2
            # = 1 - 2<phi|prod_q Ry_q(θ_q)|0> + ||prod_q Ry_q(θ_q)|0>||^2
            # = 1 - 2 * Re(<phi|prod_q Ry_q(θ_q)|0>)  (since both are normalized)

            # The overlap <phi|prod_q Ry_q(θ_q)|0> can be computed as:
            # For each qubit q, the local state is [cos(θ_q/2), sin(θ_q/2)]
            # The product state amplitude for basis state |b_0 b_1 ... b_{n-1}> is
            #   prod_q [cos(θ_q/2) if b_q=0 else sin(θ_q/2)]
            # So <phi|product> = sum_b phi[b] * prod_q f_q(b_q)
            # where f_q(0) = cos(θ_q/2), f_q(1) = sin(θ_q/2)

            # This is a non-trivial optimization over n angles.
            # Use scipy minimize on just the n angles of this layer.

            def layer_obj(angles_l):
                # Build product state
                prod_state = np.ones(dim, dtype=np.float64)
                for q in range(n):
                    c = np.cos(angles_l[q] / 2.0)
                    s = np.sin(angles_l[q] / 2.0)
                    idx_0, idx_1 = qubit_indices[q]
                    prod_state[idx_0] *= c
                    prod_state[idx_1] *= s
                # Overlap
                overlap = np.dot(phi, prod_state)
                # Energy of this product state
                # Actually we want to minimize energy, not just distance
                # But minimizing distance to ground state is a good proxy
                # Let's minimize the energy directly for better results
                return float(np.dot(prod_state, H @ prod_state))

            def layer_obj_dist(angles_l):
                prod_state = np.ones(dim, dtype=np.float64)
                for q in range(n):
                    c = np.cos(angles_l[q] / 2.0)
                    s = np.sin(angles_l[q] / 2.0)
                    idx_0, idx_1 = qubit_indices[q]
                    prod_state[idx_0] *= c
                    prod_state[idx_1] *= s
                overlap = np.dot(phi, prod_state)
                return -(overlap)  # maximize overlap = minimize negative overlap

            # Try both: energy-based and distance-based
            best_layer_angles = thetas[l].copy()
            best_layer_val = float('inf')

            # Energy-based optimization
            try:
                res = minimize(layer_obj, thetas[l].copy(), method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_layer_val:
                    best_layer_val = res.fun
                    best_layer_angles = res.x.copy()
            except Exception:
                pass

            # Distance-based optimization (maximize overlap with projected target)
            try:
                res = minimize(layer_obj_dist, thetas[l].copy(), method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                # Check energy of this solution
                prod_state = np.ones(dim, dtype=np.float64)
                for q in range(n):
                    c = np.cos(res.x[q] / 2.0)
                    s = np.sin(res.x[q] / 2.0)
                    idx_0, idx_1 = qubit_indices[q]
                    prod_state[idx_0] *= c
                    prod_state[idx_1] *= s
                e = float(np.dot(prod_state, H @ prod_state))
                if e < best_layer_val:
                    best_layer_val = e
                    best_layer_angles = res.x.copy()
            except Exception:
                pass

            thetas[l] = best_layer_angles

        # Full circuit energy check
        full_state = _circuit_state_fwd(n, thetas, layers)
        full_energy = float(np.dot(full_state, H @ full_state))
        if full_energy < e0 + 1e-10:
            break

    return thetas


def angles(n, coupling, field, layers):
    global _deadline
    start_time = time.time()
    _deadline = start_time + 115.0  # Safety margin before 120s limit

    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_H(n, coupling, field)
    qubit_indices, cz_masks = _get_indices(n)
    dim = 1 << n

    # Get exact ground state for projection
    eigvals, eigvecs = np.linalg.eigh(H)
    psi_target = np.real(eigvecs[:, 0])
    psi_target /= np.linalg.norm(psi_target)

    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)

    # Strategy 1: Layer-wise projection from various initializations
    init_list = []
    # All zeros
    init_list.append(np.zeros(layers * n))
    # All pi/2
    init_list.append(np.full(layers * n, np.pi / 2))
    # All -pi/2
    init_list.append(np.full(layers * n, -np.pi / 2))
    # Alternating
    init_list.append(np.array([np.pi / 2 if (j % 2 == 0) else -np.pi / 2 for j in range(layers * n)]))
    # First layer pi/2
    t = np.zeros(layers * n)
    t[:n] = np.pi / 2
    init_list.append(t)
    # Last layer pi/2
    t = np.zeros(layers * n)
    t[-n:] = np.pi / 2
    init_list.append(t)
    # Small angles
    init_list.append(np.full(layers * n, 0.1))
    # pi/3
    init_list.append(np.full(layers * n, np.pi / 3))

    # Add random inits
    for seed in [42, 123, 456, 789, 1000, 2000]:
        np.random.seed(seed)
        init_list.append(np.random.uniform(-np.pi, np.pi, size=(layers * n)))

    # Run layer-wise projection from each init
    for init in init_list:
        if time.time() > _deadline:
            break
        try:
            thetas_proj = _layer_wise_projection(n, coupling, field, layers, init, num_iters=5)
            val = _objective(thetas_proj.flatten(), n, coupling, field, layers, e0)
            if val < best_val:
                best_val = val
                best_x = thetas_proj.flatten().copy()
        except Exception:
            pass

    # Strategy 2: Standard multi-start L-BFGS-B
    remaining_restarts = 8
    for i in range(remaining_restarts):
        if time.time() > _deadline:
            break
        if i < len(init_list):
            x_init = init_list[i]
        else:
            np.random.seed(5000 + i)
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))

        def obj(p):
            return _objective(p, n, coupling, field, layers, e0)

        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final polish with tighter tolerances if time remains
    if time.time() < _deadline - 15.0:
        try:
            def obj(p):
                return _objective(p, n, coupling, field, layers, e0)
            res = minimize(obj, best_x, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Second polish with Nelder-Mead if time remains
    if time.time() < _deadline - 8.0:
        try:
            def obj(p):
                return _objective(p, n, coupling, field, layers, e0)
            res = minimize(obj, best_x, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final layer-wise projection polish on best result
    if time.time() < _deadline - 5.0:
        try:
            thetas_best = best_x.reshape((layers, n))
            thetas_polished = _layer_wise_projection(n, coupling, field, layers, thetas_best, num_iters=3)
            val = _objective(thetas_polished.flatten(), n, coupling, field, layers, e0)
            if val < best_val:
                best_val = val
                best_x = thetas_polished.flatten().copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
