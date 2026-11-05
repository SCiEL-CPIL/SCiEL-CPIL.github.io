"""Shallow-circuit Ising ground states - direct analytical approach."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _circuit_state(theta, n, layers):
    """Compute the statevector of the 3-layer Ry/CZ circuit using vectorized ops."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    # Precompute bit masks for each qubit
    masks = [1 << i for i in range(n)]
    
    for l in range(layers):
        # Apply Ry(theta[l,i]) to each qubit i
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            m = masks[i]
            # Indices where qubit i is 0
            idx0 = np.array([b for b in range(dim) if (b & m) == 0], dtype=int)
            # Indices where qubit i is 1 (idx0 | m)
            idx1 = idx0 | m
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        # Apply CZ on adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mi = masks[i]
                mip1 = masks[i + 1]
                # Both bits set: sign flip
                both_set = np.array([b for b in range(dim) if (b & mi) and (b & mip1)], dtype=int)
                state[both_set] *= -1.0
    
    return state

def _energy_from_state(state, H):
    return float(np.dot(state, H @ state))

def _circuit_energy_fast(theta, n, H):
    """Fast energy computation for a given set of angles."""
    state = _circuit_state(theta, n, 3)
    return float(np.dot(state, H @ state))

def _energy(theta_flat, n, coupling, field, H, E0):
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3)
    return _energy_from_state(state, H)

def _exact_ground_state(n, coupling, field):
    """Get exact ground state and energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    return psi0, E0

def _project_to_circuit(psi_target, n, layers):
    """
    Directly find circuit angles that best approximate the target state.
    Uses layer-by-layer projection: for each layer, find the Ry angles
    that maximize overlap with the target after applying that layer's
    CZ (if any).
    
    This is a greedy approach that works well for shallow circuits.
    """
    dim = 1 << n
    # Start from |0>^n
    current_state = np.zeros(dim, dtype=float)
    current_state[0] = 1.0
    
    # We need to invert the circuit: target = U_circuit |0>^n
    # So |0>^n = U_circuit^{-1} target
    # U_circuit = Ry_layer_3 * CZ * Ry_layer_2 * CZ * Ry_layer_1
    # U_circuit^{-1} = Ry_layer_1^{-1} * CZ * Ry_layer_2^{-1} * CZ * Ry_layer_3^{-1}
    # Since CZ is self-inverse and Ry(-t) is inverse of Ry(t):
    # |0>^n = Ry(-theta[0]) * CZ * Ry(-theta[1]) * CZ * Ry(-theta[2]) * target
    
    # Work backwards from target
    # Step 1: Apply Ry(-theta[2]) to target, then CZ, then try to match intermediate
    # Actually, let's do it forward: we want to find theta such that
    # circuit(theta) |0> ≈ target
    
    # Greedy forward: for layer l, given we've already applied layers 0..l-1,
    # find theta[l] to maximize overlap with target after remaining CZ layers.
    
    # Alternative: use the fact that for 3 layers, we can do:
    # 1. Apply inverse of last layer's CZ (none for last layer, so just match Ry_3)
    #    After CZ_2 * Ry_2 * CZ_1 * Ry_1, we apply Ry_3.
    #    So we need Ry_3 |phi_2> ≈ target, where |phi_2> = CZ * Ry_2 * CZ * Ry_1 |0>
    #    For each qubit i, Ry_3[i] acts locally, so we can optimize independently.
    
    # Let's use a different strategy: direct numerical optimization with
    # a good initial guess from mean-field theory.
    
    return None  # Placeholder, will use numerical optimization

def _mean_field_init(n, coupling, field):
    """
    Mean-field / variational product state initialization.
    For H = -J sum Z_i Z_{i+1} - h sum X_i, a product state |theta>^n
    has energy per spin: <E>/n = -J cos(theta)^2 - h sin(theta)
    """
    theta = np.zeros((3, n))
    
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(min(ratio / 2.0, 1.0))
    else:
        t_opt = np.pi / 2
    
    # Try several distributions across layers
    return theta, t_opt

def _circuit_overlap_energy(theta_flat, n, coupling, field, H, psi0, E0):
    """Energy of the circuit state."""
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3)
    return _energy_from_state(state, H)

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    psi0, E0 = _exact_ground_state(n, coupling, field)
    
    # Mean-field initialization
    theta_mf, t_opt = _mean_field_init(n, coupling, field)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Original mean-field distribution
    tc = theta_mf.copy()
    tc[0] = t_opt * 0.5
    tc[1] = t_opt * 0.3
    tc[2] = t_opt * 0.2
    candidates.append(tc)
    
    # Equal distribution
    tc = np.full((3, n), t_opt / 3.0)
    candidates.append(tc)
    
    # Front-loaded
    tc = np.zeros((3, n))
    tc[0] = t_opt * 0.7
    tc[1] = t_opt * 0.2
    tc[2] = t_opt * 0.1
    candidates.append(tc)
    
    # Back-loaded
    tc = np.zeros((3, n))
    tc[0] = t_opt * 0.1
    tc[1] = t_opt * 0.2
    tc[2] = t_opt * 0.7
    candidates.append(tc)
    
    # All in last layer
    tc = np.zeros((3, n))
    tc[-1] = np.pi / 2
    candidates.append(tc)
    
    # Uniform pi/4
    candidates.append(np.full((3, n), np.pi / 4))
    
    # Zero
    candidates.append(np.zeros((3, n)))
    
    # Small angles
    candidates.append(np.full((3, n), 0.2))
    
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline:
            break
        e = _circuit_energy_fast(tc, n, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Random starts with different seeds
    rng = np.random.default_rng(12345)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        tc = rng.uniform(-np.pi, np.pi, size=(3, n))
        e = _circuit_energy_fast(tc, n, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Perturbations around best
    for _ in range(15):
        if time.time() > deadline - 5:
            break
        tc = best_theta + rng.normal(0, 0.3, size=(3, n))
        e = _circuit_energy_fast(tc, n, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Now do numerical optimization from the best starting point
    def objective(x):
        theta = x.reshape((3, n))
        return _circuit_energy_fast(theta, n, H)
    
    # Strategy 1: L-BFGS-B with numerical gradients - main optimization
    if time.time() < deadline - 15:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 2: Try optimization from second-best candidate too
    # Re-evaluate candidates to find second best
    second_best_theta = None
    second_best_energy = float('inf')
    for tc in candidates:
        if time.time() > deadline - 15:
            break
        e = _circuit_energy_fast(tc, n, H)
        if e < second_best_energy and e >= best_energy - 1e-10:
            # Only consider if it's close but not identical
            if np.allclose(tc, best_theta, atol=1e-6):
                continue
            second_best_energy = e
            second_best_theta = tc.copy()
    
    if second_best_theta is not None and time.time() < deadline - 15:
        try:
            res = opt.minimize(
                objective,
                second_best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 3: Nelder-Mead for polish
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 4: Another L-BFGS-B polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-17, 'gtol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final: try random restarts with optimization if time permits
    for _ in range(8):
        if time.time() > deadline - 5:
            break
        tc = best_theta + rng.normal(0, 0.4, size=(3, n))
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Ensure output is finite
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((3, n))
        best_theta[-1] = np.pi / 2
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
