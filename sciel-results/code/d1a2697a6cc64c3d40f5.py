"""State-fitting quantum circuit optimization via adjoint gradients and least-squares."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 130.0  # safety margin before 150s kill


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
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
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _exact_ground_state(H):
    """Compute exact ground state vector and energy."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    e0 = np.real(eigenvalues[0])
    psi0 = eigenvectors[:, 0].copy()
    max_idx = np.argmax(np.abs(psi0))
    phase = np.angle(psi0[max_idx])
    psi0 = psi0 * np.exp(-1j * phase)
    return psi0, e0


def _apply_ry(psi, q, theta, n):
    """Apply Ry(theta) on qubit q to state psi (flattened)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    shape = [2] * n
    psi_r = psi.reshape(shape)
    gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
    psi_moved = np.moveaxis(psi_r, q, 0)
    psi_out = np.tensordot(gate, psi_moved, axes=([1], [0]))
    return np.moveaxis(psi_out, 0, q).reshape(-1)


def _apply_cz(psi, q, n):
    """Apply CZ between qubit q and q+1 to state psi (flattened)."""
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
    factor = np.ones(psi_moved.shape, dtype=np.complex128)
    idx11 = (slice(None),) * (n - 2) + (1, 1)
    factor[idx11] = -1.0
    psi_moved = psi_moved * factor
    return np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)


def _apply_ry_dag(psi, q, theta, n):
    """Apply Ry(theta)^dag on qubit q to state psi (flattened)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    shape = [2] * n
    psi_r = psi.reshape(shape)
    gate = np.array([[c, s], [-s, c]], dtype=np.complex128)
    psi_moved = np.moveaxis(psi_r, q, 0)
    psi_out = np.tensordot(gate, psi_moved, axes=([1], [0]))
    return np.moveaxis(psi_out, 0, q).reshape(-1)


def _apply_cz_dag(psi, q, n):
    """Apply CZ^dag = CZ on qubit q and q+1 to state psi (flattened)."""
    return _apply_cz(psi, q, n)


def _circuit_state(params, n, layers):
    """Compute the final state of the circuit from angle parameters."""
    t = params.reshape(layers, n)
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            psi = _apply_ry(psi, q, t[layer, q], n)
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
    
    return psi


def _circuit_energy_and_adjoint_grad(params, n, H, e0, layers=3):
    """
    Compute energy and exact gradient using the adjoint method.
    
    Circuit: |0> -> [Ry layer 0] -> [CZ layer 0] -> [Ry layer 1] -> [CZ layer 1] -> [Ry layer 2]
    Energy E = <psi_final| H |psi_final>
    dE/dtheta_k = 2 Re(<psi_final| H |dpsi/dtheta_k>)
    
    Using adjoint: propagate |phi> = H|psi_final> backwards through the circuit.
    For each Ry gate at position k, the gradient contribution is:
    dE/dtheta_k = 2 Re( <phi_at_k| dRy/dtheta |psi_at_k> )
    where dRy/dtheta = -i/2 * X_q (in the rotated frame)
    
    More precisely, for a unitary U(theta) = Ry(theta):
    dE/dtheta = 2 Re( <psi| U^dag H U | dU/dtheta U^dag |psi> )
    
    Let |psi_k> be the state before gate k, |psi_{k+1}> = U_k |psi_k>
    Then d|psi_final>/dtheta_k = U_{final}...U_{k+1} (dU_k/dtheta) U_k^dag |psi_{k+1}>
    
    So dE/dtheta_k = 2 Re( <psi_{k+1}| U_{k+1}^dag ... U_final^dag H U_final ... U_{k+1} (dU_k/dtheta) |psi_{k+1}> )
    
    Define |phi_k> = U_k^dag U_{k+1}^dag ... U_final^dag H U_final ... U_{k+1} U_k |psi_k>
    Then dE/dtheta_k = 2 Re( <phi_k| dU_k/dtheta |psi_k> )
    
    For Ry(theta): dRy/dtheta = -i/2 * X_q (applied in the same qubit space)
    Actually: d/dtheta Ry(theta) = -i/2 * X_q * Ry(theta) = Ry(theta) * (-i/2 * X_q)
    
    So: dE/dtheta_k = 2 Re( <phi_k| (-i/2) X_q |psi_k> )
    where |phi_k> is the adjoint state propagated back to just before gate k.
    """
    t = params.reshape(layers, n)
    dim = 2 ** n
    num_params = layers * n
    
    # Forward pass: store all intermediate states
    states = []  # states[j] = state after j-th gate (total gates = layers*n + (layers-1)*(n-1))
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    gate_info = []  # list of (type, q, theta) for each gate in order
    
    for layer in range(layers):
        for q in range(n):
            theta = t[layer, q]
            states.append(psi.copy())
            psi = _apply_ry(psi, q, theta, n)
            gate_info.append(('ry', q, theta))
        if layer < layers - 1:
            for q in range(n - 1):
                states.append(psi.copy())
                psi = _apply_cz(psi, q, n)
                gate_info.append(('cz', q, 0.0))
    
    psi_final = psi.copy()
    energy = np.vdot(psi_final, H @ psi_final).real
    obj_val = (energy - e0) / n
    
    # Backward pass: compute adjoint states
    # Start with |chi> = H |psi_final>
    chi = H @ psi_final
    
    grad = np.zeros(num_params)
    
    # Map gate index to parameter index
    gate_to_param = {}
    param_idx = 0
    for layer in range(layers):
        for q in range(n):
            gate_to_param[(layer, 'ry', q)] = param_idx
            param_idx += 1
        if layer < layers - 1:
            for q in range(n - 1):
                pass  # CZ gates don't have parameters
    
    # Backward: propagate chi through gates in reverse
    # chi represents the "cost function gradient" at the current point
    # After processing gate g (going backwards), chi becomes the adjoint state before gate g
    
    chi_current = chi.copy()
    
    # Process gates in reverse order
    for g in range(len(gate_info) - 1, -1, -1):
        gtype, q, theta = gate_info[g]
        
        if gtype == 'ry':
            # Compute gradient contribution for this Ry gate
            # dE/dtheta = 2 Re( <chi_current| (-i/2) X_q |states[g]> )
            # where chi_current is the adjoint state just after this gate
            # and states[g] is the state just before this gate
            
            psi_before = states[g]
            
            # Compute X_q |psi_before>
            shape = [2] * n
            psi_r = psi_before.reshape(shape)
            x_psi = np.moveaxis(psi_r, q, 0).copy()
            x_new = np.zeros_like(x_psi)
            x_new[0] = x_psi[1]
            x_new[1] = x_psi[0]
            x_flat = np.moveaxis(x_new, 0, q).reshape(-1)
            
            # Gradient contribution
            layer = g // n if g < n * layers else None
            # Find which layer and qubit this is
            # Reconstruct from gate_info ordering
            # Actually, let's just track the parameter index directly
            
            # The gradient is: 2 Re( <chi_current| (-i/2) X_q |psi_before> )
            inner = np.vdot(chi_current, (-0.5j) * x_flat)
            
            # Find the parameter index for this gate
            # Count how many Ry gates came before this one
            ry_count = 0
            for gg in range(g):
                if gate_info[gg][0] == 'ry':
                    ry_count += 1
            
            grad[ry_count] = 2.0 * inner.real / n
            
            # Propagate chi backwards through Ry
            chi_current = _apply_ry_dag(chi_current, q, theta, n)
        
        else:  # cz
            # CZ is self-adjoint
            chi_current = _apply_cz_dag(chi_current, q, n)
    
    return obj_val, grad


def _circuit_state_jacobian(params, n, layers):
    """
    Compute the final state and its Jacobian using adjoint method.
    Returns (psi, J) where J has shape (dim, num_params).
    
    For state fitting, we need dpsi/dtheta_k for each parameter.
    Using adjoint: for each parameter k, dpsi/dtheta_k can be computed.
    But that's O(num_params) backward passes.
    
    Alternative: use the fact that we only need the Jacobian for LS,
    and compute it via finite differences if adjoint is too slow.
    
    Actually, for the LS objective, we can just use the energy gradient
    with L-BFGS-B directly, which is more efficient.
    """
    t = params.reshape(layers, n)
    dim = 2 ** n
    num_params = layers * n
    
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    J = np.zeros((dim, num_params), dtype=np.complex128)
    
    for layer in range(layers):
        for q in range(n):
            theta = t[layer, q]
            
            psi_before = psi.copy()
            psi = _apply_ry(psi, q, theta, n)
            
            for k in range(num_params):
                if np.any(J[:, k]):
                    J[:, k] = _apply_ry(J[:, k], q, theta, n)
            
            shape = [2] * n
            psi_r = psi.reshape(shape)
            x_psi = np.moveaxis(psi_r, q, 0).copy()
            x_new = np.zeros_like(x_psi)
            x_new[0] = x_psi[1]
            x_new[1] = x_psi[0]
            x_flat = np.moveaxis(x_new, 0, q).reshape(-1)
            
            k = layer * n + q
            J[:, k] += (-0.5j) * x_flat
        
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
                for k in range(num_params):
                    if np.any(J[:, k]):
                        J[:, k] = _apply_cz(J[:, k], q, n)
    
    return psi, J


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters using adjoint-gradient L-BFGS-B."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    psi_target, e0 = _exact_ground_state(H)
    num_params = layers * n
    
    def objective(params):
        psi = _circuit_state(params, n, layers)
        energy = np.vdot(psi, H @ psi).real
        return (energy - e0) / n
    
    def objective_grad(params):
        val, grad = _circuit_energy_and_adjoint_grad(params, n, H, e0, layers)
        return val, grad
    
    # Generate diverse initial guesses
    np.random.seed(42)
    ratio = field / coupling if coupling > 0 else 1.0
    
    restarts = []
    
    # Physics-based initial guesses
    base_angle = 0.5 * np.pi * np.tanh(ratio)
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.2]:
        restarts.append(np.full(num_params, base_angle * scale))
    
    # All zeros
    restarts.append(np.zeros(num_params))
    
    # All pi/2
    restarts.append(np.full(num_params, np.pi / 2))
    
    # Random initializations
    for s in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
        restarts.append(np.random.randn(num_params) * s)
    
    # Alternating patterns
    for s in [0.2, 0.4, 0.6]:
        for sign in [1, -1]:
            pattern = np.zeros(num_params)
            for i in range(num_params):
                l = i // n
                q = i % n
                pattern[i] = s * sign * (1.0 if q % 2 == 0 else -1.0)
            restarts.append(pattern)
    
    # Layer-varying patterns
    for s in [0.3, 0.5, 0.7]:
        pattern = np.zeros(num_params)
        for l in range(layers):
            for q in range(n):
                pattern[l * n + q] = s * (l + 1) / layers
        restarts.append(pattern)
    
    # Sort by objective value
    scored_restarts = []
    for x in restarts:
        x = np.clip(x, -np.pi, np.pi)
        try:
            v = objective(x)
            scored_restarts.append((v, x.copy()))
        except:
            scored_restarts.append((np.inf, x.copy()))
    scored_restarts.sort(key=lambda t: t[0])
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B with exact adjoint gradients from best initial guesses
    # This is much faster than parameter-shift since each gradient eval is O(1) circuit passes
    for i, (v0, x0) in enumerate(scored_restarts[:15]):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 30:
            break
        
        try:
            res = opt.minimize(
                objective_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * num_params,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30, 'maxfun': 150}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 25:
            break
    
    # Phase 2: Refine top candidates with more iterations
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
    candidates.extend([x for _, x in scored_restarts[:5]])
    
    for x0 in candidates[:4]:
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 15:
            break
        
        try:
            res = opt.minimize(
                objective_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * num_params,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50, 'maxfun': 300}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 10:
            break
    
    # Phase 3: Final Nelder-Mead polish
    if best_params is not None:
        elapsed = time.time() - _start_time
        if elapsed < _TIME_LIMIT - 5:
            try:
                res = opt.minimize(
                    objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        best_params = scored_restarts[0][1]
    
    theta = best_params.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
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
