"""Optimized quantum circuit parameters for Ising ground states using Newton-Raphson refinement."""
import numpy as np
import time

_DEADLINE = 150.0
_START_TIME = time.time()


def _check_deadline(margin=5.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i using Kronecker products."""
    total_dim = 2 ** n
    H = np.zeros((total_dim, total_dim), dtype=complex)
    
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        term = np.array([1.0])
        for q in range(n):
            if q == i or q == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
        
    # X terms
    for i in range(n):
        term = np.array([1.0])
        for q in range(n):
            if q == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
        
    return H


def _get_ground_state(H):
    """Get exact ground state vector."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs[:, 0]


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit in state vector using vectorized operations."""
    # Pre-compute bit masks
    mask = 1 << qubit
    indices = np.arange(2 ** n)
    bits = (indices >> qubit) & 1
    
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    new_state = np.zeros_like(state)
    
    # For bit == 0: |0> -> c|0> + s|1>
    mask_0 = bits == 0
    idx_0 = indices[mask_0]
    idx_1 = idx_0 | mask
    
    new_state[idx_0] += c * state[idx_0]
    new_state[idx_1] += s * state[idx_0]
    
    # For bit == 1: |1> -> -s|0> + c|1>
    mask_1 = bits == 1
    idx_1 = indices[mask_1]
    idx_0 = idx_1 ^ mask
    
    new_state[idx_1] += c * state[idx_1]
    new_state[idx_0] += -s * state[idx_1]
    
    return new_state


def _apply_cz(state, q1, q2, n):
    """Apply CZ gate between q1 and q2."""
    mask1 = 1 << q1
    mask2 = 1 << q2
    indices = np.arange(2 ** n)
    
    b1 = (indices >> q1) & 1
    b2 = (indices >> q2) & 1
    
    phase = np.ones(2 ** n)
    phase[(b1 == 1) & (b2 == 1)] = -1.0
    
    return phase * state


def _circuit_state(params, n, layers):
    """Build the circuit state from parameters."""
    total_dim = 2 ** n
    state = np.zeros(total_dim, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry(state, params[layer, q], q, n)
        
        # Apply CZ between adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
                
    return state


def _energy_and_gradient(params, n, coupling, field, layers, H):
    """Compute energy and gradient using parameter-shift rule."""
    # Energy
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state).real
    
    # Gradient using parameter-shift rule
    grad = np.zeros_like(params)
    for layer in range(layers):
        for q in range(n):
            # Shift +pi/2
            params_plus = params.copy()
            params_plus[layer, q] += np.pi / 2
            state_plus = _circuit_state(params_plus, n, layers)
            energy_plus = np.vdot(state_plus, H @ state_plus).real
            
            # Shift -pi/2
            params_minus = params.copy()
            params_minus[layer, q] -= np.pi / 2
            state_minus = _circuit_state(params_minus, n, layers)
            energy_minus = np.vdot(state_minus, H @ state_minus).real
            
            grad[layer, q] = (energy_plus - energy_minus) / 2.0
            
    return energy, grad


def _hessian(params, n, coupling, field, layers, H):
    """Compute Hessian numerically."""
    num_params = params.size
    hess = np.zeros((num_params, num_params))
    eps = 1e-4
    
    for i in range(num_params):
        for j in range(i, num_params):
            # Central difference for Hessian elements
            # H_ij = d^2 E / dx_i dx_j
            pp = params.copy()
            pp.flat[i] += eps
            pp.flat[j] += eps
            e_pp = _circuit_state(pp, n, layers)
            e_pp = np.vdot(e_pp, H @ e_pp).real
            
            pm = params.copy()
            pm.flat[i] += eps
            pm.flat[j] -= eps
            e_pm = _circuit_state(pm, n, layers)
            e_pm = np.vdot(e_pm, H @ e_pm).real
            
            mp = params.copy()
            mp.flat[i] -= eps
            mp.flat[j] += eps
            e_mp = _circuit_state(mp, n, layers)
            e_mp = np.vdot(e_mp, H @ e_mp).real
            
            mm = params.copy()
            mm.flat[i] -= eps
            mm.flat[j] -= eps
            e_mm = _circuit_state(mm, n, layers)
            e_mm = np.vdot(e_mm, H @ e_mm).real
            
            hess[i, j] = (e_pp - e_pm - e_mp + e_mm) / (4 * eps * eps)
            hess[j, i] = hess[i, j]
            
    return hess


def _heuristic_init(n, coupling, field, layers, H):
    """Initialize parameters based on exact ground state single-qubit expectations."""
    ground_state = _get_ground_state(H)
    params = np.zeros((layers, n))
    
    # Estimate single-qubit rotations from ground state
    # For each qubit, compute <Z_i> and <X_i> in ground state
    for q in range(n):
        mask = 1 << q
        indices = np.arange(2 ** n)
        bits = (indices >> q) & 1
        
        # <Z_i> = P(0) - P(1)
        p0 = np.sum(np.abs(ground_state[bits == 0]) ** 2)
        p1 = np.sum(np.abs(ground_state[bits == 1]) ** 2)
        z_exp = p0 - p1
        
        # <X_i> = 2 * Re(sum_{i<j, bit_i=0, bit_j=1} psi_i^* psi_j)
        x_exp = 0.0
        for idx in range(2 ** n):
            if bits[idx] == 0:
                flipped = idx | mask
                x_exp += 2 * np.real(np.conj(ground_state[idx]) * ground_state[flipped])
                
        # Ry(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>
        # <Z> = cos(theta), <X> = sin(theta)
        # So theta = atan2(<X>, <Z>)
        if abs(z_exp) < 1e-10 and abs(x_exp) < 1e-10:
            theta = 0.0
        else:
            theta = np.arctan2(x_exp, z_exp)
            
        # Distribute rotation across layers (simple equal split)
        for layer in range(layers):
            params[layer, q] = theta / layers
            
    return params


def angles(n, coupling, field, layers):
    """Optimize circuit parameters for the given Ising chain."""
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    
    # Initial guess from heuristic
    params = _heuristic_init(n, coupling, field, layers, H)
    best_params = params.copy()
    best_energy = _circuit_state(params, n, layers)
    best_energy = np.vdot(best_energy, H @ best_energy).real
    
    # Newton-Raphson refinement
    max_iters = 20
    for it in range(max_iters):
        if not _check_deadline():
            break
            
        energy, grad = _energy_and_gradient(params, n, coupling, field, layers, H)
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-8:
            break
            
        # Compute Hessian
        hess = _hessian(params, n, coupling, field, layers, H)
        
        # Regularize Hessian for stability
        hess += 1e-6 * np.eye(hess.shape[0])
        
        # Solve for step: hess * delta = -grad
        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            # Fallback to gradient descent if Hessian is singular
            delta = -grad / np.linalg.norm(grad) * 0.1
            
        # Line search to ensure decrease
        step_size = 1.0
        new_energy = energy
        for ls_iter in range(20):
            new_params = params + step_size * delta
            new_state = _circuit_state(new_params, n, layers)
            new_energy = np.vdot(new_state, H @ new_state).real
            
            if new_energy < energy:
                break
            step_size *= 0.5
            
        # Update
        params = params + step_size * delta
        
        # Track best
        if new_energy < best_energy:
            best_energy = new_energy
            best_params = params.copy()
            
    return best_params
