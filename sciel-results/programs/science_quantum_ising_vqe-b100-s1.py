# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_cz_mask_cache = {}
_idx_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms
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
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0

def _get_cz_masks(n):
    key = n
    if key in _cz_mask_cache:
        return _cz_mask_cache[key]
    dim = 1 << n
    masks = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((np.arange(dim) & mask1) != 0) & ((np.arange(dim) & mask2) != 0)
        masks.append(both_set)
    _cz_mask_cache[key] = masks
    return masks

def _get_idx_arrays(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    idx_0_arr = {}
    idx_1_arr = {}
    for q in range(n):
        mask = 1 << q
        idx_0 = np.array([i for i in range(dim) if (i & mask) == 0])
        idx_1 = idx_0 + mask
        idx_0_arr[q] = idx_0
        idx_1_arr[q] = idx_1
    _idx_cache[key] = (idx_0_arr, idx_1_arr)
    return _idx_cache[key]

def _circuit_state_fast(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    idx_0_arr, idx_1_arr = _get_idx_arrays(n)
    cz_masks = _get_cz_masks(n)

    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0 = idx_0_arr[q]
            idx_1 = idx_1_arr[q]
            
            vals_0 = state[idx_0]
            vals_1 = state[idx_1]
            
            new_vals_0 = c * vals_0 - s * vals_1
            new_vals_1 = s * vals_0 + c * vals_1
            
            state[idx_0] = new_vals_0
            state[idx_1] = new_vals_1

        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state[cz_masks[i]] *= -1.0

    return state

def _circuit_state_and_grad(n, thetas, layers):
    """
    Compute state and analytical gradient of energy w.r.t. thetas.
    Returns (psi, grad) where grad has shape (layers, n).
    """
    dim = 1 << n
    idx_0_arr, idx_1_arr = _get_idx_arrays(n)
    cz_masks = _get_cz_masks(n)
    
    # Forward pass: store states after each Ry gate application
    # states_after_ry[layer][q] = state after applying Ry(layer, q)
    # We need the state BEFORE each Ry to compute gradients
    # Store: pre_ry[layer][q] = state before Ry(layer, q)
    
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    # Store pre-Ry states for gradient computation
    # For each layer and qubit, store the state before that Ry
    # This is memory-heavy but n is small (4-6) and layers=3
    pre_ry_states = []  # list of lists: pre_ry_states[layer][q]
    
    for layer in range(layers):
        layer_pre = []
        for q in range(n):
            # Store state before this Ry
            layer_pre.append(state.copy())
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0 = idx_0_arr[q]
            idx_1 = idx_1_arr[q]
            
            vals_0 = state[idx_0]
            vals_1 = state[idx_1]
            
            new_vals_0 = c * vals_0 - s * vals_1
            new_vals_1 = s * vals_0 + c * vals_1
            
            state[idx_0] = new_vals_0
            state[idx_1] = new_vals_1
        pre_ry_states.append(layer_pre)
        
        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state[cz_masks[i]] *= -1.0
    
    psi = state
    
    # Backward pass: compute gradient
    # dE/dtheta = 2 * Re[ (dpsi/dtheta)^H H psi ]
    # For each Ry(layer, q), dpsi/dtheta is the derivative of the state w.r.t. that theta
    
    H = _build_H(n, coupling, field)  # Need coupling and field
    
    # Actually, we need to pass H or compute it. Let's restructure.
    # This function is getting complex. Let me use a different approach.
    
    # Use automatic differentiation via finite differences but with a better step size
    # Or, compute the gradient using the chain rule properly
    
    # Let's use a simpler approach: compute gradient by perturbing each parameter
    # But use a central difference with a small step
    
    grad = np.zeros((layers, n))
    eps = 1e-7
    
    for l in range(layers):
        for q in range(n):
            theta_plus = thetas.copy()
            theta_plus[l, q] += eps
            psi_plus = _circuit_state_fast(n, theta_plus, layers)
            e_plus = np.real(np.conj(psi_plus) @ H @ psi_plus)
            
            theta_minus = thetas.copy()
            theta_minus[l, q] -= eps
            psi_minus = _circuit_state_fast(n, theta_minus, layers)
            e_minus = np.real(np.conj(psi_minus) @ H @ psi_minus)
            
            grad[l, q] = (e_plus - e_minus) / (2 * eps)
    
    return psi, grad

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)

def _objective_and_grad(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    e_circuit = np.real(np.conj(psi) @ H @ psi)
    val = (e_circuit - e0) / n
    
    # Compute gradient using finite differences (central)
    grad = np.zeros_like(thetas)
    eps = 1e-7
    for l in range(layers):
        for q in range(n):
            tp = thetas.copy()
            tp[l, q] += eps
            psi_p = _circuit_state_fast(n, tp, layers)
            e_p = np.real(np.conj(psi_p) @ H @ psi_p)
            
            tm = thetas.copy()
            tm[l, q] -= eps
            psi_m = _circuit_state_fast(n, tm, layers)
            e_m = np.real(np.conj(psi_m) @ H @ psi_m)
            
            grad[l, q] = (e_p - e_m) / (2 * eps * n)
    
    return val, grad.flatten()

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _heuristic_init(n, coupling, field, layers):
    """
    Heuristic initialization based on ground state marginals.
    For a single Ry layer on |0>^n, the state is a product state.
    We estimate angles from the ground state's single-qubit marginals.
    """
    H = _build_H(n, coupling, field)
    dim = 1 << n
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = eigvecs[:, 0]
    
    # Estimate single-qubit excitation probability for each qubit
    # For a product state from one Ry layer: P(qubit i = 1) = sin^2(theta_i / 2)
    p_excite = np.zeros(n)
    for q in range(n):
        mask = 1 << q
        idx_1 = np.array([i for i in range(dim) if (i & mask) != 0])
        p_excite[q] = np.sum(np.abs(psi0[idx_1])**2)
    
    # Invert: theta = 2 * arcsin(sqrt(p_excite))
    p_excite = np.clip(p_excite, 1e-10, 1 - 1e-10)
    theta_est = 2.0 * np.arcsin(np.sqrt(p_excite))
    
    # For 3 layers, distribute the rotation across layers
    # Simple heuristic: equal distribution
    thetas = np.zeros((layers, n))
    for l in range(layers):
        thetas[l, :] = theta_est / layers
    
    return thetas.flatten()

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Get physics-informed initialization
    phys_init = _heuristic_init(n, coupling, field, layers)
    
    best_x = phys_init.copy()
    best_val = _objective(best_x, n, coupling, field, layers, e0)
    
    # Generate diverse initializations
    inits = []
    
    # Init 0: Physics-informed (best first)
    inits.append(phys_init.copy())
    
    # Init 1: All zeros
    inits.append(np.zeros(layers * n))
    
    # Init 2: All pi/2
    inits.append(np.full(layers * n, np.pi / 2))
    
    # Init 3: All pi
    inits.append(np.full(layers * n, np.pi))
    
    # Init 4: Based on ratio
    ratio = field / coupling if coupling != 0 else field
    theta_est = (np.pi / 2) * (1 - np.exp(-ratio))
    inits.append(np.full(layers * n, theta_est))
    
    # Init 5: Small angles
    inits.append(np.full(layers * n, 0.5))
    inits.append(np.full(layers * n, 1.0))
    inits.append(np.full(layers * n, 1.5))
    
    # Init 6-15: Random with different seeds
    for i in range(10):
        np.random.seed(100 + i)
        inits.append(np.random.uniform(-np.pi, np.pi, size=(layers * n)))
    
    # Init 16-20: Random with bias toward [0, pi]
    for i in range(5):
        np.random.seed(200 + i)
        inits.append(np.random.uniform(0, np.pi, size=(layers * n)))
    
    # Sort: put physics-informed first, then others
    # Already have phys_init at index 0
    
    num_restarts = min(len(inits), 20 if n <= 4 else 16)
    
    H = _build_H(n, coupling, field)
    
    def obj_and_grad(p):
        thetas = p.reshape((layers, n))
        psi = _circuit_state_fast(n, thetas, layers)
        e_circuit = np.real(np.conj(psi) @ H @ psi)
        val = (e_circuit - e0) / n
        
        # Compute gradient using finite differences (central)
        grad = np.zeros_like(thetas)
        eps = 1e-7
        for l in range(layers):
            for q in range(n):
                tp = thetas.copy()
                tp[l, q] += eps
                psi_p = _circuit_state_fast(n, tp, layers)
                e_p = np.real(np.conj(psi_p) @ H @ psi_p)
                
                tm = thetas.copy()
                tm[l, q] -= eps
                psi_m = _circuit_state_fast(n, tm, layers)
                e_m = np.real(np.conj(psi_m) @ H @ psi_m)
                
                grad[l, q] = (e_p - e_m) / (2 * eps * n)
        
        return val, grad.flatten()
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        x_init = inits[i].copy()
        
        try:
            # Use BFGS with analytical gradient
            res = minimize(obj_and_grad, x_init, method='BFGS', jac=True,
                          options={'maxiter': 300, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish from best solution (derivative-free, can escape local minima)
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res_nm = minimize(lambda p: _objective(p, n, coupling, field, layers, e0), best_x,
                             method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Final high-precision BFGS polish with gradient
    remaining = deadline - time.time()
    if remaining > 3.0:
        try:
            res = minimize(obj_and_grad, best_x, method='BFGS', jac=True,
                          options={'maxiter': 1000, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
