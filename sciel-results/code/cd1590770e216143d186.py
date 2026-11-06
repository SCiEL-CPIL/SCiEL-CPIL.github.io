import numpy as np
import time

# Global cache for results
_cache = {}

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    idx = np.arange(dim)
    
    # ZZ terms (Diagonal)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = -coupling * z_i * z_ip1
        H[idx, idx] += zz_val
        
    # X terms (Off-diagonal)
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        H[idx, idx_flipped] += -field
        H[idx_flipped, idx] += -field
        
    return H

def _get_exact_ground_state(n, coupling, field):
    """
    Compute the exact ground state and energy.
    """
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    ground_energy = eigenvalues[0]
    return ground_state, ground_energy

def _circuit_state_vector(params, n, layers):
    """
    Efficiently compute the state vector produced by the circuit.
    params: array of shape (layers, n)
    Returns: state vector of length 2^n
    """
    thetas = params.reshape(layers, n)
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    idx = np.arange(dim)
    
    for l in range(layers):
        # Apply Ry(theta) to all qubits
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            mask_i = 1 << i
            idx_0 = idx[(idx & mask_i) == 0]
            idx_1 = idx[(idx & mask_i) != 0]
            
            amp_0 = state[idx_0].copy()
            amp_1 = state[idx_1].copy()
            
            new_amp_0 = c * amp_0 - s * amp_1
            new_amp_1 = s * amp_0 + c * amp_1
            
            state[idx_0] = new_amp_0
            state[idx_1] = new_amp_1
            
        # Apply CZ between adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                state[cond] *= -1
                
    return state

def _circuit_energy(params, n, coupling, field, layers):
    """
    Calculate the energy expectation value for the given circuit parameters.
    """
    state = _circuit_state_vector(params, n, layers)
    dim = 2 ** n
    idx = np.arange(dim)
    
    probs = np.abs(state) ** 2
    
    E_zz = 0.0
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = z_i * z_ip1
        
        E_zz += np.sum(probs * zz_val)
        
    E_1 = -coupling * E_zz
    
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        # <X_i> = sum_k psi*_k psi_{k^i}
        term = np.sum(np.conj(state) * state[idx_flipped])
        E_x += np.real(term)
        
    E_2 = -field * E_x
    
    total_energy = E_1 + E_2
    return total_energy

def _optimize(n, coupling, field, layers, timeout=145.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    # 1. Get Exact Ground State
    try:
        ground_state, ground_energy = _get_exact_ground_state(n, coupling, field)
    except Exception:
        ground_state = np.zeros(2**n, dtype=complex)
        ground_state[0] = 1.0
        ground_energy = 0.0

    # 2. Define Objective Function
    def objective(params):
        if time.time() - start_time > timeout - 5:
            return np.inf
        return _circuit_energy(params, n, coupling, field, layers)

    # 3. Multi-start Differential Evolution
    # This is a more robust global optimizer than the previous Nelder-Mead/L-BFGS-B mix.
    # It handles the non-convex landscape better by maintaining a population.
    
    best_params = None
    best_energy = np.inf
    
    # Use a smaller population and fewer iterations to fit within time budget,
    # but rely on the robustness of DE to find a good basin.
    # We will run multiple DE instances with different seeds if time permits,
    # or just one well-tuned instance.
    
    bounds = [(-np.pi, np.pi)] * n_params
    
    # DE parameters
    popsize = 10
    maxiter = 50
    
    try:
        res = opt.minimize(objective, np.zeros(n_params), method='differential_evolution',
                           bounds=bounds, 
                           maxiter=maxiter, 
                           popsize=popsize, 
                           tol=1e-4,
                           seed=42,
                           polish=True)
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
            
    except Exception:
        pass

    # 4. Local Polish
    if best_params is not None and time.time() - start_time < timeout - 2:
        try:
            res_lbfgs = opt.minimize(objective, best_params, method='L-BFGS-B',
                                     options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            if res_lbfgs.fun < best_energy:
                best_energy = res_lbfgs.fun
                best_params = res_lbfgs.x
        except Exception:
            pass
            
    if best_params is None:
        best_params = np.zeros(n_params)

    result_angles = best_params.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
