import numpy as np
from scipy.optimize import minimize

def _apply_ry(state, theta, i):
    """Apply Ry(theta) to qubit i. Returns new state."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    mask = 1 << i
    indices_0 = np.where((np.arange(len(state)) & mask) == 0)[0]
    indices_1 = indices_0 | mask
    
    v0 = state[indices_0].copy()
    v1 = state[indices_1].copy()
    
    new_state = state.copy()
    new_state[indices_0] = c * v0 - s * v1
    new_state[indices_1] = s * v0 + c * v1
    return new_state

def _apply_cz(state, i, j):
    """Apply CZ gate between qubits i and j. Returns new state."""
    mask_i = 1 << i
    mask_j = 1 << j
    indices = np.where(((np.arange(len(state)) & mask_i) != 0) & ((np.arange(len(state)) & mask_j) != 0))[0]
    new_state = state.copy()
    new_state[indices] *= -1.0
    return new_state

def _compute_energy_and_grad(n, coupling, field, angles):
    """
    Compute energy and analytic gradient using backpropagation.
    angles: shape (layers, n)
    """
    layers = angles.shape[0]
    dim = 2**n
    
    # Precompute masks and indices for speed
    indices_0_list = []
    indices_1_list = []
    for i in range(n):
        mask = 1 << i
        idx0 = np.where((np.arange(dim) & mask) == 0)[0]
        idx1 = idx0 | mask
        indices_0_list.append(idx0)
        indices_1_list.append(idx1)
        
    cz_indices_list = []
    for i in range(n-1):
        mask_i = 1 << i
        mask_j = 1 << (i+1)
        idx = np.where(((np.arange(dim) & mask_i) != 0) & ((np.arange(dim) & mask_j) != 0))[0]
        cz_indices_list.append(idx)

    # Forward pass
    # Store states before each Ry gate
    states_before_ry = np.zeros((layers, n, dim), dtype=complex)
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            states_before_ry[l, i] = state
            state = _apply_ry(state, angles[l, i], i)
        
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1)
                
    final_state = state
    
    # Compute H |psi>
    h_vec = np.zeros(dim, dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        z_i = np.where((np.arange(dim) & mask_i) == 0, 1.0, -1.0)
        z_j = np.where((np.arange(dim) & mask_j) == 0, 1.0, -1.0)
        h_vec += -coupling * (z_i * z_j) * final_state
        
    # X terms
    for i in range(n):
        idx0 = indices_0_list[i]
        idx1 = indices_1_list[i]
        h_vec[idx0] += -field * final_state[idx1]
        h_vec[idx1] += -field * final_state[idx0]
        
    energy = np.vdot(final_state, h_vec).real
    
    # Backpropagation
    v = h_vec
    grad = np.zeros_like(angles)
    
    for l in range(layers - 1, -1, -1):
        if l < layers - 1:
            for i in range(n - 2, -1, -1):
                idx = cz_indices_list[i]
                v[idx] *= -1.0
                
        for i in range(n - 1, -1, -1):
            theta = angles[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            idx0 = indices_0_list[i]
            idx1 = indices_1_list[i]
            
            v0_out = v[idx0].copy()
            v1_out = v[idx1].copy()
            
            v0_in = c * v0_out + s * v1_out
            v1_in = -s * v0_out + c * v1_out
            
            psi_in = states_before_ry[l, i]
            psi0_in = psi_in[idx0]
            psi1_in = psi_in[idx1]
            
            dpsi0 = -0.5 * (s * psi0_in + c * psi1_in)
            dpsi1 = 0.5 * (c * psi0_in - s * psi1_in)
            
            inner = np.vdot(v0_out, dpsi0) + np.vdot(v1_out, dpsi1)
            
            grad[l, i] = 2.0 * inner.real
            
            v[idx0] = v0_in
            v[idx1] = v1_in
            
    return energy, grad

def angles(n, coupling, field, layers):
    """
    Optimize parameters using L-BFGS-B with analytic gradient.
    """
    n_params = layers * n
    
    def unpack(params):
        return params.reshape((layers, n))
        
    def objective(params):
        ang = unpack(params)
        e, g = _compute_energy_and_grad(n, coupling, field, ang)
        return e, g.flatten()

    best_energy = np.inf
    best_params = None
    
    np.random.seed(42)
    guesses = [
        np.zeros(n_params),
        np.random.uniform(-np.pi, np.pi, n_params),
        np.full(n_params, np.pi / 4),
        np.full(n_params, -np.pi / 4)
    ]
    
    for guess in guesses:
        try:
            res = minimize(objective, guess, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
        
    return unpack(best_params)
