import numpy as np
import scipy.optimize as opt
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
    
    # Precompute bit values for efficiency
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

def _circuit_energy_and_grad(params, n, coupling, field, layers):
    """
    Calculate the energy expectation value and its gradient for the given circuit parameters.
    Returns (energy, gradient).
    """
    thetas = params.reshape(layers, n)
    
    # Start with |0...0>
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    # Precompute indices for CZ
    idx = np.arange(2**n)
    
    # We will store the state before each Ry layer to compute gradients via backprop
    # states_before_ry[l] is the state before applying Ry at layer l
    states_before_ry = [None] * layers
    
    # Forward pass
    current_state = state.copy()
    
    for l in range(layers):
        states_before_ry[l] = current_state.copy()
        
        # Apply Ry(theta) to all qubits
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            mask_i = 1 << i
            idx_0 = idx[(idx & mask_i) == 0]
            idx_1 = idx[(idx & mask_i) != 0]
            
            amp_0 = current_state[idx_0].copy()
            amp_1 = current_state[idx_1].copy()
            
            new_amp_0 = c * amp_0 - s * amp_1
            new_amp_1 = s * amp_0 + c * amp_1
            
            current_state[idx_0] = new_amp_0
            current_state[idx_1] = new_amp_1
            
        # Apply CZ between adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                current_state[cond] *= -1
                
    final_state = current_state
    
    # Calculate Energy and Gradient components
    # E = -coupling * sum <Z_i Z_{i+1}> - field * sum <X_i>
    
    probs = np.abs(final_state) ** 2
    
    # Precompute Z_i and Z_i Z_{i+1} values for each basis state
    z_vals = np.zeros((n, 2**n))
    zz_vals = np.zeros((n-1, 2**n))
    for i in range(n):
        mask_i = 1 << i
        bit_i = (idx >> i) & 1
        z_vals[i] = 1 - 2 * bit_i
        
    for i in range(n - 1):
        zz_vals[i] = z_vals[i] * z_vals[i+1]
        
    # Energy
    E_zz = np.sum(probs * zz_vals)
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        term = np.sum(np.conj(final_state) * final_state[idx_flipped])
        E_x += np.real(term)
        
    total_energy = -coupling * E_zz - field * E_x
    
    # Gradient Calculation
    # dE/dtheta_{l,i} = Re[ <psi_final| H | d psi_final / d theta_{l,i} > ]
    # We use the trick: dE/dtheta = 2 * Re[ <psi| H | O | psi> ] is not quite right for non-Hermitian O.
    # Correct formula: dE/dtheta = Re[ <psi| H | dpsi/dtheta> + <dpsi/dtheta| H | psi> ]
    # Since H is Hermitian, <dpsi|H|psi> = (<psi|H|dpsi>)*
    # So dE/dtheta = 2 * Re[ <psi| H | dpsi/dtheta> ]
    
    # Let's compute |dpsi/dtheta> for each parameter.
    # This is expensive if done naively. Instead, we can use the chain rule backwards.
    # However, for small n, we can just simulate the forward pass with the perturbed state.
    # But that's O(N_params * N_ops).
    # A better way: Backpropagation of the adjoint.
    # Let |phi> be the adjoint state such that dE/dtheta = 2 Re[ <phi | dU/dtheta | psi_before> ]
    # Actually, it's easier to just compute the gradient numerically if analytical is too complex,
    # BUT the prompt suggests analytical gradients are faster.
    
    # Let's try a semi-analytical approach.
    # For each layer l and qubit i, we need the derivative of the final state w.r.t theta_{l,i}.
    # This is equivalent to applying the "error" from the later layers back to the derivative of the gate.
    
    # Let's implement the adjoint method.
    # 1. Compute H |psi_final>
    H_psi = np.zeros(2**n, dtype=complex)
    # H = -coupling * sum ZZ - field * sum X
    # ZZ is diagonal
    zz_diag = np.zeros(2**n)
    for i in range(n-1):
        zz_diag += zz_vals[i]
    H_psi -= coupling * zz_diag * final_state
    
    # X is off-diagonal
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        H_psi[idx] -= field * final_state[idx_flipped]
        # Wait, H_psi[k] = sum_j H[k,j] psi[j]
        # H[k, k^i] = -field
        # So H_psi[k] += -field * psi[k^i]
        
    # 2. Propagate <phi| backwards
    # phi_final = H |psi_final>
    phi = H_psi.copy()
    
    grads = np.zeros_like(thetas)
    
    for l in range(layers - 1, -1, -1):
        # Before applying CZ (if any) at layer l, we need to undo the CZ on phi
        # If l < layers - 1, CZ was applied AFTER Ry.
        # So in reverse, we undo CZ first.
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                phi[cond] *= -1 # CZ is self-inverse and Hermitian
                
        # Now phi is the adjoint state BEFORE the Ry layer l.
        # We need to compute dE/dtheta_{l,i} = 2 Re[ <phi | dU_Ry(theta_{l,i}) | psi_before_ry_l> ]
        # dU_Ry/dtheta = 1/2 * [[-sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
        
        psi_before = states_before_ry[l]
        
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            mask_i = 1 << i
            idx_0 = idx[(idx & mask_i) == 0]
            idx_1 = idx[(idx & mask_i) != 0]
            
            # Extract components
            phi_0 = phi[idx_0]
            phi_1 = phi[idx_1]
            
            psi_0 = psi_before[idx_0]
            psi_1 = psi_before[idx_1]
            
            # dU/dtheta * psi
            # dU/dtheta = 0.5 * [[-s, -c], [c, -s]]
            # dpsi_0/dtheta = 0.5 * (-s * psi_0 - c * psi_1)
            # dpsi_1/dtheta = 0.5 * (c * psi_0 - s * psi_1)
            
            dpsi_0 = 0.5 * (-s * psi_0 - c * psi_1)
            dpsi_1 = 0.5 * (c * psi_0 - s * psi_1)
            
            # <phi | dpsi> = sum phi_0^* dpsi_0 + sum phi_1^* dpsi_1
            inner = np.sum(np.conj(phi_0) * dpsi_0) + np.sum(np.conj(phi_1) * dpsi_1)
            
            grads[l, i] = 2 * np.real(inner)
            
        # Now propagate phi backwards through the Ry layer to get phi for the previous layer
        # phi_prev = (dU/dtheta)^dag phi ? No, phi_prev = U^dag phi
        # U = [[c, -s], [s, c]]
        # U^dag = [[c, s], [-s, c]]
        
        new_phi_0 = c * phi_0 + s * phi_1
        new_phi_1 = -s * phi_0 + c * phi_1
        
        phi[idx_0] = new_phi_0
        phi[idx_1] = new_phi_1
        
    return total_energy, grads.reshape(-1)

def _optimize(n, coupling, field, layers, timeout=145.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    # 1. Get Exact Ground State for Initialization
    try:
        ground_state, ground_energy = _get_exact_ground_state(n, coupling, field)
        
        # Compute magnetizations from exact ground state
        dim = 2 ** n
        idx = np.arange(dim)
        probs = np.abs(ground_state) ** 2
        mags = np.zeros(n)
        for i in range(n):
            mask = 1 << i
            bit_vals = (idx >> i) & 1
            z_vals = 1 - 2 * bit_vals
            mags[i] = np.sum(probs * z_vals)
            
        theta_mag = np.arccos(np.clip(mags, -1, 1))
        
        # Initialization strategies
        # 1. All layers zero except last
        init1 = np.zeros(n_params)
        init1[-n:] = theta_mag
        
        # 2. Distributed
        init2 = np.tile(theta_mag / layers, layers)
        
        # 3. Random
        init3 = np.random.uniform(0, 2*np.pi, n_params)
        
    except Exception:
        init1 = np.random.uniform(0, 2*np.pi, n_params)
        init2 = init1.copy()
        init3 = init1.copy()
        
    inits = [init1, init2, init3]
    
    best_params = inits[0]
    best_energy, _ = _circuit_energy_and_grad(best_params, n, coupling, field, layers)
    
    def objective_and_grad(params):
        return _circuit_energy_and_grad(params, n, coupling, field, layers)
        
    # 2. Local Refinement with L-BFGS-B using analytical gradients
    for init in inits:
        if time.time() - start_time > timeout - 5:
            break
        try:
            res = opt.minimize(objective_and_grad, init, jac=True, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    # Check time
    current_time = time.time()
    if current_time - start_time < timeout - 10:
        # Run Differential Evolution (no gradients, but robust)
        # We can use the gradient-based result as the best initial guess for DE
        # or just run DE from scratch with a smaller population to save time.
        
        def de_objective(x):
            if time.time() - start_time > timeout - 5:
                return np.inf
            e, _ = _circuit_energy_and_grad(x, n, coupling, field, layers)
            return e
            
        bounds = [(0, 2*np.pi)] * n_params
        
        try:
            # Use a small population since we have good local minima already
            res_de = opt.minimize(de_objective, best_params, method='differential_evolution',
                                  bounds=bounds, 
                                  maxiter=10, 
                                  popsize=5, 
                                  tol=1e-3,
                                  seed=42,
                                  polish=False)
            if res_de.fun < best_energy:
                best_energy = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
            
    # 3. Final Local Polish
    current_time = time.time()
    if current_time - start_time < timeout - 2:
        try:
            res_lbfgs = opt.minimize(objective_and_grad, best_params, jac=True, method='L-BFGS-B',
                                     options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_lbfgs.fun < best_energy:
                best_energy = res_lbfgs.fun
                best_params = res_lbfgs.x
        except Exception:
            pass
            
    result_angles = best_params.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
