import numpy as np
import scipy.optimize as opt
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    idx = np.arange(dim)
    
    # ZZ term: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for k in range(dim):
            bit_i = (k >> i) & 1
            bit_ip1 = (k >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            H[k, k] += -coupling * z_i * z_ip1
            
    # X term: -field * sum X_i
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            k_flipped = k ^ mask
            H[k, k_flipped] += -field
            H[k_flipped, k] += -field
            
    return H

def _circuit_energy(params, n, layers, coupling, field, H):
    """
    Compute the energy expectation value for the given circuit parameters.
    This is the core function that will be differentiated.
    """
    thetas = params.reshape(layers, n)
    
    # State as a tensor for easy gate application
    # Index mapping: state[b_0, b_1, ..., b_{n-1}]
    # Basis index k = sum b_i 2^i
    # Numpy reshape of flat vector: first axis is MSB (bit n-1)
    # So qubit i (bit i) corresponds to axis n-1-i
    
    state = np.zeros([2] * n, dtype=complex)
    state[tuple([0] * n)] = 1.0
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            axis = n - 1 - i
            state = np.moveaxis(state, axis, -1)
            
            old_0 = state[..., 0]
            old_1 = state[..., 1]
            
            # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            new_0 = c * old_0 - s * old_1
            new_1 = s * old_0 + c * old_1
            
            new_state_slice = np.empty_like(state)
            new_state_slice[..., 0] = new_0
            new_state_slice[..., 1] = new_1
            
            state = np.moveaxis(new_state_slice, -1, axis)
        
        # Apply CZ if not the last layer
        if l < layers - 1:
            flat_state = state.reshape(-1)
            dim = 2 ** n
            idx = np.arange(dim)
            
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                flat_state[cond] *= -1
            
            state = flat_state.reshape([2] * n)
    
    state_flat = state.reshape(-1)
    energy = np.real(np.vdot(state_flat, H @ state_flat))
    return energy

def _optimize(n, coupling, field, layers, timeout=145.0):
    start_time = time.time()
    dim = 2 ** n
    H = _build_hamiltonian(n, coupling, field)
    
    # Compute exact ground energy for reference (not used in optimization, but good for sanity)
    # eigenvalues, eigenvectors = np.linalg.eigh(H)
    # ground_energy = eigenvalues[0]
    
    n_params = n * layers
    
    # Initial guess: heuristic based on mean field or just random
    # For transverse field Ising, in the strong field limit, ground state is close to |+>^n
    # |+> corresponds to Ry(pi/2) on |0>
    # In the strong coupling limit, ground state is close to |0>^n or |1>^n (depending on boundary)
    # Let's use a mix of random and heuristic starts.
    
    def objective(params):
        return _circuit_energy(params, n, layers, coupling, field, H)
    
    # Use multiple random restarts with L-BFGS-B
    # This is more robust than a single global optimizer like DE for low-dimensional smooth landscapes
    # Especially with accurate gradients.
    
    best_energy = np.inf
    best_params = np.random.uniform(0, 2 * np.pi, n_params)
    
    bounds = [(0, 2 * np.pi)] * n_params
    
    # Heuristic initial guess
    # For field > 0, <X> > 0, so theta should be around pi/2
    # For coupling > 0, <ZZ> > 0, so <Z> > 0, so theta should be small (near 0)
    # This is a tension.
    # Let's just use a few random starts and one heuristic.
    
    inits = []
    # Heuristic: all pi/4 (compromise between 0 and pi/2)
    inits.append(np.full(n_params, np.pi / 4))
    # Heuristic: all pi/2 (field dominated)
    inits.append(np.full(n_params, np.pi / 2))
    # Heuristic: all 0 (coupling dominated)
    inits.append(np.zeros(n_params))
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(0, 2 * np.pi, n_params))
    
    for init in inits:
        if time.time() - start_time > timeout - 10:
            break
            
        try:
            res = opt.minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass
            
    result_angles = best_params.reshape(layers, n)
    return result_angles

def angles(n, coupling, field, layers):
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
