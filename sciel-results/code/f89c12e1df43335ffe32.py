import numpy as np
import scipy.optimize as opt
import time

_cache = {}

def _simulate_state_fast(params, n, layers):
    """
    Simulate the circuit and return the final state vector.
    Uses matrix operations where possible for speed.
    """
    thetas = params.reshape(layers, n)
    
    # State as a tensor
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
            
            # Vectorized CZ application
            # CZ flips sign if both qubits are 1
            # We can do this by iterating pairs, but vectorizing over all pairs is tricky with masks.
            # However, n is small (4 or 6).
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                flat_state[cond] *= -1
            
            state = flat_state.reshape([2] * n)
            
    return state.reshape(-1)

def _compute_energy(state_flat, n, coupling, field):
    """
    Compute <H> for the given state vector.
    """
    dim = 2 ** n
    idx = np.arange(dim)
    probs = np.abs(state_flat) ** 2
    
    # ZZ Term
    E_zz = 0.0
    for i in range(n - 1):
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = z_i * z_ip1
        E_zz += np.sum(probs * zz_val)
        
    E_1 = -coupling * E_zz
    
    # X Term
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        term = np.sum(np.conj(state_flat) * state_flat[idx_flipped])
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
    
    # 1. Get Exact Ground State Properties
    try:
        dim = 2 ** n
        H = np.zeros((dim, dim), dtype=complex)
        idx = np.arange(dim)
        
        # Build H
        for i in range(n - 1):
            mask_i = 1 << i
            mask_ip1 = 1 << (i + 1)
            for k in range(dim):
                bit_i = (k >> i) & 1
                bit_ip1 = (k >> (i + 1)) & 1
                z_i = 1 - 2 * bit_i
                z_ip1 = 1 - 2 * bit_ip1
                H[k, k] += -coupling * z_i * z_ip1
                
        for i in range(n):
            mask = 1 << i
            for k in range(dim):
                k_flipped = k ^ mask
                H[k, k_flipped] += -field
                H[k_flipped, k] += -field
                
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        ground_state = eigenvectors[:, 0]
        ground_energy = eigenvalues[0]
        
        # Compute local magnetizations of ground state
        probs = np.abs(ground_state) ** 2
        mags = np.zeros(n)
        for i in range(n):
            mask = 1 << i
            bit_vals = (idx >> i) & 1
            z_vals = 1 - 2 * bit_vals
            mags[i] = np.sum(probs * z_vals)
            
        # Heuristic Initial Guess:
        # The last layer of Ry gates determines the local Z magnetization of the output state
        # IF the state before the last layer were a product state.
        # Even with entanglement, aligning the last layer to the ground state's local magnetizations
        # is a strong prior.
        theta_last = np.arccos(np.clip(mags, -1, 1))
        
        # Construct initial parameter array
        # We assume the first layers do the "entanglement work" and the last layer does the "angle work".
        # Let's try a few initializations:
        # 1. Last layer = theta_last, others = 0
        init_1 = np.zeros(n_params)
        init_1[-n:] = theta_last
        
        # 2. All layers = theta_last / layers (distributed)
        init_2 = np.tile(theta_last / layers, layers)
        
        # 3. Random
        init_3 = np.random.uniform(0, 2 * np.pi, n_params)
        
        inits = [init_1, init_2, init_3]
        
    except Exception:
        inits = [np.random.uniform(0, 2 * np.pi, n_params)]
        
    def objective(params):
        state = _simulate_state_fast(params, n, layers)
        return _compute_energy(state, n, coupling, field)
        
    best_params = inits[0]
    best_energy = objective(best_params)
    
    # Use Nelder-Mead for robustness (no gradient needed, handles non-smoothness better)
    # We will run multiple Nelder-Mead optimizations from different starts.
    
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        try:
            res = opt.minimize(
                objective, 
                init, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    # If time remains, do a few more random restarts with Nelder-Mead
    current_time = time.time()
    remaining = timeout - current_time
    if remaining > 10:
        # Nelder-Mead is slower per iteration but doesn't need gradient computation.
        # Estimate cost: ~500 function evals.
        # Let's do a few more random starts if we have time.
        n_extra = int(min(remaining / 5, 5))
        for _ in range(n_extra):
            if time.time() - start_time > timeout:
                break
            init = np.random.uniform(0, 2 * np.pi, n_params)
            try:
                res = opt.minimize(
                    objective, 
                    init, 
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_params = res.x
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
