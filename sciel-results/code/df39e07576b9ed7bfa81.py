"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def _apply_ry(state, n, i, theta):
    """Apply Ry(theta) to qubit i of the state vector (vectorized)."""
    dim = 1 << n
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    state_tensor = state.reshape((2,) * n)
    axis = n - 1 - i
    
    state_tensor = np.moveaxis(state_tensor, axis, 0)
    
    new_state_tensor = np.empty_like(state_tensor)
    new_state_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
    new_state_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
    
    new_state_tensor = np.moveaxis(new_state_tensor, 0, axis)
    
    return new_state_tensor.reshape(-1)

def _apply_cz(state, n, i):
    """Apply CZ between qubit i and i+1."""
    dim = 1 << n
    indices = np.arange(dim)
    shift_i = n - 1 - i
    shift_j = n - 1 - (i + 1)
    
    bit_i = (indices >> shift_i) & 1
    bit_j = (indices >> shift_j) & 1
    
    mask = (bit_i == 1) & (bit_j == 1)
    
    state = state.copy()
    state[mask] *= -1
    
    return state

def _build_state(theta, n, layers):
    """Build the circuit state from angles."""
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0...0>
    
    for layer in range(layers):
        for i in range(n):
            state = _apply_ry(state, n, i, theta[layer, i])
        
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, n, i)
                
    return state

def _compute_energy(state, n, coupling, field):
    """Compute <H> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i"""
    dim = 1 << n
    energy = 0.0
    probs = np.abs(state) ** 2
    
    indices = np.arange(dim)
    
    for i in range(n - 1):
        shift_i = n - 1 - i
        shift_j = n - 1 - (i + 1)
        z_i = np.where((indices >> shift_i) & 1, -1.0, 1.0)
        z_j = np.where((indices >> shift_j) & 1, -1.0, 1.0)
        zz_vals = -coupling * z_i * z_j
        energy += np.sum(zz_vals * probs)
        
    for i in range(n):
        shift = n - 1 - i
        flipped_indices = indices ^ (1 << shift)
        cross_terms = np.conj(state) * state[flipped_indices]
        energy += -field * np.sum(cross_terms).real
        
    return energy

def _objective(params, n, coupling, field, layers):
    """Objective function for optimization."""
    theta = params.reshape(layers, n)
    state = _build_state(theta, n, layers)
    energy = _compute_energy(state, n, coupling, field)
    return energy

def _physics_init(n, field, layers):
    """Physics-based initialization."""
    theta = np.zeros((layers, n))
    frac = min(field / 2.0, 1.0)
    base_angle = frac * np.pi / 2
    theta[:, :] = base_angle
    return theta

def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate Ising ground state."""
    start_time = time.time()
    deadline = 118.0  # Safety margin before 120s kill
    
    # Generate diverse initializations
    np.random.seed(42)  # Reproducible
    inits = []
    
    # Physics-based init
    theta_phys = _physics_init(n, field, layers)
    inits.append(theta_phys.flatten())
    
    # All zeros
    inits.append(np.zeros(layers * n))
    
    # All pi/2
    inits.append(np.full(layers * n, np.pi / 2))
    
    # Random inits with different scales
    for scale in [0.3, 0.7, 1.0, 1.5]:
        inits.append(np.random.uniform(-scale, scale, size=layers * n))
    
    # Perturbations around physics init
    for scale in [0.1, 0.3, 0.5]:
        inits.append(theta_phys.flatten() + np.random.normal(0, scale, size=layers * n))
    
    best_params = inits[0].copy()
    best_energy = _objective(best_params, n, coupling, field, layers)
    
    # Optimize from each initialization
    for idx, init in enumerate(inits):
        if time.time() - start_time > deadline - 5:
            break
        
        try:
            result = scipy.optimize.minimize(
                _objective, 
                init, 
                args=(n, coupling, field, layers),
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()
                
        except Exception:
            pass
    
    # Final polish: run L-BFGS-B from best with tighter tolerances
    if time.time() - start_time < deadline - 3:
        try:
            result_final = scipy.optimize.minimize(
                _objective, 
                best_params, 
                args=(n, coupling, field, layers),
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result_final.fun < best_energy:
                best_energy = result_final.fun
                best_params = result_final.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(layers, n)
# EVOLVE-BLOCK-END
