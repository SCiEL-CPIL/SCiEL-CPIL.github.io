"""Quantum information: shallow-circuit Ising ground states - optimized parameters."""
import numpy as np
import time
from scipy.optimize import minimize

# Cache for exact ground energies
_ground_energy_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        # Z_i is diag(1, -1) on qubit i (bit i)
        # We need to build Z_i Z_{i+1}
        # Bit i corresponds to the (n-1-i)th position in binary if we use standard ordering
        # Actually, let's use the convention: qubit i is bit i, so state |b_{n-1}...b_0>
        # Z_i acts on bit i.
        # Let's construct using Kronecker products or direct indexing.
        
        # Direct indexing is faster for small n.
        for state in range(dim):
            # Get bits
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            # Z eigenvalue is +1 if bit is 0, -1 if bit is 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            H[state, state] += -coupling * z_i * z_ip1
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            # X flips bit i
            new_state = state ^ (1 << i)
            H[state, new_state] += -field
    
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key not in _ground_energy_cache:
        H = _build_hamiltonian(n, coupling, field)
        # Hermitian eigenvalues
        eigs = np.linalg.eigvalsh(H)
        _ground_energy_cache[key] = eigs[0].real
    return _ground_energy_cache[key]

def _circuit_state(n, coupling, field, layers, thetas):
    """
    Apply the circuit to |0>^n.
    thetas: shape (layers, n)
    Layer l: Ry(theta) on all qubits, then CZ on adjacent pairs (except after last layer).
    """
    dim = 2 ** n
    # Start with |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            # Ry = [[c, -s], [s, c]]
            # This acts on qubit i.
            # We need to apply this to the state vector.
            # Efficient way: iterate over states where bit i is 0 or 1.
            new_state = np.zeros(dim, dtype=complex)
            for state_idx in range(dim):
                bit_i = (state_idx >> i) & 1
                if bit_i == 0:
                    # |0> -> c|0> + s|1>
                    partner = state_idx | (1 << i)
                    new_state[state_idx] += c * state[state_idx]
                    new_state[partner] += s * state[state_idx]
                else:
                    # |1> -> -s|0> + c|1>
                    partner = state_idx & ~(1 << i)
                    new_state[partner] += -s * state[state_idx]
                    new_state[state_idx] += c * state[state_idx]
            state = new_state
        
        # Apply CZ on adjacent pairs (if not last layer)
        if l < layers - 1:
            for i in range(n - 1):
                # CZ on qubits i and i+1
                # CZ = |0><0|⊗I + |1><1|⊗I - |1><1|⊗|1><1|
                # It flips the sign of |11> for the pair.
                for state_idx in range(dim):
                    bit_i = (state_idx >> i) & 1
                    bit_ip1 = (state_idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[state_idx] *= -1
    
    return state

def _circuit_energy(n, coupling, field, layers, thetas):
    """Compute the expectation value of H in the circuit state."""
    state = _circuit_state(n, coupling, field, layers, thetas)
    H = _build_hamiltonian(n, coupling, field)
    energy = np.vdot(state, H @ state).real
    return energy

def _objective(params, n, coupling, field, layers, exact_energy):
    """Objective function for optimization."""
    thetas = params.reshape(layers, n)
    energy = _circuit_energy(n, coupling, field, layers, thetas)
    # We want to minimize (energy - exact_energy) / n
    # Since exact_energy is constant, minimizing energy is sufficient.
    # But the contract says: mean (circuit energy - exact ground energy)/n
    # Let's return the actual objective value to be safe.
    return (energy - exact_energy) / n

def _optimize_for_config(n, coupling, field, layers, time_limit):
    """Optimize parameters for a specific configuration."""
    exact_energy = _exact_ground_energy(n, coupling, field)
    num_params = layers * n
    
    # Initial guess: random
    np.random.seed(42)
    best_params = np.random.uniform(-np.pi/2, np.pi/2, num_params)
    best_val = _objective(best_params, n, coupling, field, layers, exact_energy)
    
    start_time = time.time()
    
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B is usually faster for smooth functions.
    # Let's try a few restarts.
    
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            x0 = best_params
        else:
            x0 = np.random.uniform(-np.pi, np.pi, num_params)
        
        # Try L-BFGS-B
        try:
            res = minimize(_objective, x0, args=(n, coupling, field, layers, exact_energy),
                           method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success or res.fun < best_val:
                best_params = res.x
                best_val = res.fun
        except Exception:
            pass
        
        # Also try Nelder-Mead for robustness
        try:
            res_nm = minimize(_objective, x0, args=(n, coupling, field, layers, exact_energy),
                              method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm.fun < best_val:
                best_params = res_nm.x
                best_val = res_nm.fun
        except Exception:
            pass
    
    return best_params, best_val

def angles(n, coupling, field, layers):
    # Deadline guard
    start_time = time.time()
    time_limit = 140.0  # 150s limit, keep 10s margin
    
    # Optimize
    params, val = _optimize_for_config(n, coupling, field, layers, time_limit)
    
    # Reshape to (layers, n)
    thetas = params.reshape(layers, n)
    
    return thetas
