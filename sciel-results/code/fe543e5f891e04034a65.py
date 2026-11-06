"""Quantum information: shallow-circuit Ising ground states - optimized parameters."""
import numpy as np
import time
from scipy.optimize import minimize

# Cache for exact ground energies
_ground_energy_cache = {}
_H_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i."""
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            H[state, state] += -coupling * z_i * z_ip1
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            new_state = state ^ (1 << i)
            H[state, new_state] += -field
    
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key not in _ground_energy_cache:
        H = _build_hamiltonian(n, coupling, field)
        eigs = np.linalg.eigvalsh(H)
        _ground_energy_cache[key] = eigs[0].real
    return _ground_energy_cache[key]

def _circuit_state(n, layers, thetas):
    """
    Apply the circuit to |0>^n.
    thetas: shape (layers, n)
    Layer l: Ry(theta) on all qubits, then CZ on adjacent pairs (except after last layer).
    """
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Apply Ry on qubit i using vectorized operations
            # Create masks for states where bit i is 0 or 1
            # We'll iterate over blocks to be efficient
            new_state = np.zeros_like(state)
            
            for base in range(0, dim, 2 ** (i + 1)):
                for offset in range(2 ** i):
                    idx0 = base + offset
                    idx1 = idx0 + 2 ** i
                    
                    # |0> -> c|0> + s|1>
                    new_state[idx0] += c * state[idx0]
                    new_state[idx1] += s * state[idx0]
                    
                    # |1> -> -s|0> + c|1>
                    new_state[idx0] += -s * state[idx1]
                    new_state[idx1] += c * state[idx1]
            
            state = new_state
        
        # Apply CZ on adjacent pairs (if not last layer)
        if l < layers - 1:
            for i in range(n - 1):
                # CZ flips sign of |11> for qubits i and i+1
                for base in range(0, dim, 2 ** (i + 2)):
                    for offset in range(2 ** i):
                        idx11 = base + offset + 2 ** i + 2 ** (i + 1)
                        state[idx11] *= -1
    
    return state

def _circuit_energy(n, coupling, field, layers, thetas):
    """Compute the expectation value of H in the circuit state."""
    state = _circuit_state(n, layers, thetas)
    H = _build_hamiltonian(n, coupling, field)
    energy = state @ H @ state
    return energy

def _objective(params, n, coupling, field, layers, exact_energy):
    """Objective function for optimization."""
    thetas = params.reshape(layers, n)
    energy = _circuit_energy(n, coupling, field, layers, thetas)
    return (energy - exact_energy) / n

def _optimize_for_config(n, coupling, field, layers, time_limit):
    """Optimize parameters for a specific configuration."""
    exact_energy = _exact_ground_energy(n, coupling, field)
    num_params = layers * n
    
    start_time = time.time()
    
    # Multiple random restarts with L-BFGS-B
    best_params = None
    best_val = np.inf
    
    np.random.seed(123)
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        if restart == 0:
            # Start with a reasonable initial guess
            x0 = np.zeros(num_params)
            for i in range(n):
                x0[i] = 0.3  # Small positive angles
        elif restart == 1:
            x0 = np.random.uniform(-np.pi/4, np.pi/4, num_params)
        else:
            x0 = np.random.uniform(-np.pi, np.pi, num_params)
        
        try:
            res = minimize(_objective, x0, args=(n, coupling, field, layers, exact_energy),
                           method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
                
                # If we found a very good solution, do a polish pass
                if best_val < 0.1:
                    if time.time() - start_time > time_limit - 10:
                        break
                    res2 = minimize(_objective, best_params, args=(n, coupling, field, layers, exact_energy),
                                   method='L-BFGS-B',
                                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000})
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_params = res2.x.copy()
                        
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.zeros(num_params)
        best_val = _objective(best_params, n, coupling, field, layers, exact_energy)
    
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
