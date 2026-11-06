"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Vectorized circuit simulation with multi-start L-BFGS-B local optimization.
This avoids the global search overhead of differential evolution by using a physically
motivated initialization (product state approximation) and fast gradient-based local
optimization with multiple restarts to escape local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

# Cache for exact ground energies and Hamiltonians
_HAM_CACHE = {}
_EE_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    
    # H_IZ = sum Z_i Z_{i+1}
    H_IZ = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_IZ[state, state] -= 1.0
            else:
                H_IZ[state, state] += 1.0
    
    # H_IX = sum X_i
    H_IX = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_IX[state, flipped] = 1.0
            
    _HAM_CACHE[key] = (H_IZ, H_IX)
    return _HAM_CACHE[key]

def _compute_exact_ground_energy(n, coupling, field):
    """Compute the exact ground state energy."""
    key = (n, coupling, field)
    if key in _EE_CACHE:
        return _EE_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # Exact diagonalization
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0].real
    _EE_CACHE[key] = e0
    return e0

def _apply_circuit(theta, n):
    """
    Apply the circuit to |0...0> and return the state vector.
    theta: array of shape (layers, n)
    Vectorized implementation using reshape for efficiency.
    """
    dim = 1 << n
    layers = theta.shape[0]
    
    # Start with |0...0>
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            # Reshape state to tensor [2]*n where axis i corresponds to qubit i
            state_tensor = state.reshape([2]*n)
            
            # Extract the two components along axis i
            # comp0: states where bit i is 0
            # comp1: states where bit i is 1
            comp0 = np.take(state_tensor, 0, axis=i)
            comp1 = np.take(state_tensor, 1, axis=i)
            
            # Apply Ry: 
            # new_comp0 = c*comp0 - s*comp1
            # new_comp1 = s*comp0 + c*comp1
            new_comp0 = c * comp0 - s * comp1
            new_comp1 = s * comp0 + c * comp1
            
            # Update the tensor
            np.put_along_axis(state_tensor, 0, new_comp0, axis=i)
            np.put_along_axis(state_tensor, 1, new_comp1, axis=i)
            
            state = state_tensor.reshape(-1)
        
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                # CZ: |11> -> -|11>
                # Only affects states where bit i and bit i+1 are both 1.
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        state[idx] *= -1.0
                        
    return state

def _compute_energy(theta, n, coupling, field):
    """Compute the energy of the state produced by the circuit."""
    state = _apply_circuit(theta, n)
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # <psi| H |psi>
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    return energy

def _objective(params, n, coupling, field, e_exact):
    """Objective function for the optimizer."""
    theta = params.reshape(3, n)
    e_circ = _compute_energy(theta, n, coupling, field)
    return e_circ - e_exact

def _mean_field_initial_guess(n, coupling, field):
    """
    Generate initial guess based on mean-field theory.
    For large field, |+> is good (theta ~ pi/2).
    For large coupling, |0> is good (theta ~ 0).
    """
    # Simple interpolation
    # At field=0, coupling dominates -> theta=0
    # At field>>coupling, field dominates -> theta=pi/2
    ratio = field / max(coupling, 1e-10)
    # tanh interpolation
    theta_val = (np.pi / 2) * np.tanh(ratio)
    
    theta = np.zeros((3, n))
    theta[0, :] = theta_val
    theta[1, :] = theta_val
    theta[2, :] = theta_val
    return theta

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Compute exact ground energy
    e_exact = _compute_exact_ground_energy(n, coupling, field)
    
    # 2. Define bounds
    # Angles are in radians. Ry(theta) is periodic with 2pi.
    # We can restrict to [-pi, pi] or [0, 2pi].
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # 3. Generate multiple initial guesses
    initial_guesses = []
    
    # Mean-field guess
    theta_mf = _mean_field_initial_guess(n, coupling, field)
    initial_guesses.append(theta_mf.flatten())
    
    # Uniform guesses
    initial_guesses.append(np.zeros(layers * n))
    initial_guesses.append(np.full(layers * n, np.pi / 2))
    initial_guesses.append(np.full(layers * n, np.pi / 4))
    
    # Random guesses
    rng = np.random.RandomState(42)
    for _ in range(5):
        initial_guesses.append(rng.uniform(-np.pi, np.pi, size=layers * n))
    
    # 4. Optimize from each initial guess using L-BFGS-B
    best_x = initial_guesses[0]
    best_val = _objective(best_x, n, coupling, field, e_exact)
    
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(n, coupling, field, e_exact),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
                
        except Exception:
            continue
    
    # Final check: if time is running out, return best found.
    # Convert back to (layers, n) array
    theta_opt = best_x.reshape(layers, n)
    
    return theta_opt

# EVOLVE-BLOCK-END
