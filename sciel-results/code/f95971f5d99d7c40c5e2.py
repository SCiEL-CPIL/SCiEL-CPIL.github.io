"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_j = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_j = 1 if bit_j == 0 else -1
            H[state, state] += -coupling * z_i * z_j
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            flip = state ^ (1 << i)
            H[state, flip] += -field
    
    return H

def _get_ground_energy(n, coupling, field):
    """Compute the exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigs = np.linalg.eigvalsh(H)
    return float(eigs[0])

def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[cos, -sin], [sin, cos]]
    # We need to apply this to the qubit
    dim = 1 << len(state)
    new_state = np.zeros_like(state)
    for idx in range(dim):
        bit = (idx >> qubit) & 1
        # The other bits
        other = idx ^ (bit << qubit)
        # If bit is 0, |0> -> cos|0> + sin|1>
        # If bit is 1, |1> -> -sin|0> + cos|1>
        if bit == 0:
            new_state[idx] += c * state[idx]
            new_state[idx ^ (1 << qubit)] += s * state[idx]
        else:
            new_state[idx ^ (1 << qubit)] += -s * state[idx]
            new_state[idx] += c * state[idx]
    return new_state

def _apply_cz(state, q1, q2):
    """Apply CZ gate between q1 and q2."""
    dim = 1 << len(state)
    new_state = state.copy()
    for idx in range(dim):
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        if b1 == 1 and b2 == 1:
            new_state[idx] *= -1
    return new_state

def _compute_energy(state, n, coupling, field):
    """Compute <psi|H|psi>."""
    dim = 1 << n
    # Compute Z_i Z_{i+1}
    zz_energy = 0.0
    for i in range(n - 1):
        for idx in range(dim):
            b1 = (idx >> i) & 1
            b2 = (idx >> (i + 1)) & 1
            z1 = 1 if b1 == 0 else -1
            z2 = 1 if b2 == 0 else -1
            zz_energy += -coupling * z1 * z2 * (np.abs(state[idx])**2)
    
    # Compute X_i
    x_energy = 0.0
    for i in range(n):
        for idx in range(dim):
            flip = idx ^ (1 << i)
            x_energy += -field * (np.conj(state[idx]) * state[flip])
    
    total = zz_energy + x_energy
    return float(np.real(total))

def _circuit_energy(params, n, coupling, field, layers):
    """Compute the energy of the state produced by the circuit."""
    theta = params.reshape(layers, n)
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            state = _apply_ry(state, theta[layer, q], q)
        
        # Apply CZ between adjacent pairs (if not the last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1)
    
    return _compute_energy(state, n, coupling, field)

def _objective(params, n, coupling, field, layers, ground_energy):
    """Objective: mean energy gap per spin."""
    e_circuit = _circuit_energy(params, n, coupling, field, layers)
    return (e_circuit - ground_energy) / n

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate 
    ground states of open transverse-field Ising chains.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Get exact ground energy
    ground_energy = _get_ground_energy(n, coupling, field)
    
    # Initial guess: try to mimic product state
    # For small field, ground state is close to |0>^n
    # For large field, ground state is close to |+>^n
    # Intermediate: some mix
    
    # Multiple restarts with different initial guesses
    best_params = None
    best_score = np.inf
    
    # Strategy 1: Start from |+> state (all pi/2)
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2  # Last layer only, to get |+>
    
    # Strategy 2: Start from |0> state (all 0)
    theta1 = np.zeros((layers, n))
    
    # Strategy 3: Start from intermediate
    theta2 = np.zeros((layers, n))
    theta2[-1] = np.pi / 4
    
    # Strategy 4: Random
    theta3 = np.random.uniform(-np.pi/4, np.pi/4, (layers, n))
    
    initial_guesses = [theta0.flatten(), theta1.flatten(), theta2.flatten(), theta3.flatten()]
    
    for init in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective,
                init,
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    # If we still have time, do a final polish with tighter tolerance
    if time.time() < deadline - 10.0 and best_params is not None:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_score:
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: return |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_params.reshape(layers, n)
# EVOLVE-BLOCK-END
