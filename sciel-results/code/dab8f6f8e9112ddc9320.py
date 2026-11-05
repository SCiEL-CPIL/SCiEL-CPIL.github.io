import numpy as np
from scipy.optimize import minimize
import time

# Cache for computed angles
_angle_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix for open chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zz = 1 if bit_i == bit_ip1 else -1
            H[state, state] += -coupling * zz
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            # X flips bit i
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
            H[state, flipped] += -field  # X is Hermitian, already covered by flip
    
    # Actually, let me redo this more carefully
    # X_i |state> = |state with bit i flipped>
    # So <flipped|X_i|state> = 1, meaning H[flipped, state] += -field
    # But since we're building the matrix element H[a,b] = <a|H|b>
    # X_i maps |b> to |a> where a = b with bit i flipped
    # So H[a, b] += -field where a = b ^ (1<<i)
    
    # Let me rebuild properly
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms: diagonal
    for state in range(dim):
        energy = 0.0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zz = 1.0 if bit_i == bit_ip1 else -1.0
            energy += -coupling * zz
        H[state, state] += energy
    
    # X terms: off-diagonal
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H

def _apply_circuit(state_vec, angles, n):
    """Apply the 3-layer Ry/CZ circuit to the state vector."""
    layers = angles.shape[0]
    # state_vec is a 1D array of length 2^n
    # Qubit i is bit i (little-endian)
    
    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            theta = angles[layer, i]
            # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            # Apply to qubit i
            new_state = np.zeros_like(state_vec)
            for state in range(len(state_vec)):
                bit_i = (state >> i) & 1
                if bit_i == 0:
                    # |0> -> c|0> + s|1>
                    partner = state | (1 << i)
                    new_state[state] += c * state_vec[state]
                    new_state[partner] += s * state_vec[state]
                else:
                    # |1> -> -s|0> + c|1>
                    partner = state & ~(1 << i)
                    new_state[partner] += -s * state_vec[state]
                    new_state[state] += c * state_vec[state]
            state_vec = new_state
        
        # Apply CZ between adjacent pairs (if not the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ on qubits i and i+1
                # CZ flips phase when both bits are 1
                for state in range(len(state_vec)):
                    bit_i = (state >> i) & 1
                    bit_ip1 = (state >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state_vec[state] *= -1
    
    return state_vec

def _energy_from_angles(angles, n, coupling, field, H, ground_energy):
    """Compute the energy of the state produced by the circuit."""
    # Start state |0>^n
    dim = 1 << n
    state_vec = np.zeros(dim, dtype=np.complex128)
    state_vec[0] = 1.0  # |0>^n
    
    state_vec = _apply_circuit(state_vec, angles, n)
    
    # Compute energy = <psi|H|psi>
    energy = np.vdot(state_vec, H @ state_vec).real
    return energy

def angles(n, coupling, field, layers):
    """Return optimized angles for the given parameters."""
    cache_key = (n, coupling, field, layers)
    if cache_key in _angle_cache:
        return _angle_cache[cache_key]
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    
    # Find exact ground energy
    eigvals = np.linalg.eigvalsh(H)
    ground_energy = eigvals[0]
    
    dim = 1 << n
    n_params = layers * n
    
    # Initial guess: all zeros (product |0> state) or pi/2 (product |+> state)
    # Let's try a few initial guesses
    initial_guesses = [
        np.zeros(n_params),
        np.full(n_params, np.pi / 2),
        np.random.uniform(-np.pi, np.pi, n_params) * 0.1,
    ]
    
    best_energy = np.inf
    best_params = None
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        x0 = guess.reshape(layers, n)
        
        def objective(x):
            if time.time() > deadline:
                return 1e10
            a = x.reshape(layers, n)
            return _energy_from_angles(a, n, coupling, field, H, ground_energy)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             bounds=[(-np.pi, np.pi)] * n_params,
                             options={'maxiter': 50, 'ftol': 1e-10})
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: use the seed's approach
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _angle_cache[cache_key] = theta
        return theta
    
    _angle_cache[cache_key] = best_params
    return best_params
