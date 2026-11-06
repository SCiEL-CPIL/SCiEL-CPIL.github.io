"""Optimize shallow Ry/CZ circuit angles to approximate Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute exact ground states and energies for the known test cases
# to avoid recomputation during optimization. We'll compute them on the fly
# but cache them.

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        # Z on qubit i
        Z_i = np.zeros((dim, dim), dtype=np.complex128)
        for state in range(dim):
            bit_i = (state >> i) & 1
            Z_i[state, state] = 1.0 if bit_i == 0 else -1.0
        
        # Z on qubit i+1
        Z_ip1 = np.zeros((dim, dim), dtype=np.complex128)
        for state in range(dim):
            bit_ip1 = (state >> (i + 1)) & 1
            Z_ip1[state, state] = 1.0 if bit_ip1 == 0 else -1.0
        
        H += -coupling * (Z_i @ Z_ip1)
    
    # X_i terms
    for i in range(n):
        X_i = np.zeros((dim, dim), dtype=np.complex128)
        for state in range(dim):
            bit_i = (state >> i) & 1
            if bit_i == 0:
                X_i[state + (1 << i), state] = 1.0
            else:
                X_i[state - (1 << i), state] = 1.0
        
        H += -field * X_i
    
    return H


def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to the given qubit in the state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    dim = len(state)
    n_qubits = int(np.log2(dim))
    new_state = np.zeros_like(state)
    
    for state_idx in range(dim):
        bit = (state_idx >> qubit) & 1
        if bit == 0:
            partner = state_idx + (1 << qubit)
            new_state[state_idx] += c * state[state_idx] - s * state[partner]
            new_state[partner] += s * state[state_idx] + c * state[partner]
        # else: bit == 1, handled when we hit the partner with bit 0
    
    return new_state


def _apply_cz(state, qubit_a, qubit_b):
    """Apply CZ between qubit_a and qubit_b."""
    dim = len(state)
    new_state = state.copy()
    
    for state_idx in range(dim):
        bit_a = (state_idx >> qubit_a) & 1
        bit_b = (state_idx >> qubit_b) & 1
        if bit_a == 1 and bit_b == 1:
            new_state[state_idx] = -state[state_idx]
    
    return new_state


def _circuit_state(theta, n, layers):
    """Apply the circuit to |0>^n and return the final state."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to each qubit i
        for i in range(n):
            state = _apply_ry(state, theta[layer, i], i)
        
        # Apply CZ on every adjacent pair in ascending order (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1)
    
    return state


def _energy(state, H):
    """Compute the expectation value of H in the given state."""
    return np.real(np.vdot(state, H @ state))


# Cache for exact ground states and energies
_exact_cache = {}

def _get_exact_ground(n, coupling, field):
    """Get the exact ground state and energy, caching results."""
    key = (n, coupling, field)
    if key not in _exact_cache:
        H = _build_hamiltonian(n, coupling, field)
        eigvals, eigvecs = np.linalg.eigh(H)
        ground_energy = eigvals[0]
        ground_state = eigvecs[:, 0]
        _exact_cache[key] = (ground_state, ground_energy, H)
    return _exact_cache[key]


def _objective(params, n, coupling, field, layers):
    """Objective function: mean energy gap per spin."""
    theta = params.reshape(layers, n)
    state = _circuit_state(theta, n, layers)
    ground_state, ground_energy, H = _get_exact_ground(n, coupling, field)
    e_circuit = _energy(state, H)
    return (e_circuit - ground_energy) / n


def _optimize_for_case(n, coupling, field, layers, deadline):
    """Optimize angles for a single case using multi-start L-BFGS-B."""
    ground_state, ground_energy, H = _get_exact_ground(n, coupling, field)
    
    best_theta = None
    best_score = np.inf
    
    # Multiple initial guesses
    initial_guesses = []
    
    # 1. Product |+> state (seed)
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    initial_guesses.append(theta0.flatten())
    
    # 2. All zeros
    initial_guesses.append(np.zeros((layers, n)).flatten())
    
    # 3. Random initializations
    np.random.seed(42)
    for _ in range(10):
        initial_guesses.append(np.random.uniform(-np.pi, np.pi, (layers, n)).flatten())
    
    # 4. Heuristic: try small angles
    for scale in [0.1, 0.5, 1.0]:
        theta_h = np.zeros((layers, n))
        theta_h[-1] = scale
        initial_guesses.append(theta_h.flatten())
    
    for x0 in initial_guesses:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                _objective, x0, args=(n, coupling, field, layers),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                best_score = result.fun
                best_theta = result.x.reshape(layers, n)
        except Exception:
            continue
    
    if best_theta is None:
        # Fallback to seed
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2
    
    return best_theta


def angles(n, coupling, field, layers):
    """Optimize circuit angles for the given Ising chain parameters."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to optimize for this specific case
    try:
        theta = _optimize_for_case(n, coupling, field, layers, deadline)
    except Exception:
        # Fallback to seed
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
    
    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    return theta
# EVOLVE-BLOCK-END
