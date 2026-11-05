"""Optimized program: gate-by-gate state evolution with L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

# Precompute exact ground energies for the known cases
# H = -coupling * sum Z_i Z_{i+1} - field * sum X_i, open chain

def _build_hamiltonian(n, coupling, field):
    """Build the sparse Hamiltonian matrix."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zz = 1.0 if bit_i == bit_ip1 else -1.0
            H[state, state] += -coupling * zz
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    
    return H


def _get_ground_energy(n, coupling, field):
    """Compute the exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


# Precompute ground energies for known cases
_GROUND_ENERGY_CACHE = {}

def _ground_energy_cached(n, coupling, field):
    key = (n, coupling, field)
    if key not in _GROUND_ENERGY_CACHE:
        _GROUND_ENERGY_CACHE[key] = _get_ground_energy(n, coupling, field)
    return _GROUND_ENERGY_CACHE[key]


def _apply_ry(state, qubit, theta):
    """Apply Ry(theta) to qubit of state vector using tensor operations."""
    n = int(np.log2(len(state)))
    # Reshape to tensor
    tensor = state.reshape([2] * n)
    
    # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    
    # Apply to qubit axis
    tensor = np.tensordot(ry, tensor, axes=[1, qubit])
    # Move axis back
    tensor = np.transpose(tensor, list(range(1, n)) + [0])
    
    return tensor.reshape(-1)


def _apply_cz(state, q1, q2):
    """Apply CZ gate between qubits q1 and q2."""
    n = int(np.log2(len(state)))
    tensor = state.reshape([2] * n)
    
    # CZ = |0><0| ⊗ I + |1><1| ⊗ Z
    # Apply Z to q2 conditioned on q1
    # For each value of q1:
    # If q1=0: identity on q2
    # If q1=1: flip sign of q2=1 component
    
    # Move q1 and q2 to first two axes
    axes = list(range(n))
    axes.remove(q1)
    axes.remove(q2)
    axes = [q1, q2] + axes
    
    tensor = np.transpose(tensor, axes)
    
    # Apply Z to q2 when q1=1
    tensor[1, 1] *= -1
    
    # Move back
    tensor = np.transpose(tensor, np.argsort(axes))
    
    return tensor.reshape(-1)


def _circuit_state(params, n, layers):
    """Compute the state vector from circuit parameters."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[layer, q]
            state = _apply_ry(state, q, theta)
        
        # Apply CZ between adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1)
    
    return state


def _energy(params, n, coupling, field, layers, H):
    """Compute the energy expectation value."""
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state).real
    return energy


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Get exact ground energy
    e0 = _ground_energy_cached(n, coupling, field)
    
    # Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    
    # Objective: mean energy gap per spin = (E_circuit - E_0) / n
    def objective(params_flat):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return np.inf
        params = params_flat.reshape(layers, n)
        e = _energy(params, n, coupling, field, layers, H)
        return (e - e0) / n
    
    # Bounds: theta in [-pi, pi]
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Initial guess: try several starting points
    best_params = None
    best_energy = np.inf
    
    # Try a few initial points
    init_candidates = [
        np.zeros(layers * n),
        np.pi / 2 * np.ones(layers * n),
        np.random.randn(layers * n) * 0.5,
        np.random.randn(layers * n) * 0.5,
    ]
    
    for init in init_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        try:
            result = minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy and np.isfinite(result.fun):
                best_energy = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # If no optimization succeeded, use the |+> state
    if best_params is None:
        params = np.zeros((layers, n))
        params[-1] = np.pi / 2
        return params
    
    return best_params.reshape(layers, n)
