"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0
_START_TIME = time.time()


def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i using Kronecker products."""
    total_dim = 2 ** n
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    H = np.zeros((total_dim, total_dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i or q == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    # X terms
    for i in range(n):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    return H


def _get_ground_energy(H):
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real


def _circuit_state(params, n, layers):
    """Build circuit state using efficient tensor operations."""
    total_dim = 2 ** n
    state = np.zeros(total_dim, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            new_state = np.zeros_like(state)
            for idx in range(total_dim):
                bit = (idx >> q) & 1
                if bit == 0:
                    new_state[idx] += c * state[idx]
                    flipped = idx | (1 << q)
                    new_state[flipped] += s * state[idx]
                else:
                    new_state[idx] += c * state[idx]
                    flipped = idx ^ (1 << q)
                    new_state[flipped] += -s * state[idx]
            state = new_state
        
        # Apply CZ between adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                for idx in range(total_dim):
                    b1 = (idx >> q) & 1
                    b2 = (idx >> (q + 1)) & 1
                    if b1 == 1 and b2 == 1:
                        state[idx] *= -1.0
    
    return state


def _circuit_energy(params, n, coupling, field, layers, H):
    state = _circuit_state(params, n, layers)
    return np.vdot(state, H @ state).real


def _objective(params_flat, n, coupling, field, layers, H, ground_energy):
    params = params_flat.reshape(layers, n)
    energy = _circuit_energy(params, n, coupling, field, layers, H)
    return (energy - ground_energy) / n


def _heuristic_initial(n, field, layers):
    """Generate a physics-inspired initial guess."""
    params = np.zeros((layers, n))
    # For transverse-field Ising, optimal angles are often near pi/2 for low field
    # and smaller for high field
    base_angle = np.arctan(1.0 / max(field, 0.1))
    for layer in range(layers):
        for q in range(n):
            # Slight variation per layer and qubit
            params[layer, q] = base_angle + 0.1 * (layer - 1) + 0.05 * (q - n / 2)
    return params


def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(H)
    
    best_params = _heuristic_initial(n, field, layers)
    best_score = _objective(best_params.flatten(), n, coupling, field, layers, H, ground_energy)
    
    # Try multiple initial points with different strategies
    np.random.seed(12345)
    
    initial_points = []
    # Heuristic
    initial_points.append(_heuristic_initial(n, field, layers).flatten())
    # Zeros
    initial_points.append(np.zeros(layers * n))
    # Random
    for _ in range(3):
        initial_points.append(np.random.uniform(-np.pi, np.pi, size=layers * n))
    # Small random perturbations around heuristic
    for _ in range(2):
        pert = _heuristic_initial(n, field, layers).flatten() + np.random.normal(0, 0.3, size=layers * n)
        initial_points.append(pert)
    
    for trial, x0 in enumerate(initial_points):
        if not _check_deadline():
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.reshape(layers, n)
                
        except Exception:
            continue
    
    return best_params
# EVOLVE-BLOCK-END
