# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:
                H[b, b_flip] += -field
                H[b_flip, b] += -field
    
    return H

def _precompute_indices(n):
    """Precompute index arrays for efficient gate application."""
    dim = 1 << n
    ry_indices = {}
    for qubit in range(n):
        mask = 1 << qubit
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
        idx1 = idx0 | mask
        ry_indices[qubit] = (idx0, idx1)
    
    cz_indices = {}
    for q1 in range(n - 1):
        q2 = q1 + 1
        mask1 = 1 << q1
        mask2 = 1 << q2
        both_set = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
        cz_indices[(q1, q2)] = both_set
    
    return ry_indices, cz_indices

def _circuit_state(params, n, layers, ry_indices, cz_indices):
    """Prepare the circuit state from parameters.
    params: flat array of shape (layers * n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer * n + q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_indices[q]
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
        
        # Apply CZ on adjacent pairs (not after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                both_set = cz_indices[(q, q + 1)]
                state[both_set] = -state[both_set]
    
    return state

def _energy(state, H):
    """Compute <state|H|state>."""
    return np.real(np.vdot(state, H @ state))

def _objective(params, n, coupling, field, layers, H, E0, ry_indices, cz_indices):
    """Objective: mean energy gap per spin = (E_circuit - E0) / n."""
    state = _circuit_state(params, n, layers, ry_indices, cz_indices)
    E = _energy(state, H)
    return (E - E0) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Build Hamiltonian and find exact ground state
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    E0 = eigvals[0]
    
    # Precompute indices
    ry_indices, cz_indices = _precompute_indices(n)
    
    # Initial guess: all zeros
    best_params = np.zeros(layers * n)
    best_val = _objective(best_params, n, coupling, field, layers, H, E0, ry_indices, cz_indices)
    
    # Try multiple initializations
    np.random.seed(42)
    
    # Compute a physically motivated base angle
    ratio = field / coupling
    if ratio < 1.0:
        base_angle = 0.5 * np.arctan(ratio)
    elif ratio < 2.0:
        base_angle = np.pi / 4
    else:
        base_angle = 0.5 * (np.pi / 2 + np.arctan(1.0 / (ratio - 1)))
    
    n_restarts = 10 if n == 4 else 6
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            params0 = np.zeros(layers * n)
        elif restart == 1:
            params0 = base_angle * np.ones(layers * n)
        elif restart == 2:
            params0 = np.pi / 2 * np.ones(layers * n)
        elif restart == 3:
            # Alternating pattern
            params0 = np.zeros(layers * n)
            for l in range(layers):
                for q in range(n):
                    if q % 2 == 0:
                        params0[l * n + q] = base_angle
                    else:
                        params0[l * n + q] = -base_angle
        elif restart == 4:
            params0 = np.random.uniform(-np.pi/4, np.pi/4, size=(layers * n))
        elif restart == 5:
            params0 = np.random.uniform(-np.pi/2, np.pi/2, size=(layers * n))
        elif restart == 6:
            params0 = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        elif restart == 7:
            params0 = base_angle * np.ones(layers * n) + np.random.normal(0, 0.3, size=(layers * n))
        elif restart == 8:
            params0 = base_angle * np.ones(layers * n) + np.random.normal(0, 0.5, size=(layers * n))
        else:
            params0 = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        
        # Use L-BFGS-B with numerical gradient
        try:
            result = minimize(
                _objective,
                params0,
                args=(n, coupling, field, layers, H, E0, ry_indices, cz_indices),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    # Final polish: try a tighter optimization from the best point
    if time.time() < deadline - 15.0:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, H, E0, ry_indices, cz_indices),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Additional Nelder-Mead polish if time permits
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, H, E0, ry_indices, cz_indices),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Reshape to (layers, n)
    return best_params.reshape(layers, n)

# EVOLVE-BLOCK-END
