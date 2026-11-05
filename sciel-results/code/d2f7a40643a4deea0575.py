"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = None

def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _remaining_time(margin=5.0):
    return 150.0 - (time.time() - _get_start_time()) - margin

def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    # ZZ coupling: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bip1 = (state >> (i + 1)) & 1
            zz = (1 - 2 * bi) * (1 - 2 * bip1)
            H[state, state] += -coupling * zz
    # X field: -field * sum X_i
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    return H

def _ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _circuit_state(params, n, layers):
    """Apply the circuit: start |0>, for each layer: Ry then CZ (except last)."""
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0
    for layer in range(layers):
        # Ry on each qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Ry = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            # Apply to qubit q: for each basis state, split
            new_state = np.zeros_like(state)
            for basis in range(2 ** n):
                bit = (basis >> q) & 1
                if bit == 0:
                    # |0> -> c|0> + s|1>
                    idx0 = basis
                    idx1 = basis | (1 << q)
                    new_state[idx0] += state[basis] * c
                    new_state[idx1] += state[basis] * s
                else:
                    # |1> -> -s|0> + c|1>
                    idx1 = basis
                    idx0 = basis & ~(1 << q)
                    new_state[idx0] += state[basis] * (-s)
                    new_state[idx1] += state[basis] * c
            state = new_state
        # CZ on adjacent pairs (not after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                for basis in range(2 ** n):
                    bi = (basis >> i) & 1
                    bip1 = (basis >> (i + 1)) & 1
                    if bi == 1 and bip1 == 1:
                        state[basis] *= -1.0
    return state

def _energy(params, n, coupling, field, layers, ground_e):
    state = _circuit_state(params, n, layers)
    # Energy = <state|H|state>
    # Build H efficiently
    dim = 2 ** n
    H = _build_hamiltonian(n, coupling, field)
    energy = np.real(np.vdot(state, H @ state))
    return (energy - ground_e) / n

def _physics_init(n, coupling, field, layers):
    """Physics-inspired initialization using mean-field / arctan heuristic."""
    theta = np.zeros((layers, n), dtype=np.float64)
    # Mean-field: for a single spin in field h and coupling J, 
    # optimal angle ~ arctan(h/J) for the transverse component
    # For the Ising model H = -J sum ZZ - h sum X
    # Mean field: each spin feels effective field from neighbors
    # Simple heuristic: theta = arctan(field / coupling) for all
    base_angle = np.arctan2(field, coupling) if coupling != 0 else np.pi / 2
    # Layer 1: small rotation to build up correlation
    theta[0] = base_angle * 0.3
    # Layer 2: medium rotation
    theta[1] = base_angle * 0.6
    # Layer 3: full rotation
    theta[2] = base_angle
    return theta

def angles(n, coupling, field, layers):
    t_start = time.time()
    
    # Compute exact ground energy
    ground_e = _ground_energy(n, coupling, field)
    
    # Physics-inspired initialization
    theta0 = _physics_init(n, coupling, field, layers)
    
    # Objective function
    def objective(params_flat):
        params = params_flat.reshape(layers, n)
        return _energy(params, n, coupling, field, layers, ground_e)
    
    # Try optimization with deadline guard
    best_params = theta0.flatten()
    best_val = objective(best_params)
    
    # Use L-BFGS-B with bounds
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    try:
        if _remaining_time(10.0) > 5.0:
            res = minimize(
                objective, best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
    except Exception:
        pass
    
    # Try a second round with different init if time permits
    if _remaining_time(10.0) > 5.0:
        try:
            # Alternative init: start from pi/2 (|+> state)
            theta_alt = np.zeros((layers, n), dtype=np.float64)
            theta_alt[-1] = np.pi / 2
            alt_params = theta_alt.flatten()
            alt_val = objective(alt_params)
            if alt_val < best_val:
                best_val = alt_val
                best_params = alt_params
            
            res2 = minimize(
                objective, best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    # Third round: random restart if time permits
    if _remaining_time(10.0) > 5.0:
        try:
            rng = np.random.default_rng(42)
            for _ in range(3):
                if _remaining_time(10.0) < 5.0:
                    break
                rand_params = rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten()
                rand_val = objective(rand_params)
                if rand_val < best_val:
                    best_val = rand_val
                    best_params = rand_params
                res3 = minimize(
                    objective, best_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 300}
                )
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_params = res3.x
        except Exception:
            pass
    
    return best_params.reshape(layers, n)
# EVOLVE-BLOCK-END
