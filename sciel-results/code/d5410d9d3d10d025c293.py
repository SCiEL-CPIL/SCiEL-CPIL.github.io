"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 148.0
_START_TIME = time.time()


def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    total_dim = 2 ** n
    H = np.zeros((total_dim, total_dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(total_dim):
            b1 = (state >> i) & 1
            b2 = (state >> (i + 1)) & 1
            zz = (1 if b1 == 0 else -1) * (1 if b2 == 0 else -1)
            H[state, state] += -coupling * zz
    
    # X terms: -field * X_i
    for i in range(n):
        for state in range(total_dim):
            partner = state ^ (1 << i)
            H[state, partner] += -field
    
    return H


def _get_ground_energy(H):
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real


def _apply_ry(state, n, q, theta):
    """Apply Ry(theta) to qubit q of state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    new_state = np.zeros_like(state)
    for idx in range(2 ** n):
        bit = (idx >> q) & 1
        if bit == 0:
            new_state[idx] += c * state[idx]
            new_state[idx | (1 << q)] += s * state[idx]
        else:
            new_state[idx] += c * state[idx]
            new_state[idx ^ (1 << q)] += -s * state[idx]
    return new_state


def _apply_cz(state, n, q1, q2):
    """Apply CZ between qubits q1 and q2."""
    for idx in range(2 ** n):
        if ((idx >> q1) & 1) == 1 and ((idx >> q2) & 1) == 1:
            state[idx] *= -1.0
    return state


def _circuit_state(params, n, layers):
    """Build circuit state from parameters."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry(state, n, q, params[layer, q])
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, n, q, q + 1)
    
    return state


def _objective_flat(x, n, coupling, field, layers, H, ground_energy):
    """Objective function for flat parameter vector."""
    params = x.reshape(layers, n)
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state).real
    return (energy - ground_energy) / n


def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(H)
    
    # Bounds for angles
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Differential evolution for global search
    if _check_deadline(margin=10.0):
        try:
            de_result = differential_evolution(
                _objective_flat,
                bounds,
                args=(n, coupling, field, layers, H, ground_energy),
                seed=42,
                maxiter=50,
                tol=1e-8,
                polish=False,
                workers=1
            )
            best_params = de_result.x.reshape(layers, n)
            best_score = _objective_flat(de_result.x, n, coupling, field, layers, H, ground_energy)
        except Exception:
            best_params = np.zeros((layers, n))
            best_score = _objective_flat(np.zeros(layers * n), n, coupling, field, layers, H, ground_energy)
    else:
        best_params = np.zeros((layers, n))
        best_score = _objective_flat(np.zeros(layers * n), n, coupling, field, layers, H, ground_energy)
    
    # Local refinement with L-BFGS-B
    if _check_deadline(margin=5.0):
        try:
            result = minimize(
                _objective_flat,
                best_params.flatten(),
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Additional refinement from perturbed starting points
    if _check_deadline(margin=3.0):
        np.random.seed(999)
        for _ in range(3):
            if not _check_deadline(margin=3.0):
                break
            perturbed = best_params.flatten() + np.random.normal(0, 0.1, size=layers * n)
            try:
                result = minimize(
                    _objective_flat,
                    perturbed,
                    args=(n, coupling, field, layers, H, ground_energy),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.reshape(layers, n)
            except Exception:
                continue
    
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)
    return best_params
# EVOLVE-BLOCK-END
