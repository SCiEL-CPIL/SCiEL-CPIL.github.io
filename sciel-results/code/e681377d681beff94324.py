"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _apply_ry(state, theta, n):
    """Apply Ry(theta) to every qubit in place."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    # Ry = [[c, -s], [s, c]]
    # For each qubit, reshape and apply
    for i in range(n):
        # Reshape state so qubit i is the last axis
        shape = [2] * n
        # Move axis i to the end
        state = np.moveaxis(state, i, -1)
        state = state.reshape([-1, 2])
        # Apply Ry
        new_state = np.empty_like(state)
        new_state[:, 0] = c * state[:, 0] - s * state[:, 1]
        new_state[:, 1] = s * state[:, 0] + c * state[:, 1]
        state = new_state
        # Move axis back
        state = np.moveaxis(state, -1, i)
        state = state.reshape(shape)
    return state

def _apply_cz(state, i, j, n):
    """Apply CZ between qubits i and j."""
    # CZ = diag(1, 1, 1, -1) in computational basis
    # Reshape so qubits i and j are the last two axes
    state = np.moveaxis(state, i, -2)
    state = np.moveaxis(state, j, -1)
    shape = state.shape
    state = state.reshape([-1, 2, 2])
    # CZ: flip sign of |11> component
    state[:, 1, 1] *= -1.0
    state = state.reshape(shape)
    state = np.moveaxis(state, -1, j)
    state = np.moveaxis(state, -2, i)
    return state

def _build_state(theta, n, layers):
    """Build the circuit state from parameters."""
    # Start |0>^n
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry(theta[layer]) to every qubit
        state = _apply_ry(state, theta[layer], n)
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    return state

def _compute_energy(state, n, coupling, field):
    """Compute <psi|H|psi> for H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2**n
    energy = 0.0
    
    # ZZ terms: -coupling * sum_{i=0}^{n-2} Z_i Z_{i+1}
    for i in range(n - 1):
        # Z_i Z_{i+1} expectation
        zz = 0.0
        for bit in range(dim):
            bi = (bit >> i) & 1
            bj = (bit >> (i + 1)) & 1
            zi = 1 - 2 * bi
            zj = 1 - 2 * bj
            zz += zi * zj * (state[bit].real * state[bit].real + state[bit].imag * state[bit].imag)
        energy += -coupling * zz
    
    # X terms: -field * sum X_i
    for i in range(n):
        xi = 0.0
        for bit in range(dim):
            # X_i flips bit i
            other = bit ^ (1 << i)
            # <psi| X_i |psi> = sum_k psi_k^* psi_{k^flip}
            # This is 2*Re(sum_k psi_k * psi_{k^flip}) for k where bit i = 0
            if (bit >> i) & 1 == 0:
                xi += 2.0 * (state[bit].real * state[other].real + state[bit].imag * state[other].imag)
        energy += -field * xi
    
    return energy

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        for bit in range(dim):
            bi = (bit >> i) & 1
            bj = (bit >> (i + 1)) & 1
            zi = 1 - 2 * bi
            zj = 1 - 2 * bj
            H[bit, bit] += -coupling * zi * zj
    
    # X terms
    for i in range(n):
        for bit in range(dim):
            other = bit ^ (1 << i)
            H[bit, other] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _objective(params, n, coupling, field, layers, ground_energy):
    """Objective: mean energy gap per spin."""
    theta = params.reshape(layers, n)
    state = _build_state(theta, n, layers)
    circuit_energy = _compute_energy(state, n, coupling, field)
    gap = (circuit_energy - ground_energy) / n
    return gap

def angles(n, coupling, field, layers):
    """Return optimized circuit parameters."""
    total_params = layers * n
    
    # Compute exact ground energy
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    # Multi-start Nelder-Mead optimization
    best_params = None
    best_obj = np.inf
    
    # Initial guesses: try several starting points
    rng = np.random.default_rng(42)
    
    # Start 1: all zeros (product |0>)
    starts = [np.zeros(total_params)]
    
    # Start 2: all pi/2 (product |+>)
    starts.append(np.full(total_params, np.pi / 2))
    
    # Start 3-8: random
    for _ in range(7):
        starts.append(rng.uniform(-np.pi, np.pi, total_params))
    
    for start in starts:
        if _time_left() < 5.0:
            break
        
        try:
            result = minimize(
                _objective,
                start,
                args=(n, coupling, field, layers, ground_energy),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    if best_params is None:
        # Fallback: product |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_params.reshape(layers, n)
# EVOLVE-BLOCK-END
