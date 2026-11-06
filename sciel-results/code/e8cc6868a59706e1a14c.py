"""Optimized shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Precompute basis states and operators for fast energy evaluation
# For n=4: 16 basis states, for n=6: 64 basis states
# We precompute the Z_i Z_{i+1} and X_i operators as sparse-ish structures

def _build_operators(n):
    """Build operator matrices for the Hamiltonian terms."""
    dim = 1 << n
    # Z_i Z_{i+1} for each adjacent pair
    ZZ_ops = []
    for i in range(n - 1):
        op = np.ones(dim)
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_j = (state >> (i + 1)) & 1
            if bit_i != bit_j:
                op[state] = -1.0
        ZZ_ops.append(op)
    # X_i for each qubit
    X_ops = []
    for i in range(n):
        op = np.zeros(dim)
        for state in range(dim):
            flipped = state ^ (1 << i)
            op[state] = 1.0 if flipped != state else 0.0
        # X is off-diagonal
        X_ops.append(op)
    return ZZ_ops, X_ops

# Precompute for n=4 and n=6
_OPS_CACHE = {}

def _get_ops(n):
    if n not in _OPS_CACHE:
        _OPS_CACHE[n] = _build_operators(n)
    return _OPS_CACHE[n]


def _apply_ry(state, theta):
    """Apply Ry(theta) to a single qubit in a state vector (complex array)."""
    # state is a 2^n complex vector, theta is the rotation angle
    # We need to apply Ry to each qubit. For a product of single-qubit gates,
    # we can do this efficiently.
    # Actually, let's just build the full circuit application.
    pass


def _circuit_state(theta, n, layers):
    """
    Build the circuit state from angles.
    theta: (layers, n) array of angles
    n: number of qubits
    layers: number of layers (fixed at 3)
    
    Circuit: Start |0>^n
    For each layer l:
        Apply Ry(theta[l, i]) to qubit i for all i
        If not the last layer, apply CZ on every adjacent pair in ascending order
    """
    dim = 1 << n
    # Start state: |0>^n = |00...0>
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry to each qubit
        # Ry(theta) = cos(theta/2) * I - i*sin(theta/2) * X
        # For each qubit i, we apply this gate
        for i in range(n):
            c = np.cos(theta[l, i] / 2.0)
            s = -1j * np.sin(theta[l, i] / 2.0)
            # Apply to qubit i: for each basis state, if bit i is 0, 
            # the amplitude gets multiplied by c for |0> and s for |1>
            # We need to do this in-place
            new_state = np.zeros(dim, dtype=complex)
            for state_idx in range(dim):
                amp = state[state_idx]
                if amp == 0:
                    continue
                bit_i = (state_idx >> i) & 1
                if bit_i == 0:
                    # |0> component: c * |0> + s * |1>
                    # This state has bit i = 0, so it contributes to |0> with c
                    new_state[state_idx] += c * amp
                    # And to state with bit i flipped to 1 with s
                    new_state[state_idx | (1 << i)] += s * amp
                else:
                    # bit_i = 1, so this state has bit i = 1
                    # Ry maps |1> -> s*|0> + c*|1>
                    new_state[state_idx & ~(1 << i)] += s * amp
                    new_state[state_idx] += c * amp
            state = new_state
        
        # Apply CZ if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                # CZ: |11> -> -|11>, others unchanged
                for state_idx in range(dim):
                    bit_i = (state_idx >> i) & 1
                    bit_j = (state_idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_j == 1:
                        state[state_idx] *= -1.0
    
    return state


def _compute_energy(state, n, coupling, field):
    """Compute <state|H|state> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i"""
    dim = 1 << n
    energy = 0.0
    prob = np.abs(state) ** 2
    
    # ZZ terms
    for i in range(n - 1):
        for state_idx in range(dim):
            bit_i = (state_idx >> i) & 1
            bit_j = (state_idx >> (i + 1)) & 1
            zz_val = 1.0 if bit_i == bit_j else -1.0
            energy += -coupling * zz_val * prob[state_idx]
    
    # X terms: <state|X_i|state> = sum_{s} <s|X_i|s> = sum of off-diagonal
    # <state|X_i|state> = sum_{s} conj(state[s]) * state[s ^ (1<<i)]
    for i in range(n):
        x_val = 0.0
        for state_idx in range(dim):
            flipped = state_idx ^ (1 << i)
            x_val += np.conj(state[state_idx]) * state[flipped]
        # X is Hermitian, so <X> is real
        x_val = np.real(x_val)
        energy += -field * x_val
    
    return energy


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        for state_idx in range(dim):
            bit_i = (state_idx >> i) & 1
            bit_j = (state_idx >> (i + 1)) & 1
            zz_val = 1.0 if bit_i == bit_j else -1.0
            H[state_idx, state_idx] += -coupling * zz_val
    
    # X terms
    for i in range(n):
        for state_idx in range(dim):
            flipped = state_idx ^ (1 << i)
            H[state_idx, flipped] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _objective(params, n, coupling, field, layers, exact_e):
    """Objective: mean energy gap per spin = (E_circuit - E_exact) / n"""
    theta = params.reshape(layers, n)
    state = _circuit_state(theta, n, layers)
    e_circuit = _compute_energy(state, n, coupling, field)
    return (e_circuit - exact_e) / n


def angles(n, coupling, field, layers):
    """
    Return optimal angles for the circuit.
    """
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Get exact ground energy
    exact_e = _exact_ground_energy(n, coupling, field)
    
    # Initial guess: symmetric angles
    # For the transverse field Ising model, the ground state has symmetry
    # Try a reasonable initial guess
    theta0 = np.zeros((layers, n))
    # Heuristic: larger field -> more rotation toward |1>
    # At zero coupling, it's like independent transverse field
    # For single qubit: -field*X, ground state is |-> = (|0>-|1>)/sqrt(2)
    # Ry(theta) |0> = cos(theta/2)|0> - i sin(theta/2)|1>
    # To get |->, we need theta = -pi/2 (or 3pi/2)
    # But with coupling, it's more complex
    
    # Use a simple heuristic based on field strength
    # For strong field, rotate more
    base_theta = -np.pi / 2 * min(1.0, field / 2.0)
    theta0[:, :] = base_theta
    
    # Try multiple initializations and local optimization
    best_params = theta0.flatten()
    best_obj = _objective(best_params, n, coupling, field, layers, exact_e)
    
    # Local optimization with L-BFGS-B
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    try:
        result = minimize(
            _objective,
            best_params,
            args=(n, coupling, field, layers, exact_e),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 5000}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_params.reshape(layers, n)
    
    # Try differential evolution for global search
    try:
        de_result = differential_evolution(
            _objective,
            bounds=bounds,
            args=(n, coupling, field, layers, exact_e),
            maxiter=50,
            popsize=15,
            tol=0.01,
            seed=42
        )
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_params = de_result.x
    except Exception:
        pass
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_params.reshape(layers, n)
    
    # Polish with local optimization from best point
    try:
        result = minimize(
            _objective,
            best_params,
            args=(n, coupling, field, layers, exact_e),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 10000}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    return best_params.reshape(layers, n)

# EVOLVE-BLOCK-END
