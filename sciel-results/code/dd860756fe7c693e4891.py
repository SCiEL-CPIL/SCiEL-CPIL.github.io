"""Optimize 3-layer Ry/CZ circuit for Ising ground states using parameter-shift gradients."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    for state in range(dim):
        # Z_i Z_{i+1} terms (diagonal)
        zz_energy = 0.0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_energy += z_i * z_ip1
        H[state, state] += -coupling * zz_energy
        
        # X_i terms (off-diagonal)
        for i in range(n):
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H


def _apply_ry_matrix(theta, qubit, n):
    """Build the Ry rotation matrix for a given qubit."""
    dim = 2 ** n
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    R = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        bit = (idx >> qubit) & 1
        if bit == 0:
            R[idx, idx] += c
            R[idx | (1 << qubit), idx] += s
        else:
            R[idx ^ (1 << qubit), idx] += -s
            R[idx, idx] += c
    return R


def _apply_cz_matrix(q1, q2, n):
    """Build the CZ gate matrix for a given pair of qubits."""
    dim = 2 ** n
    CZ = np.eye(dim, dtype=complex)
    for idx in range(dim):
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        if b1 and b2:
            CZ[idx, idx] = -1.0
    return CZ


def _circuit_state(n, layers, thetas):
    """Apply the full circuit to |0>^n and return the state vector."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            c = np.cos(thetas[layer, q] / 2)
            s = np.sin(thetas[layer, q] / 2)
            new_state = np.zeros_like(state)
            for idx in range(dim):
                bit = (idx >> q) & 1
                if bit == 0:
                    new_state[idx] += c * state[idx]
                    new_state[idx | (1 << q)] += s * state[idx]
                else:
                    new_state[idx ^ (1 << q)] += -s * state[idx]
                    new_state[idx] += c * state[idx]
            state = new_state
        # Apply CZ between adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                for idx in range(dim):
                    b1 = (idx >> q) & 1
                    b2 = (idx >> (q + 1)) & 1
                    if b1 and b2:
                        state[idx] = -state[idx]
    
    return state


def _energy(state, H):
    """Compute <state|H|state>."""
    return np.real(np.vdot(state, H @ state))


def _exact_ground_energy(H):
    """Compute exact ground energy."""
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _objective(params, n, coupling, field, layers, H, ground_energy):
    """Objective: mean energy gap per spin."""
    thetas = params.reshape(layers, n)
    state = _circuit_state(n, layers, thetas)
    e = _energy(state, H)
    return (e - ground_energy) / n


def _gradient(params, n, coupling, field, layers, H, ground_energy):
    """Compute gradient using parameter-shift rule for Ry gates."""
    thetas = params.reshape(layers, n)
    grad = np.zeros_like(params)
    shift = np.pi / 2
    
    for layer in range(layers):
        for q in range(n):
            idx = layer * n + q
            
            # Evaluate at theta + pi/2
            thetas_plus = thetas.copy()
            thetas_plus[layer, q] += shift
            obj_plus = _objective(thetas_plus.flatten(), n, coupling, field, layers, H, ground_energy)
            
            # Evaluate at theta - pi/2
            thetas_minus = thetas.copy()
            thetas_minus[layer, q] -= shift
            obj_minus = _objective(thetas_minus.flatten(), n, coupling, field, layers, H, ground_energy)
            
            # Parameter-shift rule: gradient = (f(theta+pi/2) - f(theta-pi/2)) / 2
            grad[idx] = (obj_plus - obj_minus) / 2
    
    return grad


def _physics_init(n, coupling, field, layers):
    """Physics-inspired initialization based on arctan."""
    thetas = np.zeros((layers, n), dtype=float)
    base_angle = np.arctan2(field, coupling) if coupling != 0 else np.pi / 2
    
    # Layer 1: small angle to create some superposition
    thetas[0, :] = base_angle * 0.3
    # Layer 2: medium angle
    thetas[1, :] = base_angle * 0.7
    # Layer 3: full angle
    thetas[2, :] = base_angle
    
    return thetas


def _random_init(n, layers, scale=1.0):
    """Random initialization."""
    return np.random.uniform(-scale, scale, size=(layers, n))


def angles(n, coupling, field, layers):
    """Return optimized circuit parameters."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build Hamiltonian and compute exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(H)
    
    dim = 2 ** n
    n_params = layers * n
    
    best_thetas = None
    best_obj = np.inf
    
    # Generate diverse initializations
    inits = []
    
    # Physics-inspired
    inits.append(_physics_init(n, coupling, field, layers))
    
    # Variations around physics init
    for _ in range(5):
        init = _physics_init(n, coupling, field, layers)
        init += np.random.normal(0, 0.5, size=(layers, n))
        inits.append(init)
    
    # Random inits with different scales
    for scale in [0.5, 1.0, 1.5, 2.0]:
        inits.append(_random_init(n, layers, scale))
    
    # All zeros and all pi/2
    inits.append(np.zeros((layers, n)))
    inits.append(np.full((layers, n), np.pi / 2))
    
    # Evaluate initializations
    for init in inits:
        obj = _objective(init.flatten(), n, coupling, field, layers, H, ground_energy)
        if obj < best_obj:
            best_obj = obj
            best_thetas = init.copy()
    
    # Optimize from best initializations using parameter-shift gradients
    # Sort inits by objective value and try the best ones
    init_objs = []
    for init in inits:
        obj = _objective(init.flatten(), n, coupling, field, layers, H, ground_energy)
        init_objs.append((obj, init))
    init_objs.sort(key=lambda x: x[0])
    
    # Try optimizing from the top initializations
    max_optimizations = 6
    for i, (init_obj, init) in enumerate(init_objs[:max_optimizations]):
        if time.time() > deadline - 25:
            break
        
        try:
            result = minimize(
                _objective,
                init.flatten(),
                args=(n, coupling, field, layers, H, ground_energy),
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50, 'maxfun': 2000}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_thetas = result.x.reshape(layers, n)
        except Exception:
            continue
    
    # If we have time, do a high-precision polish on the best solution
    if time.time() < deadline - 10 and best_thetas is not None:
        try:
            result = minimize(
                _objective,
                best_thetas.flatten(),
                args=(n, coupling, field, layers, H, ground_energy),
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100, 'maxfun': 5000}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_thetas = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    if best_thetas is None:
        return _physics_init(n, coupling, field, layers)
    
    return best_thetas
