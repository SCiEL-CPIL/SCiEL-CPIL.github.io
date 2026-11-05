# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix efficiently."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: diagonal
    for state in range(dim):
        zz_energy = 0.0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_energy += z_i * z_ip1
        H[state, state] += -coupling * zz_energy
    
    # X terms: off-diagonal
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H

def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit on a state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    new_state = np.zeros_like(state)
    dim = len(state)
    for idx in range(dim):
        bit = (idx >> qubit) & 1
        if bit == 0:
            # |0> -> c|0> + s|1>
            new_state[idx] += c * state[idx]
            new_state[idx | (1 << qubit)] += s * state[idx]
        else:
            # |1> -> -s|0> + c|1>
            new_state[idx ^ (1 << qubit)] += -s * state[idx]
            new_state[idx] += c * state[idx]
    return new_state

def _apply_cz(state, q1, q2):
    """Apply CZ between qubits q1 and q2."""
    new_state = state.copy()
    dim = len(state)
    for idx in range(dim):
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        if b1 and b2:
            new_state[idx] = -state[idx]
    return new_state

def _circuit_state(n, layers, thetas):
    """Apply the full circuit to |0>^n and return the state vector."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry(state, thetas[layer, q], q)
        # Apply CZ between adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1)
    
    return state

def _energy(state, H):
    """Compute <state|H|state>."""
    return np.real(np.vdot(state, H @ state))

def _exact_ground_energy(H):
    """Compute exact ground energy."""
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _physics_init(n, coupling, field, layers):
    """Physics-inspired initialization based on arctan."""
    thetas = np.zeros((layers, n), dtype=float)
    # For a single qubit with effective transverse field, optimal angle ~ arctan(field/coupling)
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

def _objective(params, n, coupling, field, layers, H, ground_energy):
    """Objective: mean energy gap per spin."""
    thetas = params.reshape(layers, n)
    state = _circuit_state(n, layers, thetas)
    e = _energy(state, H)
    return (e - ground_energy) / n

def _objective_with_grad(params, n, coupling, field, layers, H, ground_energy):
    """
    Objective with exact gradient using the parameter-shift rule for Ry gates.
    For an Ry(theta) gate, the parameter-shift rule gives:
    d/dtheta <psi|H|psi> = 0.5 * (<psi(theta+pi/2)|H|psi(theta+pi/2)> - <psi(theta-pi/2)|H|psi(theta-pi/2)>)
    This is exact (not finite-difference) for rotation gates.
    """
    thetas = params.reshape(layers, n)
    state = _circuit_state(n, layers, thetas)
    e = _energy(state, H)
    obj = (e - ground_energy) / n
    
    # Compute exact gradient via parameter-shift rule
    grad = np.zeros_like(params)
    shift = np.pi / 2
    
    for layer in range(layers):
        for q in range(n):
            idx = layer * n + q
            
            # theta + pi/2
            thetas_plus = thetas.copy()
            thetas_plus[layer, q] += shift
            state_plus = _circuit_state(n, layers, thetas_plus)
            e_plus = _energy(state_plus, H)
            
            # theta - pi/2
            thetas_minus = thetas.copy()
            thetas_minus[layer, q] -= shift
            state_minus = _circuit_state(n, layers, thetas_minus)
            e_minus = _energy(state_minus, H)
            
            # Parameter-shift rule: dE/dtheta = 0.5 * (E(θ+π/2) - E(θ-π/2))
            grad[idx] = 0.5 * (e_plus - e_minus) / n
    
    return obj, grad

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
    for _ in range(4):
        init = _physics_init(n, coupling, field, layers)
        init += np.random.normal(0, 0.4, size=(layers, n))
        inits.append(init)
    
    # Random inits with different scales
    for scale in [0.3, 0.7, 1.2]:
        inits.append(_random_init(n, layers, scale))
    
    # All zeros and all pi/2
    inits.append(np.zeros((layers, n)))
    inits.append(np.full((layers, n), np.pi / 2))
    inits.append(np.full((layers, n), -np.pi / 2))
    
    # Evaluate initializations
    for init in inits:
        obj = _objective(init.flatten(), n, coupling, field, layers, H, ground_energy)
        if obj < best_obj:
            best_obj = obj
            best_thetas = init.copy()
    
    # Sort inits by objective value
    init_objs = []
    for init in inits:
        obj = _objective(init.flatten(), n, coupling, field, layers, H, ground_energy)
        init_objs.append((obj, init))
    init_objs.sort(key=lambda x: x[0])
    
    # Optimize from best initializations using gradient-based method with exact gradient
    max_optimizations = 8
    for idx in range(min(max_optimizations, len(init_objs))):
        if time.time() > deadline - 20:
            break
        
        obj_init, init = init_objs[idx]
        
        try:
            result = minimize(
                _objective_with_grad,
                init.flatten(),
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-9, 'maxls': 50}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_thetas = result.x.reshape(layers, n)
        except Exception:
            continue
    
    # High-precision polish on the best solution
    if time.time() < deadline - 10 and best_thetas is not None:
        try:
            result = minimize(
                _objective_with_grad,
                best_thetas.flatten(),
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 80}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_thetas = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final ultra-precision polish if time allows
    if time.time() < deadline - 3 and best_thetas is not None:
        try:
            result = minimize(
                _objective_with_grad,
                best_thetas.flatten(),
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_thetas = result.x.reshape(layers, n)
        except Exception:
            pass
    
    if best_thetas is None:
        return _physics_init(n, coupling, field, layers)
    
    return best_thetas
# EVOLVE-BLOCK-END
