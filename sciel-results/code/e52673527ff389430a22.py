import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_(i+1) - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_(i+1) terms (diagonal)
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * zi * zip1
    
    # X_i terms (off-diagonal)
    for i in range(n):
        for idx in range(dim):
            flipped = idx ^ (1 << i)
            H[idx, flipped] += -field
    
    return H

def _get_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _apply_ry_vectorized(state, theta, qubit, n):
    """Apply Ry(theta) to qubit using vectorized operations."""
    dim = 2 ** n
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    
    mask = 1 << qubit
    idx0 = np.arange(dim)
    is_0 = (idx0 & mask) == 0
    
    result = np.zeros_like(state)
    idx0_0 = idx0[is_0]
    idx1_0 = idx0_0 | mask
    v0 = state[idx0_0]
    v1 = state[idx1_0]
    result[idx0_0] = c * v0 - s * v1
    result[idx1_0] = s * v0 + c * v1
    
    return result

def _apply_cz_vectorized(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2 using vectorized operations."""
    dim = 2 ** n
    mask1 = 1 << q1
    mask2 = 1 << q2
    
    idx = np.arange(dim)
    both_1 = ((idx & mask1) == mask1) & ((idx & mask2) == mask2)
    
    result = state.copy()
    result[both_1] *= -1.0
    
    return result

def _build_circuit_state_fast(theta, n, layers):
    """Build the state vector from the circuit parameters using vectorized gates."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry_vectorized(state, theta[layer, q], q, n)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz_vectorized(state, q, q + 1, n)
    
    return state

def _compute_energy_fast(state, H):
    """Compute expectation value using pre-built Hamiltonian."""
    return np.real(np.vdot(state, H @ state))

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    # Get ground energy and Hamiltonian
    ground_energy = _get_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    n_params = layers * n
    
    def objective(params):
        theta = params.reshape((layers, n))
        state = _build_circuit_state_fast(theta, n, layers)
        energy = _compute_energy_fast(state, H)
        return (energy - ground_energy) / n
    
    def objective_and_grad(params):
        theta = params.reshape((layers, n))
        state = _build_circuit_state_fast(theta, n, layers)
        energy = _compute_energy_fast(state, H)
        
        # Compute gradient via central finite differences
        grad = np.zeros_like(params)
        eps = 1e-6
        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            theta_plus = params_plus.reshape((layers, n))
            theta_minus = params_minus.reshape((layers, n))
            state_plus = _build_circuit_state_fast(theta_plus, n, layers)
            state_minus = _build_circuit_state_fast(theta_minus, n, layers)
            energy_plus = _compute_energy_fast(state_plus, H)
            energy_minus = _compute_energy_fast(state_minus, H)
            grad[i] = (energy_plus - energy_minus) / (2 * eps)
        
        return (energy - ground_energy) / n, grad / n
    
    best_result = None
    best_val = np.inf
    
    # Generate diverse starting points
    start_points = []
    
    # Starting point 1: all zeros
    start_points.append(np.zeros(n_params))
    
    # Starting point 2: pi/2 in last layer (transverse field dominated)
    x0_2 = np.zeros(n_params)
    x0_2[(layers-1)*n:(layers)*n] = np.pi / 2
    start_points.append(x0_2)
    
    # Starting point 3: pi/4 in all layers
    start_points.append(np.full(n_params, np.pi / 4))
    
    # Starting point 4: -pi/4 in all layers
    start_points.append(np.full(n_params, -np.pi / 4))
    
    # Starting point 5: alternating pi/4, -pi/4
    x0_5 = np.zeros(n_params)
    for i in range(n_params):
        x0_5[i] = np.pi / 4 if i % 2 == 0 else -np.pi / 4
    start_points.append(x0_5)
    
    # Starting point 6: small angles
    start_points.append(np.full(n_params, 0.1))
    
    # Starting point 7: larger angles
    start_points.append(np.full(n_params, 0.3))
    
    # Add random starting points
    np.random.seed(42)
    for _ in range(15):
        start_points.append(np.random.uniform(-np.pi, np.pi, size=n_params))
    
    # Add small random perturbations
    for _ in range(10):
        start_points.append(np.random.uniform(-0.5, 0.5, size=n_params))
    
    # Phase 1: Coarse optimization with L-BFGS-B
    for x0 in start_points:
        if time.time() > deadline - 30:  # Reserve 30s for polish
            break
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with tighter tolerance
    if best_result is not None and time.time() < deadline - 5:
        try:
            res_polish = minimize(objective_and_grad, best_result, method='L-BFGS-B',
                                  options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_result = res_polish.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final local search with Nelder-Mead for very fine tuning
    if best_result is not None and time.time() < deadline - 2:
        try:
            res_nm = minimize(objective, best_result, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_result = res_nm.x.copy()
        except Exception:
            pass
    
    if best_result is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_result.reshape((layers, n))
