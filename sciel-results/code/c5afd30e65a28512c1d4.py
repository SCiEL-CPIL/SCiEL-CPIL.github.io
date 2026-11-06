"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 142.0  # safety margin before 150s kill


def _ry_matrix(theta):
    """Return 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit of state (2^n vector). Qubit i is bit i (LSB)."""
    if abs(theta) < 1e-15:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    gate = _ry_matrix(theta)
    state_moved = np.moveaxis(state_r, qubit, 0)
    state_out = np.tensordot(gate, state_moved, axes=([1], [0]))
    state_moved_back = np.moveaxis(state_out, 0, qubit)
    return state_moved_back.reshape(-1)


def _apply_cz(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2. Qubit i is bit i (LSB)."""
    if q1 == q2:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    state_moved = np.moveaxis(state_r, [q1, q2], [-2, -1])
    factor = np.ones(state_moved.shape, dtype=np.complex128)
    idx11 = (slice(None),) * (n - 2) + (1, 1)
    factor[idx11] = -1.0
    state_moved = state_moved * factor
    state_moved_back = np.moveaxis(state_moved, [-2, -1], [q1, q2])
    return state_moved_back.reshape(-1)


def _circuit_state(params, n, layers):
    """Compute the state vector produced by the circuit."""
    thetas = params.reshape(layers, n)
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry(state, thetas[layer, q], q, n)
        
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state


def _build_hamiltonian_correct(n, coupling, field):
    """Build the Hamiltonian matrix - corrected version."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalization."""
    H = _build_hamiltonian_correct(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _circuit_energy(params, n, coupling, field, layers, H=None):
    """Compute expectation value of H for given circuit parameters."""
    if H is None:
        H = _build_hamiltonian_correct(n, coupling, field)
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state)
    return np.real(energy)


def _energy_and_grad(params, n, coupling, field, layers, H):
    """Compute energy and gradient using finite differences."""
    e_circ = _circuit_energy(params, n, coupling, field, layers, H)
    e0 = _exact_ground_energy(n, coupling, field)
    energy = (e_circ - e0) / n
    
    # Numerical gradient
    grad = np.zeros_like(params)
    h = 1e-7
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += h
        e_plus = _circuit_energy(params_plus, n, coupling, field, layers, H)
        params_minus = params.copy()
        params_minus[i] -= h
        e_minus = _circuit_energy(params_minus, n, coupling, field, layers, H)
        grad[i] = (e_plus - e_minus) / (2 * h) / n
    
    return energy, grad


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian_correct(n, coupling, field)
    
    def objective(params):
        e_circ = _circuit_energy(params, n, coupling, field, layers, H)
        return (e_circ - e0) / n
    
    def objective_and_grad(params):
        e_circ = _circuit_energy(params, n, coupling, field, layers, H)
        energy = (e_circ - e0) / n
        
        # Central difference gradient
        grad = np.zeros_like(params)
        h = 1e-6
        for i in range(len(params)):
            params_p = params.copy()
            params_p[i] += h
            e_p = _circuit_energy(params_p, n, coupling, field, layers, H)
            params_m = params.copy()
            params_m[i] -= h
            e_m = _circuit_energy(params_m, n, coupling, field, layers, H)
            grad[i] = (e_p - e_m) / (2 * h) / n
        
        return energy, grad
    
    # Initial guesses based on physics
    x0 = np.zeros(layers * n)
    
    # For transverse field Ising model, ground state has magnetization along X
    # So angles should be near pi/2 for X polarization
    if field > 0.5:
        # Strong field: more X-like
        base_angle = np.pi / 2
    else:
        # Weak field: more Z-like
        base_angle = np.pi / 4
    
    # Layer 0: small rotation
    x0[0:n] = base_angle * 0.3
    # Layer 1: medium rotation  
    x0[n:2*n] = base_angle * 0.6
    # Layer 2: full rotation
    x0[2*n:3*n] = base_angle
    
    best_result = None
    best_val = np.inf
    
    # Multiple restart strategies
    np.random.seed(42)
    restart_configs = [
        ('zeros', np.zeros(layers * n)),
        ('pi/2 last', np.array([0]*2*n + [np.pi/2]*n)),
        ('pi/4 all', np.full(layers * n, np.pi/4)),
        ('physics', x0.copy()),
        ('random1', np.clip(np.random.randn(layers * n) * 0.5, -np.pi, np.pi)),
        ('random2', np.clip(np.random.randn(layers * n) * 0.8, -np.pi, np.pi)),
        ('symmetric', np.array([np.pi/8, np.pi/4, np.pi/2]*n)),
        ('antisymmetric', np.array([-np.pi/8, np.pi/4, np.pi/2]*n)),
    ]
    
    # Add more random restarts
    for r in range(4):
        restart_configs.append((f'random{r+3}', np.clip(np.random.randn(layers * n) * 0.6, -np.pi, np.pi)))
    
    for name, x_init in restart_configs:
        if time.time() - _start_time > _TIME_LIMIT:
            break
        
        try:
            # Try Nelder-Mead first
            res = opt.minimize(objective, x_init, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                       'adaptive': True})
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
        except Exception:
            pass
    
    if best_result is None:
        best_result = x0
    
    # Polish with L-BFGS-B using gradient
    if time.time() - _start_time < _TIME_LIMIT - 30:
        try:
            res2 = opt.minimize(objective_and_grad, best_result, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with tighter tolerances
    if time.time() - _start_time < _TIME_LIMIT - 15:
        try:
            res3 = opt.minimize(objective, best_result, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11,
                                        'adaptive': True})
            if res3.fun < best_val:
                best_val = res3.fun
                best_result = res3.x.copy()
        except Exception:
            pass
    
    theta = best_result.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
