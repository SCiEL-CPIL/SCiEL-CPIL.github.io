# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_phase_cache = {}
_idx_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0

def _get_phases(n):
    """
    Precompute phase vectors for CZ gates.
    CZ_{i, i+1} applies -1 to states where both i and i+1 are 1.
    """
    key = n
    if key in _phase_cache:
        return _phase_cache[key]
    dim = 1 << n
    phases = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        phase = np.ones(dim, dtype=np.float64)
        # Indices where both bits are set
        # Efficient way: iterate or use logical ops
        # Using np.arange for clarity and speed for small n
        idx = np.arange(dim)
        mask = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        phase[mask] = -1.0
        phases.append(phase)
    _phase_cache[key] = phases
    return phases

def _get_idx_pairs(n):
    """
    Precompute index pairs for Ry gates.
    Returns dict q -> (idx_0, idx_1)
    """
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    idx_pairs = {}
    for q in range(n):
        mask = 1 << q
        idx_0 = np.arange(dim)[(np.arange(dim) & mask) == 0]
        idx_1 = idx_0 + mask
        idx_pairs[q] = (idx_0, idx_1)
    _idx_cache[key] = idx_pairs
    return idx_pairs

def _circuit_state_fast(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 + 0j
    
    phases = _get_phases(n)
    idx_pairs = _get_idx_pairs(n)

    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = idx_pairs[q]
            
            v0 = psi[idx_0]
            v1 = psi[idx_1]
            
            psi[idx_0] = c * v0 - s * v1
            psi[idx_1] = s * v0 + c * v1

        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                psi *= phases[i]

    return psi

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.vdot(psi, H @ psi))

def _objective_full(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _symmetry_mapped_params(params, n, layers):
    """
    Maps a vector of length layers * n/2 to a full symmetric matrix of shape (layers, n).
    Assumes symmetry: theta[layer, i] == theta[layer, n-1-i]
    """
    half_n = n // 2
    thetas = np.zeros((layers, n))
    for l in range(layers):
        for i in range(half_n):
            val = params[l * half_n + i]
            thetas[l, i] = val
            thetas[l, n - 1 - i] = val
    return thetas

def _objective_sym(params, n, coupling, field, layers, e0):
    thetas = _symmetry_mapped_params(params, n, layers)
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _heuristic_init(n, coupling, field, layers):
    """
    Generate heuristic initial parameters based on physics intuition.
    Returns list of parameter vectors (symmetry-reduced if n is even).
    """
    inits = []
    half_n = n // 2
    
    # 1. Zero angles (|0...0>)
    inits.append(np.zeros(layers * half_n))
    
    # 2. Pi/2 angles (|+...+>)
    inits.append(np.full(layers * half_n, np.pi / 2))
    
    # 3. Pi angles (|1...1>)
    inits.append(np.full(layers * half_n, np.pi))
    
    # 4. Physics-inspired: Transverse field rotation
    # For H = -J sum Z Z - g sum X, if g >> J, state is near |+>. If g << J, state is near |0> or |1>.
    # Simple approximation: theta ~ arctan(g/J) or similar scaling.
    ratio = field / coupling if coupling != 0 else field
    # Map ratio to [0, pi/2] range
    theta_est = np.arctan(ratio) * 0.8 # Scaled down
    inits.append(np.full(layers * half_n, theta_est))
    
    # 5. Alternating small angles
    inits.append(np.full(layers * half_n, 0.5))
    inits.append(np.full(layers * half_n, 1.0))
    
    # 6. Random restarts
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(0, np.pi, size=(layers * half_n)))
        
    return inits

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 142.0 # 8s safety margin for 150s limit
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Use symmetry if n is even (which it is for 4 and 6)
    use_symmetry = (n % 2 == 0)
    if use_symmetry:
        dim_params = layers * (n // 2)
        obj_func = lambda p: _objective_sym(p, n, coupling, field, layers, e0)
        map_func = lambda p: _symmetry_mapped_params(p, n, layers)
        inits = _heuristic_init(n, coupling, field, layers)
    else:
        dim_params = layers * n
        obj_func = lambda p: _objective_full(p, n, coupling, field, layers, e0)
        map_func = lambda p: p.reshape((layers, n))
        inits = [np.zeros(dim_params)]
        inits.append(np.full(dim_params, np.pi/2))
        np.random.seed(42)
        for _ in range(10):
             inits.append(np.random.uniform(0, np.pi, size=(dim_params)))

    best_x = inits[0]
    best_val = obj_func(best_x)
    
    # Evaluate all inits first to find the best starting point
    for x_init in inits:
        if time.time() > deadline:
            break
        val = obj_func(x_init)
        if val < best_val:
            best_val = val
            best_x = x_init

    # Optimize from best init and potentially a few others
    # We prioritize the best init, then a couple of diverse ones
    candidates_to_optimize = [best_x]
    
    # Add more diverse inits now that we have more time budget
    # Each optimization takes ~1-5s typically for n=4,6 with L-BFGS-B
    seen = set()
    for x_init in inits:
        if tuple(np.round(x_init, 5)) in seen:
            continue
        seen.add(tuple(np.round(x_init, 5)))
        candidates_to_optimize.append(x_init)
        if len(candidates_to_optimize) >= 6:
            break

    for x_init in candidates_to_optimize:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(obj_func, x_init, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final polish if time remains - use the extra time for tighter convergence
    remaining = deadline - time.time()
    if remaining > 3.0:
        try:
            res = minimize(obj_func, best_x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Second polish pass if still time remains
    remaining = deadline - time.time()
    if remaining > 2.0:
        try:
            res = minimize(obj_func, best_x, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Map back to full shape if symmetry was used
    full_thetas = map_func(best_x)
    
    # Ensure finite values
    if not np.all(np.isfinite(full_thetas)):
        return np.zeros((layers, n))
        
    return full_thetas
# EVOLVE-BLOCK-END
