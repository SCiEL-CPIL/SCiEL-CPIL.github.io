"""Improved optimizer for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_psi0_cache = {}
_idx_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0]
    psi0 = eigvecs[:, 0]
    _e0_cache[key] = (e0, psi0)
    return e0, psi0

def _get_indices(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    ry_idx = []
    for q in range(n):
        mask = 1 << q
        idx = np.arange(dim)
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_idx.append((idx_0, idx_1))
    cz_idx = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        idx = np.arange(dim)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_idx.append(both_set)
    _idx_cache[key] = (ry_idx, cz_idx)
    return _idx_cache[key]

def _circuit_state_fast(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    ry_idx, cz_idx = _get_indices(n)

    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = ry_idx[q]
            vals_0 = state[idx_0]
            vals_1 = state[idx_1]
            state[idx_0] = c * vals_0 - s * vals_1
            state[idx_1] = s * vals_0 + c * vals_1

        if layer < layers - 1:
            for i in range(n - 1):
                both_set = cz_idx[i]
                state[both_set] *= -1.0

    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _mps_informed_init(n, coupling, field, layers, psi0):
    """Fit Ry angles to match single-qubit expectation values of ground state."""
    dim = 1 << n
    thetas = np.zeros((layers, n))
    
    for q in range(n):
        # Compute <Z_q> and <X_q> from psi0
        ez = 0.0
        ex = 0.0
        mask = 1 << q
        for state in range(dim):
            bit = (state >> q) & 1
            amp = psi0[state]
            ez += np.real(amp * np.conj(amp) * (1.0 if bit == 0 else -1.0))
            flipped = state ^ mask
            ex += np.real(np.conj(amp) * psi0[flipped])
        
        # Match to Ry(theta)|0> state: <Z> = cos(theta), <X> = sin(theta)
        theta = np.arctan2(ex, ez)
        
        # Distribute equally across layers
        per_layer = theta / layers
        for layer in range(layers):
            thetas[layer, q] = per_layer
    
    return thetas

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    e0, psi0 = _exact_ground(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)

    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # MPS-inspired initialization
    try:
        mps_thetas = _mps_informed_init(n, coupling, field, layers, psi0)
        mps_x = mps_thetas.flatten()
        mps_val = obj(mps_x)
        if mps_val < best_val:
            best_val = mps_val
            best_x = mps_x
    except Exception:
        pass
    
    # Determine number of restarts based on time budget
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if n <= 4:
        num_restarts = min(30, int(remaining / 3.0))
    else:
        num_restarts = min(15, int(remaining / 5.0))
    
    # Generate initial points
    inits = []
    
    # 1. Zero
    inits.append(np.zeros(layers * n))
    
    # 2. Pi/2
    inits.append(np.full(layers * n, np.pi/2))
    
    # 3. Small angles
    inits.append(np.full(layers * n, 0.1))
    inits.append(np.full(layers * n, -0.1))
    
    # 4. MPS-inspired
    try:
        mps_thetas2 = _mps_informed_init(n, coupling, field, layers, psi0)
        inits.append(mps_thetas2.flatten())
    except Exception:
        pass
    
    # 5. Physics-informed random restarts
    if field > 1.0:
        center = np.pi/2
        spread = 0.6
    else:
        center = 0.0
        spread = 0.6

    for i in range(num_restarts):
        if i % 3 == 0:
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        elif i % 3 == 1:
            x_init = center + np.random.normal(0, spread, size=(layers * n))
        else:
            x_init = center + np.random.normal(0, spread * 1.5, size=(layers * n))
        
        inits.append(x_init)
        
        if time.time() > deadline - 20:
            break

    # Run optimization from each start
    for x_init in inits:
        if time.time() > deadline - 15:
            break
        
        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue

    # Final high-precision polish
    if time.time() < deadline - 8.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # One more ultra-precise pass if time remains
    if time.time() < deadline - 3.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
