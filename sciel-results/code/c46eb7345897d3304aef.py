import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_circuit_indices_cache = {}

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
        mask = 1 << i
        for state in range(dim):
            flipped = state ^ mask
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

def _get_circuit_indices(n, layers):
    key = (n, layers)
    if key in _circuit_indices_cache:
        return _circuit_indices_cache[key]
    
    dim = 1 << n
    idx = np.arange(dim)
    
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_indices.append((idx_0, idx_1))
        
    cz_indices = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_indices.append(both_set)
        
    _circuit_indices_cache[key] = (ry_indices, cz_indices)
    return _circuit_indices_cache[key]

def _circuit_state(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    ry_indices, cz_indices = _get_circuit_indices(n, layers)
    
    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = ry_indices[q]
            
            v0 = state[idx_0].copy()
            v1 = state[idx_1].copy()
            
            state[idx_0] = c * v0 - s * v1
            state[idx_1] = s * v0 + c * v1
        
        if layer < layers - 1:
            for i in range(n - 1):
                mask_idx = cz_indices[i]
                state[mask_idx] *= -1.0
    
    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return np.real(np.vdot(psi, H @ psi))

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _circuit_energy_grad(n, coupling, field, thetas, layers):
    """Compute gradient of circuit energy w.r.t. thetas using finite differences."""
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    Hpsi = H @ psi
    
    grad = np.zeros_like(thetas)
    eps = 1e-7
    
    for l in range(layers):
        for q in range(n):
            thetas_p = thetas.copy()
            thetas_p[l, q] += eps
            psi_p = _circuit_state(n, thetas_p, layers)
            e_p = np.real(np.vdot(psi_p, H @ psi_p))
            
            thetas_m = thetas.copy()
            thetas_m[l, q] -= eps
            psi_m = _circuit_state(n, thetas_m, layers)
            e_m = np.real(np.vdot(psi_m, H @ psi_m))
            
            grad[l, q] = (e_p - e_m) / (2 * eps)
    
    return grad / n  # Mean per spin

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)
    
    def obj_grad(p):
        thetas = p.reshape((layers, n))
        return _circuit_energy_grad(n, coupling, field, thetas, layers).flatten()
    
    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # Generate initial points
    inits = []
    
    # 1. Zero (Ferromagnetic limit)
    inits.append(np.zeros(layers * n))
    
    # 2. Pi/2 (Paramagnetic limit)
    inits.append(np.full(layers * n, np.pi/2))
    
    # 3. Physics-informed mean-field estimate
    ratio = field / coupling if coupling > 0 else 1.0
    theta_mf = 2.0 * np.arctan(ratio)
    inits.append(np.full(layers * n, theta_mf))
    
    # 4. Alternating pattern (Antiferromagnetic hint)
    alt = np.zeros(layers * n).reshape(layers, n)
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                alt[l, q] = theta_mf
            else:
                alt[l, q] = -theta_mf
    inits.append(alt.flatten())
    
    # 5. Random restarts
    num_random = 30 if n <= 4 else 20
    
    for i in range(num_random):
        if time.time() > deadline - 25:
            break
        if i % 2 == 0:
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        else:
            x_init = theta_mf + np.random.normal(0, 0.8, size=(layers * n))
        inits.append(x_init)
    
    # Sort inits by objective value
    init_objs = []
    for x_init in inits:
        if time.time() > deadline - 20:
            break
        try:
            v = obj(x_init)
            init_objs.append((v, x_init))
        except Exception:
            continue
    
    init_objs.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B from each start with gradient
    for v_init, x_init in init_objs:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(obj, x_init, jac=obj_grad, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12,
                                   'maxls': 60})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Final polish with Nelder-Mead if time remains
    if time.time() < deadline - 10.0:
        try:
            res_nm = minimize(obj, best_x, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass
            
    # One more L-BFGS-B polish with tighter tolerance
    if time.time() < deadline - 3.0:
        try:
            res = minimize(obj, best_x, jac=obj_grad, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
