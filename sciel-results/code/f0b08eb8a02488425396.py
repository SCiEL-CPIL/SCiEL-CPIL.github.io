import numpy as np
import time
from scipy.optimize import minimize

def _build_idx_cache(n):
    """Precompute index arrays for fast gate application."""
    dim = 1 << n
    ry_cache = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
        idx1 = idx0 | mask
        ry_cache.append((idx0, idx1))
    
    cz_cache = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        idx_both = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
        cz_cache.append(idx_both)
        
    return {'ry': ry_cache, 'cz': cz_cache}

def _circuit_state(n, layers, params, idx_cache):
    """Compute state from circuit using precomputed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = idx_cache['ry'][q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = idx_cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
    return state

def _hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            H[b, b] += -coupling * (1 - 2 * bi) * (1 - 2 * bj)
    for i in range(n):
        for b in range(dim):
            bf = b ^ (1 << i)
            if b < bf:
                H[b, bf] -= field
                H[bf, b] -= field
    return H

def _energy_and_grad(n, layers, params, idx_cache, H, e_ground):
    """Compute energy and analytical gradient using parameter-shift rule."""
    dim = 1 << n
    state = _circuit_state(n, layers, params, idx_cache)
    energy = np.vdot(state, H @ state).real
    obj_val = (energy - e_ground) / n
    
    # Compute gradient using parameter-shift rule
    grad = np.zeros((layers, n), dtype=np.float64)
    shift = np.pi / 4.0
    
    for layer in range(layers):
        for q in range(n):
            # Forward shift
            params_plus = params.copy()
            params_plus[layer, q] += shift
            state_plus = _circuit_state(n, layers, params_plus, idx_cache)
            energy_plus = np.vdot(state_plus, H @ state_plus).real
            obj_plus = (energy_plus - e_ground) / n
            
            # Backward shift
            params_minus = params.copy()
            params_minus[layer, q] -= shift
            state_minus = _circuit_state(n, layers, params_minus, idx_cache)
            energy_minus = np.vdot(state_minus, H @ state_minus).real
            obj_minus = (energy_minus - e_ground) / n
            
            grad[layer, q] = (obj_plus - obj_minus) / (2 * shift)
    
    return obj_val, grad

def _mean_field_angle(n, coupling, field):
    """Compute mean-field optimal angle."""
    ratio = field / coupling
    s_target = ratio * n / (2 * (n - 1))
    if s_target > 1.0:
        return np.pi / 2
    elif s_target < -1.0:
        return -np.pi / 2
    else:
        return np.arcsin(s_target)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute
    idx_cache = _build_idx_cache(n)
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p, idx_cache)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    def obj_grad(xf):
        p = xf.reshape((layers, n))
        val, grad = _energy_and_grad(n, layers, p, idx_cache, H, e_ground)
        return val, grad.flatten()
    
    # Mean-field optimal angle
    base = _mean_field_angle(n, coupling, field)
    
    # Initial guesses
    guesses = []
    guesses.append(np.full((layers, n), base))
    guesses.append(np.full((layers, n), 0.0))
    guesses.append(np.full((layers, n), np.pi / 2))
    guesses.append(np.full((layers, n), -base))
    
    # Layer-dependent
    for scale in [0.5, 0.8, 1.2]:
        g = np.zeros((layers, n))
        for l in range(layers):
            g[l, :] = base * (0.5 + 0.5 * l / max(layers - 1, 1)) * scale
        guesses.append(g)
    
    # Alternating
    g = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            g[l, q] = base * (1 if q % 2 == 0 else -1)
    guesses.append(g)
    
    # Random guesses
    np.random.seed(42)
    for _ in range(3):
        g = np.random.uniform(-np.pi, np.pi, (layers, n))
        guesses.append(g)
        
    best_params = guesses[0].copy()
    best_val = obj(best_params.flatten())
    
    # Optimize from each guess with gradient
    for g in guesses:
        if time.time() > deadline - 10:
            break
        x0 = g.flatten()
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Polish from best with gradient
    if time.time() < deadline - 5:
        try:
            res = minimize(obj_grad, best_params.flatten(), method='L-BFGS-B', jac=True,
                         options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    return best_params
