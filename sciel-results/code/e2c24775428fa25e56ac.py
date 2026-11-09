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
    """Build Hamiltonian efficiently using Kronecker products."""
    dim = 1 << n
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        term = np.array([1.0])
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    # X terms: -field * sum X_i
    for i in range(n):
        term = np.array([1.0])
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    return H

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
    
    # Mean-field optimal angle
    base = _mean_field_angle(n, coupling, field)
    
    # Generate diverse initial guesses
    guesses = []
    guesses.append(np.full((layers, n), base))
    guesses.append(np.full((layers, n), 0.0))
    guesses.append(np.full((layers, n), np.pi / 2))
    guesses.append(np.full((layers, n), -base))
    
    # Layer-dependent patterns
    for scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
        g = np.zeros((layers, n))
        for l in range(layers):
            g[l, :] = base * (0.5 + 0.5 * l / max(layers - 1, 1)) * scale
        guesses.append(g)
    
    # Alternating patterns
    for sign_scale in [0.5, 0.8, 1.0]:
        g = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                g[l, q] = base * sign_scale * (1 if q % 2 == 0 else -1)
        guesses.append(g)
    
    # Random guesses with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(8):
        g = rng.uniform(-np.pi, np.pi, (layers, n))
        guesses.append(g)
    
    # Small perturbations around mean-field
    for _ in range(4):
        g = np.full((layers, n), base) + rng.normal(0, 0.1, (layers, n))
        guesses.append(g)
    
    best_params = guesses[0].copy()
    best_val = obj(best_params.flatten())
    
    # Sort guesses by initial objective value
    scored_guesses = []
    for g in guesses:
        v = obj(g.flatten())
        scored_guesses.append((v, g))
    scored_guesses.sort(key=lambda x: x[0])
    
    # Optimize from top guesses
    for i, (v0, g) in enumerate(scored_guesses):
        if time.time() > deadline - 15:
            break
        if v0 > best_val + 0.1:
            continue
        x0 = g.flatten()
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Polish with BFGS
    if time.time() < deadline - 10:
        try:
            res = minimize(obj, best_params.flatten(), method='BFGS',
                         options={'maxiter': 200, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    return best_params
