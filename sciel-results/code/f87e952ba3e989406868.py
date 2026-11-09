import numpy as np
import time
from scipy.optimize import minimize

# Precomputed index caches for fast gate application
_IDX_CACHE = {}

def _get_idx_cache(n):
    """Get or build precomputed index arrays for fast gate application."""
    if n not in _IDX_CACHE:
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
        
        _IDX_CACHE[n] = {'ry': ry_cache, 'cz': cz_cache}
    return _IDX_CACHE[n]

def _circuit_state(n, layers, params):
    """Compute state from circuit using precomputed indices."""
    cache = _get_idx_cache(n)
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = cache['ry'][q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
    return state

def _hamiltonian(n, coupling, field):
    """Build Hamiltonian."""
    dim = 1 << n
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    
    H = np.zeros((dim, dim), dtype=np.float64)
    
    for i in range(n - 1):
        term = np.array([1.0])
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    for i in range(n):
        term = np.array([1.0])
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    return H

def _closed_form_angle(coupling, field):
    """
    Closed-form optimal Ry angle for Ising model.
    
    For H = -J*ZZ - h*X on a single qubit (effective), the optimal rotation
    to minimize energy satisfies tan(theta/2) = h/J, so theta = 2*arctan(h/J).
    This is equivalent to arccos(J/sqrt(J^2+h^2)).
    """
    if coupling == 0:
        return np.pi / 2
    ratio = field / coupling
    # theta = 2 * arctan(ratio)
    # Clamp ratio to avoid numerical issues
    ratio = np.clip(ratio, -1e6, 1e6)
    return 2.0 * np.arctan(ratio)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 150s limit, use 140s
    
    # Compute closed-form optimal angle
    base_angle = _closed_form_angle(coupling, field)
    
    # Build Hamiltonian and compute ground state energy
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    # Generate diverse initial points
    inits = []
    
    # Closed-form uniform
    inits.append(np.full((layers, n), base_angle))
    
    # Uniform variations
    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        inits.append(np.full((layers, n), base_angle * scale))
    
    # Layer-decaying patterns
    for decay in [0.5, 0.7, 0.85, 1.0]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (decay ** l)
        inits.append(v)
    
    # Layer-increasing patterns
    for growth in [1.1, 1.3, 1.5]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (growth ** l)
        inits.append(v)
    
    # Alternating patterns
    for amp in [0.1, 0.2, 0.3]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (1.0 + amp * (-1) ** l)
        inits.append(v)
    
    # Per-qubit variations around base
    rng = np.random.default_rng(42)
    for _ in range(5):
        v = np.full((layers, n), base_angle)
        v += rng.normal(0, 0.1, size=(layers, n))
        inits.append(v)
    
    # Evaluate all initial points and keep the best few
    scored_inits = []
    for v in inits:
        if time.time() > deadline - 80:
            break
        val = obj(v.flatten())
        scored_inits.append((val, v))
    
    scored_inits.sort(key=lambda x: x[0])
    
    best_val = scored_inits[0][0]
    best_params = scored_inits[0][1].copy()
    
    # Optimize from top 3 starting points
    top_k = min(3, len(scored_inits))
    for k in range(top_k):
        if time.time() > deadline - 30:
            break
        
        start_x = scored_inits[k][1].flatten()
        
        # Nelder-Mead for global-ish exploration
        try:
            res = minimize(obj, start_x, method='Nelder-Mead',
                         options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
        
        # BFGS for local refinement
        if time.time() > deadline - 15:
            continue
        try:
            res = minimize(obj, best_params.flatten(), method='BFGS',
                         options={'maxiter': 200, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Ensure finite output
    if not np.all(np.isfinite(best_params)):
        best_params = np.full((layers, n), base_angle)
    
    return best_params
