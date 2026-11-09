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

def _circuit_state(n, layers, params, idx_cache, store_states=False):
    """Compute state from circuit using precomputed indices.
    If store_states is True, returns (state, forward_states) where forward_states[l][q] is the state BEFORE applying Ry(l, q).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    forward_states = None
    if store_states:
        forward_states = [[None for _ in range(n)] for _ in range(layers)]

    for layer in range(layers):
        for q in range(n):
            if store_states:
                # Only store if needed for gradient
                forward_states[layer][q] = state.copy()
            
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = idx_cache['ry'][q]
            
            # Optimization: Avoid copy if possible? 
            # state[idx0] and state[idx1] are disjoint views.
            # But we need old values to compute new values.
            old0 = state[idx0]
            old1 = state[idx1]
            
            # We must be careful not to overwrite idx0 before using it for idx1 calculation?
            # idx0 and idx1 are disjoint sets of indices.
            # state[idx0] = c*old0 - s*old1
            # state[idx1] = s*old0 + c*old1
            # Since idx0 and idx1 are disjoint, we can update in place safely 
            # if we store old0 and old1 first.
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = idx_cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
    if store_states:
        return state, forward_states
    return state

def _hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # ZZ terms
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            H[b, b] += -coupling * (1 - 2 * bi) * (1 - 2 * bj)
    # X terms
    for i in range(n):
        for b in range(dim):
            bf = b ^ (1 << i)
            if b < bf:
                H[b, bf] -= field
                H[bf, b] -= field
    return H

def _energy_and_grad_adjoint(n, layers, params, idx_cache, H, e_ground):
    """Compute energy and analytical gradient using adjoint method."""
    state, fwd_states = _circuit_state(n, layers, params, idx_cache, store_states=True)
    energy = np.vdot(state, H @ state).real
    obj_val = (energy - e_ground) / n
    
    # Initialize adjoint
    # dE/dpsi* = H psi. Objective is (E - E0)/n, so dObj/dpsi* = H psi / n
    adj = (H @ state) / n
    
    grad = np.zeros((layers, n), dtype=np.float64)
    
    # Backward pass
    for layer in range(layers - 1, -1, -1):
        # Undo Ry gates in reverse order
        for q in range(n - 1, -1, -1):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = idx_cache['ry'][q]
            
            psi_in = fwd_states[layer][q]
            
            # dRy/dtheta = 0.5 * [[-sin(theta/2), -cos(theta/2)], [cos(theta/2), -sin(theta/2)]]
            m_psi0 = -0.5 * s * psi_in[idx0] - 0.5 * c * psi_in[idx1]
            m_psi1 = 0.5 * c * psi_in[idx0] - 0.5 * s * psi_in[idx1]
            
            term = np.vdot(adj[idx0], m_psi0) + np.vdot(adj[idx1], m_psi1)
            grad[layer, q] = 2.0 * term.real
            
            # Update adjoint: adj_new = Ry^dag @ adj
            old_adj0 = adj[idx0].copy()
            old_adj1 = adj[idx1].copy()
            adj[idx0] = c * old_adj0 + s * old_adj1
            adj[idx1] = -s * old_adj0 + c * old_adj1
        
        # Undo CZ gates if not the first layer
        if layer > 0:
            for q in range(n - 2, -1, -1):
                idx_both = idx_cache['cz'][q]
                adj[idx_both] = -adj[idx_both]
                
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
    deadline = start_time + 145.0  # 5s safety margin
    
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
        val, grad = _energy_and_grad_adjoint(n, layers, p, idx_cache, H, e_ground)
        return val, grad.flatten()
    
    # Mean-field optimal angle
    base = _mean_field_angle(n, coupling, field)
    
    # Generate diverse initial guesses
    guesses = []
    
    # 1. Uniform mean-field
    guesses.append(np.full((layers, n), base))
    
    # 2. Uniform zero
    guesses.append(np.full((layers, n), 0.0))
    
    # 3. Uniform pi/2
    guesses.append(np.full((layers, n), np.pi / 2))
    
    # 4. Uniform -base
    guesses.append(np.full((layers, n), -base))
    
    # 5. Layer-dependent scaling
    for scale in [0.5, 0.8, 1.2]:
        g = np.zeros((layers, n))
        for l in range(layers):
            # Interpolate from 0.5*base to 1.5*base across layers
            factor = 0.5 + l / max(layers - 1, 1)
            g[l, :] = base * factor * scale
        guesses.append(g)
        
    # 6. Alternating signs
    g = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            g[l, q] = base * (1 if q % 2 == 0 else -1)
    guesses.append(g)
    
    # 7. Random guesses
    np.random.seed(42)
    for _ in range(5):
        g = np.random.uniform(-np.pi, np.pi, (layers, n))
        guesses.append(g)
        
    # 8. Small perturbations around base
    for delta in [0.1, -0.1, 0.2, -0.2]:
        guesses.append(np.full((layers, n), base + delta))

    # Evaluate all guesses to sort them
    guess_vals = []
    for g in guesses:
        try:
            v = obj(g.flatten())
            guess_vals.append((v, g))
        except:
            guess_vals.append((np.inf, g))
            
    # Sort by objective value
    guess_vals.sort(key=lambda x: x[0])
    
    best_params = guess_vals[0][1].copy()
    best_val = guess_vals[0][0]
    
    # Optimize from top K guesses
    max_guesses_to_optimize = 8
    for i in range(min(max_guesses_to_optimize, len(guess_vals))):
        if time.time() > deadline - 15:
            break
            
        x0 = guess_vals[i][1].flatten()
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(obj_grad, best_params.flatten(), method='L-BFGS-B', jac=True,
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except Exception:
            pass
            
    return best_params
