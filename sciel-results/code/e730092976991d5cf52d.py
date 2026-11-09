import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0 # 5s margin

def _time_left():
    return _DEADLINE - time.time()

def _exact_maxcut(n, edges):
    dim = 1 << n
    best = 0
    for mask in range(dim):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _qaoa_state(n, edges, angles):
    dim = 1 << n
    # Precompute cut sizes for all states
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        # Vectorized cut size calculation
        # This is done once per angles call? No, better to precompute outside if possible, 
        # but angles changes. Cut sizes are static for a graph.
        # We will compute cut_sizes inside the main loop or pass it in.
        pass
    
    # Actually, let's precompute cut_sizes in the main function and pass it.
    
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Mixing layer
        # Apply e^{-i beta sum X}
        c = np.cos(beta)
        s = np.sin(beta)
        # Rotation matrix R = [[c, -is], [-is, c]]
        # We apply this to each qubit axis.
        tensor = state.reshape([2]*n)
        for i in range(n):
            # Move axis i to 0
            tensor = np.moveaxis(tensor, i, 0)
            # Apply rotation
            # tensor[0] is the component where qubit i is 0
            # tensor[1] is the component where qubit i is 1
            # new_0 = c * old_0 - 1j*s * old_1
            # new_1 = -1j*s * old_0 + c * old_1
            new_tensor = np.empty_like(tensor)
            new_tensor[0] = c * tensor[0] - 1j * s * tensor[1]
            new_tensor[1] = -1j * s * tensor[0] + c * tensor[1]
            tensor = new_tensor
            # Move axis 0 back to i
            tensor = np.moveaxis(tensor, 0, i)
        state = tensor.flatten()
        
    return state

def _expected_cut(n, cut_sizes, angles):
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state)**2
    return np.sum(probs * cut_sizes)

def angles(n, edges, p):
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    # Precompute cut sizes
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        # Efficient vectorized update
        # Create masks for bit u and bit v
        # This loop is O(m * 2^n). For n=10, m~50, this is ~50k ops. Fast.
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> u) & 1
        bv = (states >> v) & 1
        cut_sizes += (bi != bv).astype(np.float64)
        
    c_opt = _exact_maxcut(n, edges)
    
    def objective(x):
        # x is [gamma1, beta1, gamma2, beta2]
        a = x.reshape(p, 2)
        # Clip angles to reasonable range to help optimizer
        # QAOA angles are periodic, but typically in [0, pi]
        return -_expected_cut(n, cut_sizes, a)
        
    best_score = -np.inf
    best_angles = np.tile([0.35, 0.25], (p, 1)) # Fallback
    
    # Initial guesses
    guesses = []
    # Heuristic
    guesses.append([0.35, 0.25, 0.35, 0.25])
    guesses.append([0.5, 0.3, 0.5, 0.3])
    guesses.append([0.2, 0.4, 0.2, 0.4])
    guesses.append([0.7, 0.2, 0.7, 0.2])
    
    # Coarse Grid
    # Reduce grid size to save time, rely on NM for refinement
    grid_g = np.linspace(0.1, 1.5, 4)
    grid_b = np.linspace(0.1, 1.0, 4)
    for g1 in grid_g:
        for b1 in grid_b:
            for g2 in grid_g:
                for b2 in grid_b:
                    guesses.append([g1, b1, g2, b2])
                    
    # Random
    np.random.seed(42)
    for _ in range(20):
        guesses.append(np.random.uniform(0, np.pi, 4))
        
    # Optimize each guess
    for x0 in guesses:
        if _time_left() < 5.0:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000, 'maxfev': 1000})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_angles = res.x.reshape(p, 2)
        except Exception:
            pass
            
    return best_angles
