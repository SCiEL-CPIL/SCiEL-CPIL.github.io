"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 150.0
_SAFETY = 15.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    """Precompute cut values for all 2^n basis states."""
    num_states = 2**n
    cuts = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        mask = bits_u != bits_v
        cuts[mask] += 1
    return cuts

def _expected_cut_fast(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA using flip-index method.
    """
    N = 2**n
    psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    
    # Precompute flip indices for mixer
    flip_indices = [np.arange(N) ^ (1 << k) for k in range(n)]
    
    # Layer 1
    psi = psi * np.exp(-1j * gamma1 * cuts)
    if abs(beta1) > 1e-9:
        c = np.cos(beta1)
        s = -1j * np.sin(beta1)
        for k in range(n):
            psi = c * psi + s * psi[flip_indices[k]]
            
    # Layer 2
    psi = psi * np.exp(-1j * gamma2 * cuts)
    if abs(beta2) > 1e-9:
        c = np.cos(beta2)
        s = -1j * np.sin(beta2)
        for k in range(n):
            psi = c * psi + s * psi[flip_indices[k]]

    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _local_search(n, cuts, x0, maxiter=500, tol=1e-8):
    """Perform a local Nelder-Mead search starting from x0."""
    def obj(x):
        if _remaining() < 1.0:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut_fast(n, cuts, g1, b1, g2, b2)
        return -val
    
    try:
        res = minimize(
            obj, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut_fast(n, cuts, *x0)

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    best_x = [0.5, 0.5, 0.5, 0.5]
    best_val = _expected_cut_fast(n, cuts, *best_x)
    
    # Phase 1: Coarse grid search to find good regions
    # 8 points per parameter = 4096 evaluations
    if _remaining() > 50.0:
        grid_points = 8
        grid = np.meshgrid(
            np.linspace(0, np.pi, grid_points),
            np.linspace(0, np.pi/2, grid_points),
            np.linspace(0, np.pi, grid_points),
            np.linspace(0, np.pi/2, grid_points),
            indexing='ij'
        )
        grid_flat = np.column_stack([g.ravel() for g in grid])
        
        # Evaluate in batches
        batch_size = 256
        for i in range(0, len(grid_flat), batch_size):
            if _remaining() < 40.0:
                break
            batch = grid_flat[i:i+batch_size]
            for x0 in batch:
                try:
                    val = _expected_cut_fast(n, cuts, *x0)
                    if val > best_val:
                        best_val = val
                        best_x = x0.copy()
                except Exception:
                    continue
    
    # Phase 2: Local refinement from top candidates and physics-informed guesses
    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [0.7, 0.4, 0.7, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.2, 0.6, 0.2, 0.6],
        [1.0, 0.5, 1.0, 0.5],
        [0.4, 0.2, 0.4, 0.2],
        [0.8, 0.3, 0.8, 0.3],
        [0.1, 0.1, 0.1, 0.1],
        [0.6, 0.5, 0.6, 0.5],
        [0.9, 0.3, 0.9, 0.3],
        best_x,
    ]
    
    # Add nearby points to the best grid point
    for delta in [-0.2, -0.1, 0.1, 0.2]:
        for j in range(4):
            x0 = best_x.copy()
            x0[j] = np.clip(x0[j] + delta, 0, np.pi if j in [0,2] else np.pi/2)
            initial_guesses.append(x0)
    
    for x0 in initial_guesses:
        if _remaining() < 20.0:
            break
        try:
            refined_x, refined_val = _local_search(n, cuts, x0, maxiter=300, tol=1e-7)
            if refined_val > best_val:
                best_val = refined_val
                best_x = refined_x
        except Exception:
            continue
    
    # Phase 3: Differential Evolution if time permits
    if _remaining() > 25.0:
        try:
            def de_obj(params):
                if _remaining() < 1.0:
                    return 1e10
                return -_expected_cut_fast(n, cuts, *params)
            
            bounds = [(0, np.pi), (0, np.pi/2), (0, np.pi), (0, np.pi/2)]
            de_result = differential_evolution(
                de_obj, 
                bounds, 
                maxiter=30, 
                popsize=15,
                seed=42,
                polish=False,
                tol=1e-5
            )
            de_score = float(-de_result.fun)
            if de_score > best_val:
                best_val = de_score
                best_x = de_result.x
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if _remaining() > 10.0:
        try:
            final_x, final_val = _local_search(n, cuts, best_x, maxiter=1000, tol=1e-10)
            if final_val > best_val:
                best_val = final_val
                best_x = final_x
        except Exception:
            pass

    result = np.array([
        [best_x[0], best_x[1]],
        [best_x[2], best_x[3]]
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
