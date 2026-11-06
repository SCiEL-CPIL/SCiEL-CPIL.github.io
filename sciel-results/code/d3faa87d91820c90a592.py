"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Global deadline management
_T0 = time.time()
_DEADLINE = 145.0  # Safety margin from 150s
_ANGLE_CACHE = {}

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _build_cost_diag(n, edges):
    """
    Efficiently compute the diagonal of the MaxCut Hamiltonian C.
    C(b) = number of edges crossing the cut defined by bitstring b.
    We compute this for all 2^n bitstrings.
    """
    num_states = 2 ** n
    c_diag = np.zeros(num_states)
    
    for mask in range(num_states):
        cut_val = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut_val += 1
        c_diag[mask] = cut_val
        
    return c_diag

def _apply_mixer(state, beta, n):
    """
    Apply exp(-i beta sum X) to state.
    sum X = sum_j X_j.
    exp(-i beta sum X) = prod_j (cos(beta) I_j - i sin(beta) X_j).
    This is a tensor product of identical single-qubit gates.
    Gate U = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    """
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    gate = np.array([[c, s], [s, c]], dtype=complex)
    
    state_tensor = state.reshape([2] * n)
    
    for j in range(n):
        state_tensor = np.moveaxis(state_tensor, j, 0)
        orig_shape = state_tensor.shape
        state_tensor = state_tensor.reshape(-1, 2)
        state_tensor = gate @ state_tensor
        state_tensor = state_tensor.reshape(orig_shape)
        state_tensor = np.moveaxis(state_tensor, 0, j)
        
    return state_tensor.reshape(-1)

def _expected_cut_value(n, edges, gammas, betas, c_diag=None):
    """
    Compute expected cut for given parameters.
    """
    if c_diag is None:
        c_diag = _build_cost_diag(n, edges)
        
    # Initial state |+>^n
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    
    for k in range(len(gammas)):
        # Apply Cost: exp(-i gamma C)
        phase = np.exp(-1j * gammas[k] * c_diag)
        state = state * phase
        
        # Apply Mixer: exp(-i beta sum X)
        state = _apply_mixer(state, betas[k], n)
        
    # Compute expectation <psi| C |psi>
    prob = np.abs(state)**2
    return np.dot(prob, c_diag)

def _optimize_angles(n, edges, p):
    """
    Optimize angles for a specific graph.
    Uses a combination of grid search and local refinement.
    """
    if _time_left() < 2.0:
        return np.tile([0.35, 0.25], (p, 1))
        
    c_diag = _build_cost_diag(n, edges)
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        val = _expected_cut_value(n, edges, gammas, betas, c_diag)
        return -val  # Minimize negative expected cut
        
    best_val = np.inf
    best_params = None
    
    # Strategy 1: Coarse grid search over a reasonable range
    # For p=2, we have 4 parameters. A 5x5x5x5 grid = 625 evaluations.
    # Each evaluation takes ~1-2ms for n<=10. Total ~1-2s. Acceptable.
    
    # Define grid points
    # Based on QAOA literature, optimal angles are often in [0, pi]
    # Let's use a coarse grid first.
    grid_size = 6
    gamma_vals = np.linspace(0.1, 2.5, grid_size)
    beta_vals = np.linspace(0.1, 1.5, grid_size)
    
    # Generate all combinations
    grid_points = []
    for g1 in gamma_vals:
        for b1 in beta_vals:
            for g2 in gamma_vals:
                for b2 in beta_vals:
                    grid_points.append([g1, b1, g2, b2])
    
    # Evaluate grid points
    for pt in grid_points:
        if _time_left() < 1.0:
            break
        try:
            val = objective(pt)
            if val < best_val:
                best_val = val
                best_params = np.array(pt)
        except Exception:
            continue
    
    # Strategy 2: Local refinement from the best grid point
    if best_params is not None and _time_left() > 2.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Strategy 3: If time allows, try a few more random starts with local search
    if _time_left() > 5.0:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if _time_left() < 1.0:
                break
            start = rng.uniform(0.1, 2.5, size=2*p)
            try:
                res = minimize(objective, start, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                continue
    
    if best_params is None:
        return np.tile([0.35, 0.25], (p, 1))
        
    gammas = best_params[:p]
    betas = best_params[p:]
    return np.column_stack((gammas, betas))

def angles(n, edges, p):
    """
    Entry point.
    Returns (p, 2) array of angles.
    """
    # Normalize edges for caching
    norm_edges = tuple(sorted((min(u, v), max(u, v)) for u, v in edges))
    key = (n, norm_edges, p)
    
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
        
    # If time is critical, return default
    if _time_left() < 1.0:
        res = np.tile([0.35, 0.25], (p, 1))
        return res
        
    # Optimize
    res = _optimize_angles(n, edges, p)
    
    # Cache result
    _ANGLE_CACHE[key] = res
    
    return res
# EVOLVE-BLOCK-END
