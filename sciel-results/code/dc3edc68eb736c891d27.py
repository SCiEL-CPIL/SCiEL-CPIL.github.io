"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 115.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_mixer_factors(n):
    """Precompute bit masks for efficient mixer application."""
    return list(range(n))


def _build_cost_diag(n, edges):
    """
    Efficiently compute the diagonal of the MaxCut cost operator C.
    C(b) = number of edges crossing the cut defined by bitstring b.
    Uses vectorized XOR-based approach for speed.
    """
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int32)
    
    c_diag = np.zeros(num_states, dtype=np.float64)
    for u, v in edges:
        # XOR of bits u and v gives 1 if they differ
        diff = ((states >> u) ^ (states >> v)) & 1
        c_diag += diff
    
    return c_diag


def _apply_mixer(state, beta, n):
    """
    Apply exp(-i beta sum X_i) to state.
    This is a tensor product of Rx(beta) on each qubit.
    Uses efficient in-place tensor operations.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    
    shape = [2] * n
    state_tensor = state.reshape(shape).copy()
    
    # Precompute the 2x2 Rx matrix
    Rx00 = c
    Rx01 = -1j * s
    Rx10 = -1j * s
    Rx11 = c
    
    for axis in range(n):
        # Move target axis to front
        state_tensor = np.moveaxis(state_tensor, axis, 0)
        # Apply Rx: new[0,...] = Rx00*old[0,...] + Rx01*old[1,...]
        #            new[1,...] = Rx10*old[0,...] + Rx11*old[1,...]
        old0 = state_tensor[0]
        old1 = state_tensor[1]
        state_tensor[0] = Rx00 * old0 + Rx01 * old1
        state_tensor[1] = Rx10 * old0 + Rx11 * old1
        # Move axis back
        state_tensor = np.moveaxis(state_tensor, 0, axis)
        
    return state_tensor.reshape(-1)


def _apply_cost(state, gamma, c_diag):
    """
    Apply exp(-i gamma C) to state.
    C is diagonal with entries c_diag.
    """
    phases = np.exp(-1j * gamma * c_diag)
    return state * phases


def _expected_cut(params, n, c_diag, init_state=None):
    """
    Compute expected cut for given QAOA parameters.
    params: [g1, b1, g2, b2]
    """
    dim = 1 << n
    # Initial state |+>^n
    if init_state is None:
        state = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    else:
        state = init_state.copy()
    
    g1, b1, g2, b2 = params
    
    # Layer 1
    state = _apply_cost(state, g1, c_diag)
    state = _apply_mixer(state, b1, n)
    
    # Layer 2
    state = _apply_cost(state, g2, c_diag)
    state = _apply_mixer(state, b2, n)
    
    # Expected cut = <state| C |state> = sum |state_i|^2 c_diag_i
    probs = np.abs(state) ** 2
    return float(np.dot(probs, c_diag))


def _optimize_angles(n, edges):
    """
    Optimize QAOA angles for a given graph.
    Returns array of shape (2, 2) with [gamma, beta] pairs.
    """
    c_diag = _build_cost_diag(n, edges)
    dim = 1 << n
    init_state = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    
    # Physics-informed initial guess based on graph structure
    # For MaxCut, asymptotic QAOA: gamma_p ~ pi/(4*p) for large p, 
    # but for p=2, empirical values around gamma~0.5-0.8, beta~0.2-0.4
    # Use degree-based heuristic
    degrees = np.zeros(n, dtype=np.int32)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    max_deg = max(1, int(np.max(degrees))) if len(edges) > 0 else 1
    
    # Initial guess: gamma ~ pi/(2*max_deg), beta ~ pi/(4*max_deg)
    g_init = min(np.pi / (2.0 * max_deg), 1.0)
    b_init = min(np.pi / (4.0 * max_deg), 0.6)
    
    # Multi-start grid search with better resolution
    # Use 6 points for gamma, 5 for beta
    grid_g = np.linspace(0.05, min(1.2, np.pi * 0.5), 6)
    grid_b = np.linspace(0.05, min(0.7, np.pi * 0.3), 5)
    
    best_val = -np.inf
    best_params = np.array([g_init, b_init, g_init, b_init])
    
    # Evaluate at the physics-informed point first
    val = _expected_cut(best_params, n, c_diag, init_state)
    best_val = val
    
    # Grid search
    count = 0
    total_grid = len(grid_g) ** 2 * len(grid_b) ** 2
    for g1 in grid_g:
        for b1 in grid_b:
            for g2 in grid_g:
                for b2 in grid_b:
                    if _time_left() < 5.0:
                        break
                    params = np.array([g1, b1, g2, b2])
                    val = _expected_cut(params, n, c_diag, init_state)
                    if val > best_val:
                        best_val = val
                        best_params = params
                    count += 1
            if _time_left() < 5.0:
                break
        if _time_left() < 5.0:
            break
    
    # Local refinement using Nelder-Mead with tighter tolerances
    if _time_left() > 10.0:
        def neg_cut(params):
            return -_expected_cut(params, n, c_diag, init_state)
        
        try:
            # First pass: Nelder-Mead
            res = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < neg_cut(best_params):
                best_params = res.x
        except Exception:
            pass
    
    # Final high-precision polish: very tight Nelder-Mead
    if _time_left() > 5.0:
        def neg_cut(params):
            return -_expected_cut(params, n, c_diag, init_state)
        
        try:
            res = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < neg_cut(best_params):
                best_params = res.x
        except Exception:
            pass
    
    # Ensure parameters are within [0, pi]
    best_params = np.clip(best_params, 0.0, np.pi)
    
    return best_params.reshape(2, 2)


def angles(n, edges, p):
    """
    Return QAOA angles for the given graph.
    n: number of qubits
    edges: list of edges (u, v)
    p: number of layers (fixed to 2 in this task)
    """
    if _time_left() < 1.0:
        # Fallback to some reasonable default
        return np.tile([0.5, 0.5], (p, 1))
    
    try:
        # Optimize for p=2
        angles_2 = _optimize_angles(n, edges)
        
        if p == 2:
            return angles_2
        elif p < 2:
            return angles_2[:p]
        else:
            # Tile for p > 2 (though task says p=2)
            reps = (p + 1) // 2
            return np.tile(angles_2, (reps, 1))[:p]
            
    except Exception:
        # Fallback
        return np.tile([0.5, 0.5], (p, 1))
