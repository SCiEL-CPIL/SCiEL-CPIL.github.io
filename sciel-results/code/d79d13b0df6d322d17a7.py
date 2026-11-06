"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 115.0


def _time_left():
    return _DEADLINE - time.time()


def _build_cost_diag(n, edges):
    """
    Efficiently compute the diagonal of the MaxCut cost operator C.
    C(b) = number of edges crossing the cut defined by bitstring b.
    """
    num_states = 1 << n
    c_diag = np.zeros(num_states, dtype=np.float64)

    for state in range(num_states):
        cut = 0
        for u, v in edges:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut += 1
        c_diag[state] = cut
        
    return c_diag


def _apply_mixer(state, beta, n):
    """
    Apply exp(-i beta sum X_i) to state.
    This is a tensor product of Rx(beta) on each qubit.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    Rx = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    for axis in range(n):
        state_tensor = np.moveaxis(state_tensor, axis, 0)
        state_tensor = np.tensordot(Rx, state_tensor, axes=([1], [0]))
        state_tensor = np.moveaxis(state_tensor, 0, axis)
        
    return state_tensor.reshape(-1)


def _apply_cost(state, gamma, c_diag):
    """
    Apply exp(-i gamma C) to state.
    C is diagonal with entries c_diag.
    """
    phases = np.exp(-1j * gamma * c_diag)
    return state * phases


def _expected_cut(params, n, c_diag):
    """
    Compute expected cut for given QAOA parameters.
    params: [g1, b1, g2, b2]
    """
    dim = 1 << n
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    
    g1, b1, g2, b2 = params
    
    # Layer 1
    state = _apply_cost(state, g1, c_diag)
    state = _apply_mixer(state, b1, n)
    
    # Layer 2
    state = _apply_cost(state, g2, c_diag)
    state = _apply_mixer(state, b2, n)
    
    probs = np.abs(state) ** 2
    return float(np.dot(probs, c_diag))


def _optimize_angles(n, edges):
    """
    Optimize QAOA angles for a given graph.
    Returns array of shape (2, 2) with [gamma, beta] pairs.
    """
    c_diag = _build_cost_diag(n, edges)
    
    # Initial guess
    best_val = -np.inf
    best_params = np.array([0.5, 0.5, 0.5, 0.5])
    
    # Grid search
    # Use a slightly finer grid than the previous 5x5x5x5 but keep it manageable
    # 8^4 = 4096 evaluations. For n=10, this is ~40M ops. Should be fast.
    grid_size = 8
    grid = np.linspace(0.05, np.pi - 0.05, grid_size)
    
    count = 0
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if _time_left() < 5.0:
                        return best_params.reshape(2, 2)
                    params = [g1, b1, g2, b2]
                    val = _expected_cut(params, n, c_diag)
                    if val > best_val:
                        best_val = val
                        best_params = params
                    count += 1
    
    # Local refinement using Nelder-Mead
    if _time_left() > 10.0:
        def neg_cut(params):
            return -_expected_cut(params, n, c_diag)
        
        try:
            # Use a tighter tolerance for better precision
            res = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < neg_cut(best_params):
                best_params = res.x
        except Exception:
            pass
    
    # High-precision polish pass
    # If we have time, do another round of optimization starting from the best found
    # with even tighter tolerances to ensure we are at a local minimum of the negative cut
    if _time_left() > 5.0:
        def neg_cut_polish(params):
            return -_expected_cut(params, n, c_diag)
        
        try:
            res2 = minimize(neg_cut_polish, best_params.copy(), method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res2.fun < neg_cut_polish(best_params):
                best_params = res2.x
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
        return np.tile([0.5, 0.5], (p, 1))
    
    try:
        angles_2 = _optimize_angles(n, edges)
        
        if p == 2:
            return angles_2
        elif p < 2:
            return angles_2[:p]
        else:
            reps = (p + 1) // 2
            return np.tile(angles_2, (reps, 1))[:p]
            
    except Exception:
        return np.tile([0.5, 0.5], (p, 1))
