import numpy as np
import time
import scipy.optimize

def _remaining(deadline):
    return max(0.0, deadline - time.time())

def _precompute_cuts(n, edges):
    """Precompute cost values (number of cut edges) for each basis state."""
    dim = 1 << n
    costs = np.zeros(dim, dtype=float)
    for state in range(dim):
        cut_count = 0
        for u, v in edges:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut_count += 1
        costs[state] = cut_count
    return costs

def _apply_beta(psi, beta, n, dim):
    """Apply exp(-i beta * sum X_i) using vectorized operations."""
    c = np.cos(beta)
    s = np.sin(beta)
    for i in range(n):
        bit = 1 << i
        # Vectorized: for each pair (state, state|bit) where bit i is 0
        # Create index arrays
        idx = np.arange(dim)
        mask = (idx & bit) == 0
        partners = idx[mask] | bit
        a0 = psi[idx[mask]]
        a1 = psi[partners]
        psi[idx[mask]] = c * a0 - 1j * s * a1
        psi[partners] = -1j * s * a0 + c * a1
    return psi

def _expected_cut(params, n, edges, p, costs, dim):
    """Compute expected cut value for given QAOA parameters."""
    gammas = params[0:p]
    betas = params[p:2*p]
    
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for k in range(p):
        # Apply cost unitary
        psi = psi * np.exp(-1j * gammas[k] * costs)
        # Apply mixer unitary
        psi = _apply_beta(psi, betas[k], n, dim)
    
    # Expected cut = sum(|psi_i|^2 * costs_i)
    probs = np.abs(psi)**2
    return np.sum(probs * costs)

def _optimize_4params(n, edges, p, costs, dim, deadline):
    """Optimize the 4 QAOA parameters to maximize expected cut."""
    def objective(params):
        return -_expected_cut(params, n, edges, p, costs, dim)
    
    # Physics-informed initial guesses based on asymptotic QAOA for MaxCut
    # For p=2, typical optimal angles are around gamma~0.5-1.0, beta~0.3-0.5
    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [0.7, 0.4, 0.7, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.2, 0.6, 0.2, 0.6],
        [1.0, 0.5, 1.0, 0.5],
        [0.4, 0.2, 0.4, 0.2],
        [0.8, 0.3, 0.8, 0.3],
        [0.1, 0.1, 0.1, 0.1],
    ]
    
    best_params = None
    best_val = np.inf
    
    # Use scipy.optimize.differential_evolution for global search
    bounds = [(0.0, np.pi)] * 4
    
    # First, do a quick local optimization from multiple starts
    for x0 in initial_guesses:
        if _remaining(deadline) < 5.0:
            break
        try:
            res = scipy.optimize.minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
    
    # If time remains, do a broader global search
    if _remaining(deadline) > 20.0 and best_params is not None:
        try:
            # Use differential evolution with limited iterations
            de_res = scipy.optimize.differential_evolution(
                objective, bounds, seed=42, maxiter=50, tol=1e-4,
                polish=False, init='sobol'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x
        except Exception:
            pass
    
    # Final polish
    if _remaining(deadline) > 3.0 and best_params is not None:
        try:
            res = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        return np.array([[0.5, 0.5], [0.5, 0.5]])
    
    gammas = best_params[0:p]
    betas = best_params[p:2*p]
    return np.array(list(zip(gammas, betas)))

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    dim = 1 << n
    costs = _precompute_cuts(n, edges)
    
    result = _optimize_4params(n, edges, p, costs, dim, deadline)
    
    return result
