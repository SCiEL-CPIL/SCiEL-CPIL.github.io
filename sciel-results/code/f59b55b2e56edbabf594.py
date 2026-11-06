# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_cut_diag(n, edges):
    """Precompute the cut diagonal for all 2^n states using vectorized bit ops."""
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int64)
    C_diag = np.zeros(num_states, dtype=float)
    for (u, v) in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        C_diag += (bits_u != bits_v).astype(float)
    return C_diag

def _precompute_flip_masks(n):
    """Precompute partner indices for X operations."""
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int64)
    masks = []
    for i in range(n):
        mask = 1 << i
        partner = states ^ mask
        masks.append(partner)
    return masks

def _expected_cut_value(n, gammas, betas, C_diag, flip_masks):
    """
    Compute the expected cut value for a given QAOA circuit.
    Uses exact state simulation with precomputed flip masks.
    """
    num_states = 1 << n
    psi = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i*gamma*C): diagonal
        phase = np.exp(-1j * gamma * C_diag)
        psi = psi * phase
        
        # Apply exp(-i*beta*M) where M = sum_i X_i
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            partner = flip_masks[i]
            # In-place update is tricky because psi[partner] depends on psi.
            # We need a temporary copy or vectorized swap.
            # psi_new = c*psi - 1j*s*psi[partner]
            # Since partner is a permutation, we can do:
            psi_partner = psi[partner]
            psi = c * psi - 1j * s * psi_partner
    
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * C_diag)
    
    return expected_cut

def _exhaustive_maxcut(n, edges, C_diag):
    """Compute the exact maximum cut."""
    return float(np.max(C_diag))

def _optimize_angles_for_graph(n, edges, p, time_budget=100.0):
    """Optimize QAOA angles for a specific graph using grid search + Nelder-Mead."""
    start_time = time.time()
    
    C_diag = _build_cut_diag(n, edges)
    opt_cut = _exhaustive_maxcut(n, edges, C_diag)
    if opt_cut == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    flip_masks = _precompute_flip_masks(n)
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        try:
            val = _expected_cut_value(n, gammas, betas, C_diag, flip_masks)
            return -val / opt_cut
        except Exception:
            return 1e10

    # 1. Coarse Grid Search
    # Grid size 6 for each of 4 params -> 1296 evaluations.
    # This is fast enough for n<=10.
    grid_size = 6
    g_range = np.linspace(0.0, 1.6, grid_size)
    b_range = np.linspace(0.0, 1.6, grid_size)
    
    best_val = -1e10
    best_params = np.array([0.35, 0.25, 0.35, 0.25]) # Fallback
    
    time_limit_grid = start_time + time_budget * 0.3 # 30% for grid
    
    # Iterate over grid
    for g1 in g_range:
        for b1 in b_range:
            if time.time() > time_limit_grid:
                break
            for g2 in g_range:
                for b2 in b_range:
                    if time.time() > time_limit_grid:
                        break
                    params = np.array([g1, b1, g2, b2])
                    val = -objective(params)
                    if val > best_val:
                        best_val = val
                        best_params = params
    
    # 2. Refinement with Nelder-Mead
    time_limit_total = start_time + time_budget
    if time.time() < time_limit_total - 5.0:
        res = minimize(objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < objective(best_params):
            best_params = res.x
            
    # 3. Final Polish if time permits
    if time.time() < time_limit_total - 2.0:
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < objective(best_params):
            best_params = res.x

    best_angles = np.clip(best_params, -np.pi, np.pi)
    
    result = np.zeros((p, 2))
    result[:, 0] = best_angles[:p]
    result[:, 1] = best_angles[p:]
    
    return result

def angles(n, edges, p):
    start_time = time.time()
    time_budget = 140.0 # Leave 10s margin for 150s limit
    
    try:
        result = _optimize_angles_for_graph(n, edges, p, time_budget=time_budget)
    except Exception:
        result = np.tile([0.35, 0.25], (p, 1))
    
    return result
# EVOLVE-BLOCK-END
