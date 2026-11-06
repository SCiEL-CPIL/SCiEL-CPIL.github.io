"""
QAOA p=2 graph-dependent angle optimizer.
Uses a global grid search followed by local polishing to maximize the expected cut value.
Optimized for speed via vectorized cost application.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global time management
_START_TIME = time.time()
# 150s limit, use 145s deadline for safety
_DEADLINE = _START_TIME + 145.0 

# Cache for computed values to handle repeated calls or internal retries efficiently
_value_cache = {}

def _get_elapsed():
    return time.time() - _START_TIME

def _build_state_data(n, edges):
    """
    Pre-compute bit patterns and cut counts for all 2^n states.
    Returns:
        states: array of state indices
        bits: (2^n, n) array of bits
        cut_counts: (2^n,) array of number of cut edges for each state
    """
    dim = 1 << n
    states = np.arange(dim)
    bits = np.zeros((dim, n), dtype=np.int8)
    for j in range(n):
        bits[:, j] = (states >> j) & 1
    
    # Pre-compute cut counts for each state
    # C(s) = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    # Z_u Z_v = 1 if bits are same, -1 if different.
    # (1 - Z_u Z_v)/2 is 0 if same, 1 if different.
    # So C(s) is the number of edges (u,v) such that bits[s, u] != bits[s, v].
    
    cut_counts = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        # Add 1 where bit u != bit v
        cut_counts += (bits[:, u] != bits[:, v]).astype(np.float64)
        
    return states, bits, cut_counts

def _compute_expected_cut(n, edges, gammas, betas, states, bits, cut_counts):
    """
    Compute the expected cut value for given QAOA angles.
    Uses vectorized state evolution.
    """
    dim = 1 << n
    # Initial state |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Pre-compute mixer masks for speed
    # We need to apply exp(-i beta sum X_i)
    # This is a product of single-qubit rotations.
    
    for layer in range(2):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply Cost Hamiltonian: exp(-i gamma * C)
        # Since C is diagonal, we just multiply by phase
        # Phase = exp(-i gamma * cut_counts)
        # Note: cut_counts is real
        if abs(gamma) > 1e-9:
            phases = np.exp(-1j * gamma * cut_counts)
            psi *= phases
        
        # Apply Mixer Hamiltonian: exp(-i beta * sum X_i)
        # R_x(beta) on each qubit
        # psi_new[s] = cos(beta) psi[s] - i sin(beta) psi[s ^ (1<<i)]
        
        c_b = np.cos(beta)
        s_b = np.sin(beta)
        
        if abs(beta) > 1e-9:
            for i in range(n):
                mask_bit = 1 << i
                # Indices where bit i is 0
                # Vectorized index selection
                # This creates a boolean mask or indices. 
                # For n<=10, dim<=1024, this is fast.
                # idx_0 are states with bit i = 0
                # idx_1 = idx_0 + mask_bit
                # We can construct idx_0 efficiently.
                # Alternatively, just use where.
                idx_0 = np.where((states & mask_bit) == 0)[0]
                idx_1 = idx_0 + mask_bit
                
                psi_0 = psi[idx_0]
                psi_1 = psi[idx_1]
                
                new_psi_0 = c_b * psi_0 - 1j * s_b * psi_1
                new_psi_1 = -1j * s_b * psi_0 + c_b * psi_1
                
                psi[idx_0] = new_psi_0
                psi[idx_1] = new_psi_1

    # Calculate Expected Cut Value
    # E[cut] = sum_{(u,v) in E} (1 - <Z_u Z_v>) / 2
    # We can just sum the cut_counts weighted by probabilities?
    # No, E[cut] = sum_s |psi[s]|^2 * C(s)
    # Because C(s) is the cut value of state s.
    
    probs = np.real(np.conj(psi) * psi)
    total_cut = np.sum(probs * cut_counts)
    
    return total_cut

def angles(n, edges, p):
    global _value_cache
    
    # Normalize edges for caching
    edge_set = frozenset(frozenset((u, v)) for u, v in edges)
    cache_key = (n, edge_set)
    
    if cache_key in _value_cache:
        return _value_cache[cache_key]
        
    # Pre-compute state data
    states, bits, cut_counts = _build_state_data(n, edges)
    
    # Define the objective function for the optimizer
    def objective(params):
        g1, b1, g2, b2 = params
        return -_compute_expected_cut(n, edges, [g1, g2], [b1, b2], states, bits, cut_counts)
        
    # 1. Global Grid Search
    # With the faster cost application, we can do a denser grid.
    # Let's stick to a reasonable size to ensure we have time for polishing.
    # 12x12x12x12 = 20736 evals.
    # Each eval is ~ O(2^n * n). For n=10, 1024*10 = 10k ops.
    # 20k evals * 10k ops = 200M ops. NumPy can handle this in < 5s.
    
    # Check time
    if time.time() > _DEADLINE - 30.0:
        # Emergency return default good params
        return np.array([[0.3, 0.1], [0.3, 0.1]])
        
    # Grid parameters
    # Ranges: Gamma [0.05, 0.7], Beta [0.02, 0.45]
    g_range = np.linspace(0.05, 0.7, 12)
    b_range = np.linspace(0.02, 0.45, 12)
    
    best_val = -np.inf
    best_params = None
    
    # Grid Search
    # We can vectorize the grid evaluation if we batch them, but simple loops are safer for time checking.
    # However, we can evaluate multiple points at once if we reshape psi? No, psi is state vector.
    # Let's just loop, but it's fast now.
    
    count = 0
    for g1 in g_range:
        for b1 in b_range:
            for g2 in g_range:
                for b2 in b_range:
                    if time.time() > _DEADLINE - 20.0:
                        break
                    count += 1
                    val = _compute_expected_cut(n, edges, [g1, g2], [b1, b2], states, bits, cut_counts)
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
                if time.time() > _DEADLINE - 20.0: break
            if time.time() > _DEADLINE - 20.0: break
        if time.time() > _DEADLINE - 20.0: break
        
    if best_params is None:
        # Fallback if grid didn't run or failed
        best_params = np.array([0.3, 0.1, 0.3, 0.1])
        best_val = _compute_expected_cut(n, edges, [0.3, 0.3], [0.1, 0.1], states, bits, cut_counts)
        
    # 2. Local Optimization (Polish)
    # Use BFGS starting from the best grid point
    if time.time() < _DEADLINE - 15.0:
        try:
            result = minimize(
                objective,
                best_params,
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            if result.success:
                res_val = -result.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = result.x
        except Exception:
            pass
            
    # 3. Nelder-Mead Polish (Robustness)
    if time.time() < _DEADLINE - 10.0:
        try:
            result_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if result_nm.success:
                res_val = -result_nm.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = result_nm.x
        except Exception:
            pass
    
    # 4. High-precision polish: L-BFGS-B with tight bounds and very tight tolerance
    if time.time() < _DEADLINE - 5.0:
        try:
            # Bounds: gammas in [0, pi/2], betas in [0, pi/2]
            bounds = [(0.0, np.pi/2), (0.0, np.pi/2), (0.0, np.pi/2), (0.0, np.pi/2)]
            result_lbfgs = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if result_lbfgs.success:
                res_val = -result_lbfgs.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = result_lbfgs.x
        except Exception:
            pass
            
    # 5. Multi-start from neighborhood of best point to escape potential local minima
    if time.time() < _DEADLINE - 2.0:
        try:
            # Try a few small perturbations around the best point
            # Use a grid of perturbations
            perturbations = []
            # Small steps
            for dg in [-0.02, 0.0, 0.02]:
                for db in [-0.02, 0.0, 0.02]:
                    perturbations.append([dg, db, dg, db])
                    perturbations.append([dg, db, -dg, -db])
                    
            # Deduplicate
            unique_perts = np.unique(np.array(perturbations), axis=0)
            
            for pert in unique_perts:
                if time.time() > _DEADLINE - 1.0:
                    break
                start = best_params + pert
                # Clamp to valid range
                start = np.clip(start, 0.0, np.pi/2)
                try:
                    res = minimize(
                        objective,
                        start,
                        method='L-BFGS-B',
                        bounds=[(0.0, np.pi/2)] * 4,
                        options={'maxiter': 100, 'gtol': 1e-8}
                    )
                    if res.success:
                        res_val = -res.fun
                        if res_val > best_val:
                            best_val = res_val
                            best_params = res.x
                except Exception:
                    pass
        except Exception:
            pass

    # 6. Final ultra-precise polish if time remains
    if time.time() < _DEADLINE - 0.5:
        try:
            result_final = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=[(0.0, np.pi/2)] * 4,
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            if result_final.success:
                res_val = -result_final.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = result_final.x
        except Exception:
            pass

    # Format output: (p, 2) array with rows (gamma, beta)
    out = np.zeros((p, 2))
    out[0, 0] = best_params[0]  # gamma 1
    out[0, 1] = best_params[1]  # beta 1
    if p > 1:
        out[1, 0] = best_params[2]  # gamma 2
        out[1, 1] = best_params[3]  # beta 2
        
    _value_cache[cache_key] = out
    return out
# EVOLVE-BLOCK-END
