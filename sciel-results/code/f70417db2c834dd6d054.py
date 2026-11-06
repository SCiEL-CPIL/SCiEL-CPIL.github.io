"""
QAOA p=2 graph-dependent angle optimizer.
Uses a global grid search over a graph-scaled parameter space followed by
a single high-precision Nelder-Mead refinement to escape local minima.
Optimized for speed via vectorized state evolution.
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
    cut_counts = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
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

    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]

        # Apply Cost Hamiltonian: exp(-i gamma * C)
        if abs(gamma) > 1e-9:
            phases = np.exp(-1j * gamma * cut_counts)
            psi *= phases

        # Apply Mixer Hamiltonian: exp(-i beta * sum X_i)
        c_b = np.cos(beta)
        s_b = np.sin(beta)

        if abs(beta) > 1e-9:
            for i in range(n):
                mask_bit = 1 << i
                idx_0 = np.where((states & mask_bit) == 0)[0]
                idx_1 = idx_0 + mask_bit

                psi_0 = psi[idx_0]
                psi_1 = psi[idx_1]

                new_psi_0 = c_b * psi_0 - 1j * s_b * psi_1
                new_psi_1 = -1j * s_b * psi_0 + c_b * psi_1

                psi[idx_0] = new_psi_0
                psi[idx_1] = new_psi_1

    # Calculate Expected Cut Value
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
    m = len(edges)

    # Define the objective function for the optimizer
    def objective(params):
        g1, b1, g2, b2 = params
        return -_compute_expected_cut(n, edges, [g1, g2], [b1, b2], states, bits, cut_counts)

    # Check time
    if time.time() > _DEADLINE - 30.0:
        # Emergency return default good params
        return np.array([[0.3, 0.1], [0.3, 0.1]])

    # Graph-scaled parameter space
    # Scale ranges based on graph density
    density = 2 * m / (n * (n - 1)) if n > 1 else 0.0
    
    # For denser graphs, optimal angles tend to be smaller
    # Scale factors: denser -> smaller angles
    scale_factor = max(0.5, min(1.0, 1.0 - 0.5 * density))
    
    # Grid parameters - coarsely spaced for global coverage
    # Gamma range: [0.05, 0.7] scaled by graph density
    g_max = 0.7 * scale_factor
    b_max = 0.45 * scale_factor
    
    # Use a moderately dense grid: 10x10x10x10 = 10000 points
    g_range = np.linspace(0.05, g_max, 10)
    b_range = np.linspace(0.02, b_max, 10)

    best_val = -np.inf
    best_params = None

    # Grid Search
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

    # Single high-precision Nelder-Mead refinement from the best grid point
    if time.time() < _DEADLINE - 10.0:
        try:
            # Use Nelder-Mead with tight tolerances for high-precision refinement
            result_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result_nm.success:
                res_val = -result_nm.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = result_nm.x
        except Exception:
            pass

    # Final ultra-precise polish if time remains
    if time.time() < _DEADLINE - 2.0:
        try:
            result_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
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
