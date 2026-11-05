"""QAOA parameter design: graph-dependent angle optimization via combinatorial probability."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _precompute_graph(n, edges):
    """Precompute adjacency, edge list, and cut values for all masks."""
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edge_list.append((i, j))
    
    dim = 1 << n
    # Precompute cut value for each mask
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cv = 0
        for i, j in edge_list:
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cv += 1
        cut_vals[mask] = cv
    
    return adj, edge_list, cut_vals, dim

def _compute_expected_cut_fast(params, n, cut_vals, dim):
    """
    Compute expected cut using direct state-vector simulation.
    For p=2 QAOA on MaxCut:
    1. Start with |+>^n
    2. Apply cost phase (diagonal)
    3. Apply mixer (product of single-qubit rotations)
    4. Apply cost phase
    5. Apply mixer
    6. Measure expected cut
    """
    gamma1, beta1, gamma2, beta2 = params
    
    # Initial state: uniform
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Layer 1: Cost operator (diagonal)
    phase1 = np.exp(-1j * gamma1 * cut_vals)
    state = state * phase1
    
    # Layer 1: Mixer operator
    def apply_mixer(state, beta, n):
        """Apply exp(-i beta sum X) to state vector, vectorized."""
        cb = np.cos(beta)
        sb = np.sin(beta)
        for k in range(n):
            bit_val = 1 << k
            masks = np.arange(dim)
            mask0 = masks[(masks & bit_val) == 0]
            mask1 = mask0 | bit_val
            a = state[mask0]
            b = state[mask1]
            state[mask0] = cb * a - 1j * sb * b
            state[mask1] = -1j * sb * a + cb * b
        return state
    
    state = apply_mixer(state, beta1, n)
    
    # Layer 2: Cost operator (diagonal)
    phase2 = np.exp(-1j * gamma2 * cut_vals)
    state = state * phase2
    
    # Layer 2: Mixer operator
    state = apply_mixer(state, beta2, n)
    
    # Expected cut
    probs = np.abs(state) ** 2
    expected_cut = np.dot(probs, cut_vals)
    
    return expected_cut

def _exact_maxcut(cut_vals):
    """Compute exact MaxCut value from precomputed cut values."""
    return float(np.max(cut_vals))

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Precompute graph properties
    adj, edge_list, cut_vals, dim = _precompute_graph(n, edges)
    
    # Calculate density for heuristic initialization
    max_edges = n * (n - 1) / 2
    density = len(edges) / max_edges if max_edges > 0 else 0.0
    
    # Exact MaxCut
    opt_cut = _exact_maxcut(cut_vals)
    if opt_cut == 0:
        return np.tile([0.35, 0.25], (p, 1))
        
    # Heuristic initial guess based on density
    if density < 0.3:
        init = np.array([0.35, 0.25, 0.25, 0.15])
    elif density < 0.6:
        init = np.array([0.40, 0.20, 0.30, 0.10])
    else:
        init = np.array([0.45, 0.15, 0.35, 0.05])
        
    best_params = init.copy()
    best_val = _compute_expected_cut_fast(init, n, cut_vals, dim)
    
    # Objective function to maximize
    def objective(params):
        if time.time() > deadline:
            return 1e10
        val = _compute_expected_cut_fast(params, n, cut_vals, dim)
        return -val  # Minimize negative
        
    # Strategy 1: L-BFGS-B from heuristic init
    try:
        res = minimize(objective, init, method='L-BFGS-B', 
                       options={'maxiter': 100, 'disp': False, 'maxls': 50})
        if time.time() < deadline:
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
    except Exception:
        pass
    
    # Strategy 2: Random restarts with L-BFGS-B
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(12345)
        n_restarts = 0
        while time.time() < deadline - 5.0 and n_restarts < 20:
            rand_init = rng.uniform(0.05, 1.5, size=4)
            rand_init = np.clip(rand_init, 0.01, 1.5)
            try:
                res = minimize(objective, rand_init, method='L-BFGS-B', 
                               options={'maxiter': 50, 'disp': False, 'maxls': 30})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()
            except Exception:
                pass
            n_restarts += 1
    
    # Strategy 3: Nelder-Mead from best found so far
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Differential Evolution for global search
    if time.time() < deadline - 15.0:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(0.01, 1.5), (0.01, 1.5), (0.01, 1.5), (0.01, 1.5)]
            res = differential_evolution(objective, bounds, maxiter=50, popsize=15,
                                         seed=42, tol=1e-6, polish=True)
            if time.time() < deadline:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 5: High-precision polish with tight tolerance
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 200, 'disp': False, 'maxls': 100, 'gtol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Format output
    # p=2, so we need 2 rows of (gamma, beta)
    # best_params is [gamma1, beta1, gamma2, beta2]
    out = np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ])
    
    return out

# EVOLVE-BLOCK-END
