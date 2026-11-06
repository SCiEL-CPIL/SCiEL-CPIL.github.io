"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations


def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut size for given QAOA angles using exact state simulation."""
    dim = 1 << n
    # Precompute edge eigenvalues for all states
    states = np.arange(dim)
    # For each state and each edge, compute Z_i Z_j eigenvalue (+1 same, -1 diff)
    edge_eigs = np.zeros((len(edges), dim), dtype=np.int8)
    for ei, (u, v) in enumerate(edges):
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            edge_eigs[ei, state] = 1 if bu == bv else -1
    
    # Precompute mixer masks
    masks = [1 << k for k in range(n)]
    
    # Initial state: |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost term: exp(i * gamma/2 * sum_{(i,j) in E} Z_i Z_j)
        # sum Z_i Z_j for each state
        sum_eigs = edge_eigs.sum(axis=0).astype(float)
        phases = gamma / 2.0 * sum_eigs
        psi = psi * np.exp(1j * phases)
        
        # Apply mixer: exp(-i beta * sum X_k)
        c = np.cos(beta)
        s = np.sin(beta)
        for mask in masks:
            new_psi = np.empty_like(psi)
            for state in range(dim):
                if state & mask == 0:
                    partner = state | mask
                    new_psi[state] = c * psi[state] - 1j * s * psi[partner]
                    new_psi[partner] = c * psi[partner] - 1j * s * psi[state]
            psi = new_psi
    
    # Compute expected cut
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for ei, (u, v) in enumerate(edges):
        corr = (edge_eigs[ei].astype(float) * probs).sum()
        expected_cut += (1.0 - corr) / 2.0
    
    return expected_cut


def _classical_optimum(n, edges):
    """Find exact MaxCut optimum by brute force."""
    dim = 1 << n
    states = np.arange(dim)
    best = 0
    for (u, v) in edges:
        mask_u = 1 << u
        mask_v = 1 << v
        # For each state, check if bits u and v differ
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        diff_mask = bits_u != bits_v
        # We need to count for each state how many edges it cuts
        # Better approach: build cut count array
    # Vectorized approach
    cut_counts = np.zeros(dim, dtype=np.int32)
    for (u, v) in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        cut_counts += (bits_u != bits_v).astype(np.int32)
    best = cut_counts.max()
    return best


def _spectral_init(n, edges):
    """Initialize angles using spectral/degree-based heuristics."""
    # Compute degrees
    degrees = np.zeros(n)
    for (u, v) in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    # Simple heuristic: gamma ~ 1/sqrt(max_degree), beta ~ 1/n
    max_deg = max(degrees) if len(degrees) > 0 else 1
    gamma_init = 1.0 / np.sqrt(max(max_deg, 1))
    beta_init = 1.0 / n
    
    return np.array([[gamma_init, beta_init], [gamma_init * 0.8, beta_init * 0.7]])


def angles(n, edges, p):
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 100.0  # Leave 20s margin
    
    # Compute classical optimum
    opt = _classical_optimum(n, edges)
    if opt == 0:
        return np.zeros((p, 2))
    
    # Initialize with spectral heuristic
    init = _spectral_init(n, edges)
    
    def objective(params):
        if time.time() > deadline:
            return 0.0
        gammas = params[:p]
        betas = params[p:]
        ec = _expected_cut(n, edges, gammas, betas)
        return -ec / opt  # Minimize negative ratio
    
    # Try multiple starting points
    best_angles = init
    best_val = -1.0
    
    # Starting points: spectral + several random
    starts = [init.flatten()]
    rng = np.random.RandomState(42)
    for _ in range(5):
        starts.append(rng.uniform(0, np.pi, size=2 * p))
    
    # First pass: Nelder-Mead from each start
    for start in starts:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                best_angles = np.column_stack([best_params[:p], best_params[p:]])
        except Exception:
            pass
    
    # Second pass: BFGS polish from best result
    if best_val < 0:
        best_params = best_angles.flatten()
    else:
        best_params = best_angles.flatten()
    
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, best_params, method='BFGS', 
                            options={'maxiter': 100, 'gtol': 1e-6})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                best_angles = np.column_stack([best_params[:p], best_params[p:]])
        except Exception:
            pass
    
    # Third pass: L-BFGS-B polish
    if time.time() < deadline - 10:
        try:
            result = minimize(objective, best_params, method='L-BFGS-B', 
                            options={'maxiter': 200, 'gtol': 1e-6})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                best_angles = np.column_stack([best_params[:p], best_params[p:]])
        except Exception:
            pass
    
    if best_val < 0:
        # Fallback to initial
        return init
    
    return best_angles
# EVOLVE-BLOCK-END
