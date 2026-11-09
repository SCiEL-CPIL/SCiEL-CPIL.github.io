"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _precompute_cuts(n, edges):
    """
    Precompute the cut value for every basis state |x> for the given graph.
    Returns a numpy array of shape (2^n,) containing the number of cut edges.
    """
    num_states = 1 << n
    cuts = np.zeros(num_states, dtype=np.float64)
    edge_masks = [(1 << i, 1 << j) for i, j in edges]
    
    for x in range(num_states):
        c = 0
        for mask_i, mask_j in edge_masks:
            bi = (x & mask_i) != 0
            bj = (x & mask_j) != 0
            if bi != bj:
                c += 1
        cuts[x] = c
    return cuts

def _expected_cut_fast(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute the expected cut for the QAOA circuit with p=2.
    """
    num_states = 1 << n
    amps = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Helper to apply mixer exp(-i beta sum X_i)
    def apply_mixer(amps, beta):
        cb = np.cos(beta)
        sb = -1j * np.sin(beta)
        for i in range(n):
            bit = 1 << i
            range_arr = np.arange(num_states)
            mask = (range_arr & bit) == 0
            idx_0 = range_arr[mask]
            idx_1 = idx_0 + bit
            
            old_0 = amps[idx_0].copy()
            old_1 = amps[idx_1].copy()
            
            amps[idx_0] = cb * old_0 + sb * old_1
            amps[idx_1] = sb * old_0 + cb * old_1
            
    # Apply Layer 1: Cost
    phase1 = np.exp(-1j * gamma1 * cuts)
    amps = amps * phase1
    
    # Apply Layer 1: Mixer
    apply_mixer(amps, beta1)
    
    # Apply Layer 2: Cost
    phase2 = np.exp(-1j * gamma2 * cuts)
    amps = amps * phase2
    
    # Apply Layer 2: Mixer
    apply_mixer(amps, beta2)
    
    # Measure
    probs = np.abs(amps) ** 2
    expected_cut = np.sum(probs * cuts)
    return expected_cut

def _exact_optimum(n, edges):
    """
    Compute the exact maximum cut size for the given graph.
    """
    num_states = 1 << n
    best = 0
    edge_masks = [(1 << i, 1 << j) for i, j in edges]
    for x in range(num_states):
        c = 0
        for mask_i, mask_j in edge_masks:
            bi = (x & mask_i) != 0
            bj = (x & mask_j) != 0
            if bi != bj:
                c += 1
        if c > best:
            best = c
    return best

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    if n > 12:
        return np.array([[np.pi/4, np.pi/8], [np.pi/4, np.pi/8]])
        
    # Precompute cuts
    cuts = _precompute_cuts(n, edges)
    
    # Objective function: negative expected cut
    def objective(params):
        g1, b1, g2, b2 = params
        val = _expected_cut_fast(n, cuts, g1, b1, g2, b2)
        return -val

    # Gradient function using central differences
    # L-BFGS-B can use numerical gradients, but providing them explicitly 
    # or relying on the library's internal finite differences is faster than 
    # Nelder-Mead's simplex updates.
    # We will let L-BFGS-B handle the gradient approximation internally 
    # as it is efficient for 4 variables.
    
    # Multi-start strategy to avoid local minima
    # We generate a grid of starting points
    best_params = None
    best_score = -np.inf
    
    bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]
    
    # Deterministic set of starts
    starts = []
    # Grid of 3x3x3x3 = 81 points is too many for 120s if each takes 0.1s? 
    # 81 * 0.1s = 8s. Acceptable.
    # Let's use a sparser grid or random samples.
    # We'll use a mix of heuristic and random.
    
    # Heuristic starts
    starts.append([np.pi/4, np.pi/8, np.pi/4, np.pi/8])
    starts.append([0.5, 0.2, 0.5, 0.2])
    starts.append([1.0, 0.5, 1.0, 0.5])
    starts.append([0.1, 0.1, 0.1, 0.1])
    starts.append([np.pi/2, np.pi/4, np.pi/2, np.pi/4])
    
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        starts.append(rng.uniform(0, np.pi, size=4))
    
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        best_params = starts[0]
        
    # Format output: (p, 2) array
    return np.array([[best_params[0], best_params[1]], 
                     [best_params[2], best_params[3]]])

# EVOLVE-BLOCK-END
