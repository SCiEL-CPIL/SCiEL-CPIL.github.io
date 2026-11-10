import numpy as np
import time
import random

N = 10
# Pre-compute projection indices
# x, y in 0..9
xs = np.arange(N)
ys = np.arange(N)
X, Y = np.meshgrid(xs, ys, indexing='xy')
X_flat = X.flatten()
Y_flat = Y.flatten()

# Projections:
# D: x - y  (values -9 to 9, 19 bins)
# M candidates:
# H_0: x     (values 0 to 9, 10 bins)
# H_1: x + y (values 0 to 18, 19 bins)
# H_2: x + 2y (values 0 to 27, 28 bins)

# We need bin indices for each cell (0..99)
# For a given projection function f(x,y), we map value to index.
# Since values are integers, we can just compute the value and use it as index if we shift.

# Let's create arrays of bin indices for each cell
# U: x - y + 9 -> 0..18
U_IDX = (X_flat - Y_flat + 9).astype(int)
# V: x -> 0..9
V_IDX = X_flat.astype(int)
# W: x + y -> 0..18
W_IDX = (X_flat + Y_flat).astype(int)
# Z: x + 2y -> 0..27
Z_IDX = (X_flat + 2 * Y_flat).astype(int)

U_SIZE = 19
V_SIZE = 10
W_SIZE = 19
Z_SIZE = 28

def _entropy_from_counts(counts):
    """
    Calculate Shannon entropy in bits from counts.
    counts: 1D array of non-negative floats/ints.
    """
    total = np.sum(counts)
    if total <= 1e-12:
        return 0.0
    probs = counts / total
    mask = probs > 1e-12
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    return -np.sum(p * np.log2(p))

def _compute_ratio(P):
    """
    Compute ratio D/M for probability vector P (length 100, sum 1).
    P is assumed non-negative.
    """
    # Project P onto the bins
    # We can use bincount with weights for speed
    p_u = np.bincount(U_IDX, weights=P, minlength=U_SIZE)
    p_v = np.bincount(V_IDX, weights=P, minlength=V_SIZE)
    p_w = np.bincount(W_IDX, weights=P, minlength=W_SIZE)
    p_z = np.bincount(Z_IDX, weights=P, minlength=Z_SIZE)
    
    h_u = _entropy_from_counts(p_u)
    h_v = _entropy_from_counts(p_v)
    h_w = _entropy_from_counts(p_w)
    h_z = _entropy_from_counts(p_z)
    
    D = h_u
    M = max(h_v, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _optimize_weights_on_support(support_indices):
    """
    Given a set of support indices (subset of 0..99), find the optimal weights
    on these indices to maximize the ratio.
    We use a simple iterative method: 
    1. Start with uniform weights on support.
    2. Use coordinate ascent or a simplified gradient step.
    Since the dimension of the support is small (e.g., 10-50), we can use
    scipy.optimize.minimize with SLSQP or similar, or just a simple 
    randomized local search.
    
    For speed and robustness, let's use a few steps of "entropy balancing" or 
    just random perturbations with acceptance if improved.
    Actually, for a fixed support of size K, the optimal distribution is unique 
    for a given objective? Not necessarily.
    
    Let's use a simple gradient-free local search on the simplex of the support.
    """
    K = len(support_indices)
    if K == 0:
        return np.zeros(100), 0.0
    
    # Initialize weights on support
    weights = np.ones(K) / K
    
    # Map support indices to global indices
    # We need to evaluate the full 100-vector P to compute the ratio.
    # P is zero everywhere except at support_indices.
    
    best_weights = weights.copy()
    best_ratio = 0.0
    
    # Create a temporary P vector
    P = np.zeros(100)
    P[support_indices] = weights
    
    best_ratio = _compute_ratio(P)
    
    # Local search: try moving mass between pairs of support cells
    # This is O(K^2 * eval_cost). K is small.
    # Let's do a few rounds of random pair swaps.
    num_steps = 200
    rng = np.random.default_rng(42)
    
    for _ in range(num_steps):
        i, j = rng.integers(0, K, 2)
        if i == j:
            continue
        # Try moving a small amount of mass from i to j
        # Or just random resampling of weights?
        # Let's try a "move" operation: take delta from i, add to j.
        # To be efficient, we can just try a few random deltas.
        
        # Actually, a better approach for fixed support: 
        # The ratio is D/M. D is entropy of U-projection, M is max of others.
        # This is a non-convex function.
        
        # Let's use a simple hill climbing: 
        # Generate a new random distribution on the support (Dirichlet) and accept if better.
        # This is slow if we do it 200 times.
        
        # Alternative: Use scipy.optimize on the K-dim simplex.
        # But calling scipy for every support candidate might be slow if we have many candidates.
        # However, we only call this for the top few supports.
        
        # Let's implement a quick local search here.
        # Perturb weights: w_i += eps, w_j -= eps.
        for delta in [0.05, 0.1, 0.2]:
            if weights[i] < delta: continue
            new_w = weights.copy()
            new_w[i] -= delta
            new_w[j] += delta
            # Ensure non-negative
            if np.any(new_w < 0): continue
            
            P[support_indices] = new_w
            r = _compute_ratio(P)
            if r > best_ratio:
                best_ratio = r
                best_weights = new_w.copy()
        
        # Restore best weights for next iteration
        P[support_indices] = best_weights
        
    P_final = np.zeros(100)
    P_final[support_indices] = best_weights
    # Normalize just in case
    total = np.sum(P_final)
    if total > 0:
        P_final /= total
    
    return P_final, best_ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    # Strategy: Search for sparse supports.
    # A support of size K has K degrees of freedom.
    # We want to maximize D/M.
    # D is max ~ log2(19) ~ 4.25 bits.
    # M is min of the three. To make M small, we want the projections to be "concentrated".
    # Concentrated projections imply that the cells in the support share many x, x+y, x+2y values.
    # This suggests the support should be clustered or structured.
    
    # Heuristic: 
    # 1. Try all single cells (K=1). Ratio is 0 (M=0).
    # 2. Try pairs (K=2). 
    # 3. Try small clusters.
    
    # Instead of enumerating all supports (huge), we will use a randomized search 
    # for supports, guided by the potential ratio.
    
    # Potential Ratio Estimate:
    # For a given set of cells S, let U_S be the set of unique U values, etc.
    # Max possible D is log2(|U_S|).
    # Min possible M is log2(min(|V_S|, |W_S|, |Z_S|))? No, entropy depends on weights.
    # But if |V_S| is small, H_V is bounded by log2(|V_S|).
    # So if we pick S such that |U_S| is large and |V_S|, |W_S|, |Z_S| are small, 
    # we have a high potential ratio.
    
    # We will generate random sets of cells and score them by this "potential".
    
    all_indices = np.arange(100)
    rng = np.random.default_rng(123)
    
    # We will test supports of size 5, 10, 15, 20, 30.
    # For each size, we will generate many random supports, score them by potential,
    # keep the top few, and optimize weights on them.
    
    support_sizes = [5, 10, 15, 20, 30, 50]
    
    # Pre-compute projection values for all cells to speed up potential calculation
    # We already have U_IDX, V_IDX, etc.
    
    def _potential_score(support):
        # support: array of cell indices
        u_vals = U_IDX[support]
        v_vals = V_IDX[support]
        w_vals = W_IDX[support]
        z_vals = Z_IDX[support]
        
        n_u = len(np.unique(u_vals))
        n_v = len(np.unique(v_vals))
        n_w = len(np.unique(w_vals))
        n_z = len(np.unique(z_vals))
        
        if n_v == 0 or n_w == 0 or n_z == 0:
            return 0.0
            
        # Potential D is log2(n_u)
        # Potential M is min(log2(n_v), log2(n_w), log2(n_z)) ? 
        # Actually, if we concentrate mass, H can be smaller than log2(count).
        # But log2(count) is the MAX entropy. 
        # We want D to be HIGH (max entropy) and M to be LOW (min entropy).
        # So we want n_u to be large, and min(n_v, n_w, n_z) to be small.
        
        max_h_u = np.log2(n_u) if n_u > 1 else 0.0
        min_h_m = min(np.log2(n_v), np.log2(n_w), np.log2(n_z))
        
        if min_h_m < 1e-9:
            # If one projection is constant, M can be 0? No, if all cells have same x, H_x=0.
            # If M=0, ratio is infinite? No, task says M <= 1e-9 is invalid.
            # So we need M > 0.
            # But if we can make M very small, ratio is huge.
            # Let's return a high score if min_h_m is small but > 0.
            # Actually, if n_v=1, H_v=0. Then M=0 (if others are also 0? No, max).
            # If H_v=0, H_w>0, H_z>0, then M = max(0, H_w, H_z) = H_w or H_z.
            # So M is not necessarily 0.
            # The bound log2(n) is an upper bound. The actual entropy can be lower.
            # This potential score is a heuristic.
            pass
            
        # Heuristic: Reward large spread in U, small spread in V, W, Z.
        score = max_h_u - min_h_m
        return score

    # To make the search more effective, we should bias towards supports that are 
    # "diagonal" in some sense, as diagonals have constant x-y (small U spread) 
    # but wait, we want LARGE U spread.
    # Large U spread means x-y varies a lot.
    # Small V spread means x is constant? No, small V spread means x takes few values.
    # If x is constant, H_x=0. 
    # If we pick a vertical line x=c, U = c-y, which varies from c-9 to c. (Large spread).
    # V = c (constant). H_V = 0.
    # W = c+y (varies). H_W > 0.
    # Z = c+2y (varies). H_Z > 0.
    # M = max(0, H_W, H_Z) = H_Z (usually largest).
    # D = H_U.
    # For a vertical line, U and W and Z are all linear in y with different slopes.
    # The entropies depend on the distribution of y.
    # If we pick uniform y, H_U = H_W = H_Z = log2(10). Ratio = 1.
    # If we pick non-uniform y, can we make H_U > H_Z?
    # H(p) is concave. 
    # Let's just let the optimizer find the weights.
    
    # So, vertical lines are good candidates!
    # Also horizontal lines?
    # Horizontal line y=c: U = x-c (varies), V=x (varies), W=x+c (varies), Z=x+2c (varies).
    # All have same entropy for uniform x. Ratio 1.
    
    # What about a "cross" or other shapes?
    
    # Let's explicitly add structured supports:
    # 1. All vertical lines (10 supports)
    # 2. All horizontal lines (10 supports)
    # 3. Diagonals x-y=k (19 supports) -> U is constant! D=0. Bad.
    # 4. Anti-diagonals x+y=k -> W is constant. M might be small?
    #    If W is constant, H_W=0. M = max(H_V, 0, H_Z).
    #    U = x-y. On x+y=k, x = k-y. U = k-2y. V = k-y. Z = k-y+2y = k+y.
    #    So U, V, Z are all linear in y with slopes -2, -1, 1.
    #    We can optimize y distribution to maximize H_U / max(H_V, H_Z).
    
    structured_supports = []
    
    # Vertical lines
    for x in range(10):
        idx = np.where(X_flat == x)[0]
        structured_supports.append(idx)
        
    # Horizontal lines
    for y in range(10):
        idx = np.where(Y_flat == y)[0]
        structured_supports.append(idx)
        
    # Anti-diagonals (x+y = k)
    for k in range(19):
        idx = np.where((X_flat + Y_flat) == k)[0]
        if len(idx) > 0:
            structured_supports.append(idx)
            
    # Also, let's add some random supports to explore other shapes.
    # We will generate random supports and filter by potential score.
    
    num_random_candidates = 200
    
    # We will evaluate structured supports first.
    candidates = structured_supports.copy()
    
    # Generate random supports
    # We want supports that are not too large, but large enough to have entropy.
    # Sizes 5 to 20.
    
    for _ in range(num_random_candidates):
        size = rng.integers(5, 21)
        support = rng.choice(100, size=size, replace=False)
        candidates.append(support)
        
    # Sort candidates by potential score
    # To save time, we will only optimize the top N candidates.
    
    scored_candidates = []
    for sup in candidates:
        # Ensure unique indices
        sup = np.unique(sup)
        if len(sup) < 2:
            continue
        score = _potential_score(sup)
        scored_candidates.append((score, sup))
        
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 50 for full optimization
    top_candidates = [sup for score, sup in scored_candidates[:50]]
    
    # Optimize weights on top candidates
    for sup in top_candidates:
        if time.time() > deadline - 10:
            break
            
        P_opt, ratio_opt = _optimize_weights_on_support(sup)
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt
            
    # If time remains, do a final random search with larger supports or different structures
    # Or just refine the best_P found.
    
    # Ensure validity
    P_out = best_P.reshape((10, 10))
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out

# EVOLVE-BLOCK-END
