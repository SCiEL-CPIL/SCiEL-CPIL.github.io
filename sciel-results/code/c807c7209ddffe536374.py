import time
import numpy as np
from itertools import combinations

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Exhaustive search over small support sets (size 1, 2, 3).
    2. For each support set, solve the optimal weight distribution that maximizes D/M.
    3. This avoids high-dimensional local optima by focusing on the structural
       sparsity that often yields high entropy ratios in such grid problems.
    4. Fallback to uniform if no sparse solution is found or time runs out.
    """
    n = 10
    deadline = time.time() + 580.0

    # Precompute bin indices for each entropy component
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten coordinates
    x_flat = xs.flatten()
    y_flat = ys.flatten()
    
    d_val = (x_flat - y_flat)
    m0_val = x_flat
    m1_val = (x_flat + y_flat)
    m2_val = (x_flat + 2 * y_flat)
    
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)
    
    # Precompute the bin membership for each cell (0..99)
    # cell_indices[i] is the index i in the flattened array
    # We need to know which bin each cell falls into for each entropy calculation
    
    def shannon_from_counts(counts):
        """Calculate Shannon entropy from counts (assumes total mass 1 is handled by caller or normalized)."""
        # Normalize counts to probabilities
        total = np.sum(counts)
        if total < 1e-15:
            return 0.0
        probs = counts / total
        non_zero = probs[probs > 1e-15]
        if len(non_zero) == 0:
            return 0.0
        return -np.sum(non_zero * np.log2(non_zero))

    def evaluate_support(weights, indices):
        """
        Evaluate the ratio for a given set of weights on specific indices.
        weights: array of shape (k,)
        indices: array of shape (k,)
        Returns ratio D/M.
        """
        # Build count vectors for each entropy
        # We only need to count the mass in the bins that are hit by these indices
        # But since we are normalizing the total mass to 1, we can just use the weights directly 
        # as the mass distribution.
        
        # D distribution
        d_bins = d_inv[indices]
        # Unique bins in this support
        unique_d_bins = np.unique(d_bins)
        d_counts = np.zeros(num_d)
        for i, bin_idx in enumerate(unique_d_bins):
            mask = d_bins == bin_idx
            d_counts[bin_idx] = np.sum(weights[mask])
        
        H_d = shannon_from_counts(d_counts)
        
        # M components
        m0_bins = m0_inv[indices]
        unique_m0_bins = np.unique(m0_bins)
        m0_counts = np.zeros(num_m0)
        for i, bin_idx in enumerate(unique_m0_bins):
            mask = m0_bins == bin_idx
            m0_counts[bin_idx] = np.sum(weights[mask])
        H_m0 = shannon_from_counts(m0_counts)
        
        m1_bins = m1_inv[indices]
        unique_m1_bins = np.unique(m1_bins)
        m1_counts = np.zeros(num_m1)
        for i, bin_idx in enumerate(unique_m1_bins):
            mask = m1_bins == bin_idx
            m1_counts[bin_idx] = np.sum(weights[mask])
        H_m1 = shannon_from_counts(m1_counts)
        
        m2_bins = m2_inv[indices]
        unique_m2_bins = np.unique(m2_bins)
        m2_counts = np.zeros(num_m2)
        for i, bin_idx in enumerate(unique_m2_bins):
            mask = m2_bins == bin_idx
            m2_counts[bin_idx] = np.sum(weights[mask])
        H_m2 = shannon_from_counts(m2_counts)
        
        M = max(H_m0, H_m1, H_m2)
        if M < 1e-9:
            return 0.0, weights
            
        ratio = H_d / M
        return ratio, weights

    best_ratio = 0.0
    best_weights = None
    best_indices = None
    
    total_cells = n * n
    all_indices = np.arange(total_cells)
    
    # --- Phase 1: Single cell support (k=1) ---
    # If only 1 cell has mass, H_d=0, H_m=0. Ratio is undefined or 0.
    # Actually, if H_d=0 and M=0, it's invalid (M <= 1e-9).
    # If H_d=0 and M>0, ratio is 0.
    # So k=1 is never optimal for maximizing ratio > 0.
    
    # --- Phase 2: Two cell support (k=2) ---
    # For 2 cells, the entropy of a distribution over 2 bins is maximized when probabilities are 0.5, 0.5.
    # However, the cells might fall into the same bin for some entropy, reducing the effective number of bins.
    # We need to find the pair of cells that maximizes H_d / max(H_m0, H_m1, H_m2).
    # Since we can choose the weights, for a fixed pair of cells, we should choose weights to maximize the ratio.
    # Let the two cells be c1, c2.
    # The distribution for D will have masses w1, w2 in bins d1, d2.
    # If d1 == d2, H_d = 0. Ratio = 0.
    # If d1 != d2, H_d = h(w1) where h(p) = -p log2 p - (1-p) log2(1-p).
    # Similarly for M components.
    # We want to maximize h_d(w) / max(h_m0(w), h_m1(w), h_m2(w)).
    # Note that h(p) is symmetric around 0.5 and increases from 0 to 1.
    # The optimal weight for a single entropy term is 0.5.
    # If all M components have distinct bins for the two cells, then all H_m are h(w).
    # Then ratio = h(w)/h(w) = 1.
    # If some M components have the same bins (e.g., m0 bins are same), then H_m0 = 0.
    # We want to minimize the max of the H_m's while keeping H_d high.
    
    # Let's iterate over all pairs.
    # There are 100*99/2 ~ 5000 pairs. This is fast.
    
    for i in range(total_cells):
        if time.time() > deadline - 10:
            break
        for j in range(i + 1, total_cells):
            if time.time() > deadline - 10:
                break
            
            indices = np.array([i, j])
            
            # Check bin memberships
            d_bins = d_inv[indices]
            m0_bins = m0_inv[indices]
            m1_bins = m1_inv[indices]
            m2_bins = m2_inv[indices]
            
            # If any entropy has 0 bins (i.e., both cells in same bin), its entropy is 0.
            # If D has 0 bins, ratio is 0.
            if len(np.unique(d_bins)) < 2:
                continue
            
            # We need to find w in [0, 1] (mass on i, 1-w on j) that maximizes ratio.
            # H_d(w) = h(w) if d_bins[0] != d_bins[1] else 0
            # H_m0(w) = h(w) if m0_bins[0] != m0_bins[1] else 0
            # etc.
            
            def h(p):
                if p <= 1e-15 or p >= 1 - 1e-15:
                    return 0.0
                return -p * np.log2(p) - (1-p) * np.log2(1-p)
            
            # We can sample w or solve analytically.
            # Since h is concave and max of concave functions is not necessarily concave,
            # but here the structure is simple.
            # Let's just check w=0.5 first, as it maximizes each individual entropy.
            # If all distinct bins, ratio = 1.
            # If some bins same, the max might be determined by a subset.
            
            # To be safe and fast, we'll check a few critical points: 0.5, and points where
            # the derivative of the ratio is 0. But for k=2, 0.5 is often optimal or close.
            # Let's do a small grid search for w for this pair if 0.5 doesn't give ratio 1.
            
            w_test = 0.5
            weights = np.array([w_test, 1.0 - w_test])
            ratio, _ = evaluate_support(weights, indices)
            
            # If ratio < 0.9, try to optimize w
            if ratio < 0.9:
                # Fine grid search for w
                ws = np.linspace(0.01, 0.99, 20)
                for w in ws:
                    weights = np.array([w, 1.0 - w])
                    r, _ = evaluate_support(weights, indices)
                    if r > ratio:
                        ratio = r
                        best_weights_for_pair = weights
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_indices = indices
                best_weights = weights

    # --- Phase 3: Three cell support (k=3) ---
    # This is more complex. Number of triples is ~ 160,000.
    # We can afford this if we make the evaluation fast.
    # For k=3, the optimal weights are not trivial.
    # However, we can use a heuristic: uniform weights (1/3, 1/3, 1/3) often work well.
    # Or we can use a local optimizer for the top candidates from k=2.
    
    # Let's generate a set of candidate triples:
    # 1. Triples formed by extending the best k=2 pair with a third cell.
    # 2. Random triples.
    
    candidate_triples = set()
    
    # Extend best pair
    if best_indices is not None:
        i, j = best_indices
        for k in range(total_cells):
            if k != i and k != j:
                candidate_triples.add(tuple(sorted((i, j, k))))
    
    # Add some random triples
    rng = np.random.default_rng(42)
    for _ in range(1000):
        if time.time() > deadline - 30:
            break
        triple = rng.choice(total_cells, size=3, replace=False)
        candidate_triples.add(tuple(sorted(triple)))
    
    # Evaluate top candidates
    # To save time, we'll only evaluate the first N candidates or until time runs out.
    # We'll use a simple local optimization for weights if the initial uniform guess is good.
    
    def optimize_weights_for_support(indices):
        """Use scipy minimize to find optimal weights for a fixed support."""
        from scipy.optimize import minimize
        
        k = len(indices)
        # Parameterize weights as log-softmax to ensure they sum to 1 and are positive
        # Or just use simplex constraints.
        
        def objective(log_weights):
            if time.time() > deadline - 5:
                return 1e9
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.sum(weights)
            ratio, _ = evaluate_support(weights, indices)
            return -ratio
        
        # Initial guess: uniform
        x0 = np.zeros(k)
        
        # Constraints: sum(weights) = 1, weights >= 0
        # Using log-softmax removes the need for sum constraint, but bounds are tricky.
        # Let's stick to simplex using L-BFGS-B with bounds [0, 1] and a constraint?
        # Simpler: parameterize by k-1 weights, last is 1 - sum.
        
        def objective_simplex(w_params):
            if time.time() > deadline - 5:
                return 1e9
            w = np.zeros(k)
            w[:k-1] = w_params
            w[k-1] = 1.0 - np.sum(w_params)
            # Check validity
            if np.any(w < -1e-9):
                return 1e9
            w = np.clip(w, 0, None)
            s = np.sum(w)
            if s < 1e-9:
                return 1e9
            w /= s
            ratio, _ = evaluate_support(w, indices)
            return -ratio
        
        # Bounds for first k-1 weights: [0, 1]
        bounds = [(0, 1)] * (k - 1)
        
        try:
            res = minimize(objective_simplex, np.ones(k-1)/(k), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            if res.fun < -best_ratio + 1e-6: # Only if it improves
                w_params = res.x
                w = np.zeros(k)
                w[:k-1] = w_params
                w[k-1] = 1.0 - np.sum(w_params)
                w = np.clip(w, 0, None)
                s = np.sum(w)
                if s > 1e-9:
                    w /= s
                    ratio, _ = evaluate_support(w, indices)
                    return ratio, w
        except Exception:
            pass
        return 0.0, None

    # Evaluate candidates
    for triple in candidate_triples:
        if time.time() > deadline - 10:
            break
        
        indices = np.array(triple)
        
        # Quick check with uniform weights
        weights_uniform = np.ones(3) / 3.0
        ratio_uniform, _ = evaluate_support(weights_uniform, indices)
        
        # Only optimize if uniform is promising
        if ratio_uniform > best_ratio * 0.9:
            ratio_opt, weights_opt = optimize_weights_for_support(indices)
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_indices = indices
                best_weights = weights_opt

    # Construct final P
    P = np.zeros((n, n))
    if best_indices is not None and best_weights is not None:
        for idx, w in zip(best_indices, best_weights):
            y, x = np.unravel_index(idx, (n, n))
            P[y, x] = w
    else:
        # Fallback to uniform
        P = np.ones((n, n)) / (n * n)
        
    # Validate
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P
