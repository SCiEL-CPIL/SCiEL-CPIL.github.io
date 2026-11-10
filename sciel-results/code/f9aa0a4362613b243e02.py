"""Improved heuristic for ant colony offline bin packing using parameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy: 
    1. Define a family of heuristics based on tightness of fit (inverse waste power) 
       and exponential decay.
    2. Use a fast greedy simulation proxy to evaluate the quality of each candidate 
       heuristic. The proxy simulates the ACO's decision process by greedily picking 
       the item with the highest heuristic value that fits in the current bin.
    3. Select the heuristic that minimizes the number of bins in the proxy simulation.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    cap = float(capacity)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Precompute pairwise metrics
    pair = d[:, None] + d[None, :]
    feasible = pair <= cap
    waste = cap - pair
    
    # Similarity bonus to reduce fragmentation
    size_diff = np.abs(d[:, None] - d[None, :])
    similarity = 1.0 / (1.0 + size_diff / cap)
    
    # Fast Greedy Simulation Proxy
    # Mimics ACO: Start random item, then greedily pick next item with max H[last, j]
    def simulate(d, H, cap, seed=42):
        n = len(d)
        remaining = list(range(n))
        bins_used = 0
        rng = np.random.RandomState(seed)
        
        while remaining:
            # Pick a random item to start the bin
            idx = rng.randint(len(remaining))
            start = remaining[idx]
            remaining.pop(idx)
            
            bin_sum = d[start]
            last = start
            bins_used += 1
            
            while True:
                # Find candidates that fit
                candidates = []
                # Optimization: If remaining is small, linear scan is fine.
                # For N=120, this is fast.
                for j in remaining:
                    if bin_sum + d[j] <= cap:
                        candidates.append(j)
                
                if not candidates:
                    break
                
                # Pick candidate with highest heuristic value from 'last'
                best_j = -1
                best_val = -np.inf
                for j in candidates:
                    val = H[last, j]
                    if val > best_val:
                        best_val = val
                        best_j = j
                
                # Place best_j
                # Remove from remaining (list removal is O(N), acceptable for N=120)
                remaining.remove(best_j)
                bin_sum += d[best_j]
                last = best_j
        
        return bins_used

    best_score = np.inf
    best_matrix = None
    
    # Candidate 1: Inverse Power of Waste
    # H = 1 / (waste + eps)^beta * (1 + w_sim * similarity)
    betas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    w_sim = 0.15  # Similarity weight
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        eps = 1e-6
        safe_waste = np.where(feasible, waste + eps, 1.0)
        tightness = 1.0 / (safe_waste ** beta)
        
        # Combine tightness and similarity
        H = tightness * (1.0 + w_sim * similarity)
        H = np.where(feasible, H, 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        
        score = simulate(d, H, cap)
        
        if score < best_score:
            best_score = score
            best_matrix = H.copy()
            
    # Candidate 2: Exponential Decay
    # H = exp(-waste / scale) * (1 + w_sim * similarity)
    scales = [15.0, 25.0, 37.5, 50.0, 75.0, 100.0]
    
    for scale in scales:
        if time.time() > deadline - 10:
            break
        
        gap_penalty = np.exp(-waste / scale)
        H = gap_penalty * (1.0 + w_sim * similarity)
        H = np.where(feasible, H, 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        
        score = simulate(d, H, cap)
        
        if score < best_score:
            best_score = score
            best_matrix = H.copy()

    # Candidate 3: Product-based (from Inspiration 1)
    # H = (d_i * d_j) * (fill_ratio)^alpha
    products = d[:, None] * d[None, :]
    fill_ratio = pair / cap
    
    alphas = [1.0, 2.0, 3.0]
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        
        safe_fill = np.maximum(fill_ratio, 1e-9)
        tightness = safe_fill ** alpha
        H = products * tightness * (1.0 + w_sim * similarity)
        H = np.where(feasible, H, 1e-9)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        
        score = simulate(d, H, cap)
        
        if score < best_score:
            best_score = score
            best_matrix = H.copy()

    if best_matrix is None:
        # Fallback to a simple valid matrix
        best_matrix = np.where(feasible, 1.0, 1e-9)
        
    return best_matrix
# EVOLVE-BLOCK-END
