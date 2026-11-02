import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def compute_ratio(h):
        """
        Computes the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
        Note: The problem asks for ratio of squared L2 norm to L1 norm 
        times supremum norm of autoconvolution.
        Formula: (sum(h^2)) / (sum(h) * max(conv(h,h)))
        """
        h = np.asarray(h, dtype=float)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        
        # Efficient convolution calculation for max
        # h*h is full convolution of length 2N-1
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        
        if max_conv < 1e-15:
            return 0.0
            
        return l2_sq / (l1 * max_conv)

    def optimize_heights_for_support(support_indices):
        """
        Given a fixed set of indices (support), optimize the heights at these indices.
        We fix the maximum height to 1 by normalizing later, but during optimization
        we can scale freely. To make it well-posed, we can fix one variable or normalize.
        Here, we parameterize the active heights directly.
        """
        n_active = len(support_indices)
        if n_active == 0:
            return None, 0.0
        
        def objective(x):
            h = np.zeros(N)
            h[support_indices] = x
            # Normalize to ensure max is 1? No, the ratio is scale invariant.
            # Let's check: 
            # R = sum(h^2) / (sum(h) * max(h*h))
            # If h -> c*h:
            # sum(c^2 h^2) / (c sum(h) * max(c^2 (h*h))) = c^2 sum(h^2) / (c^3 sum(h) max(h*h)) = (1/c) * R
            # Wait, the ratio is NOT scale invariant.
            # Let's re-read the task carefully.
            # "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
            # Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
            # If we scale h by c:
            # ||c h||_2^2 = c^2 ||h||_2^2
            # ||c h||_1 = c ||h||_1
            # ||c h * c h||_inf = ||c^2 (h*h)||_inf = c^2 ||h*h||_inf
            # Ratio(c h) = c^2 ||h||_2^2 / (c ||h||_1 * c^2 ||h*h||_inf) = (1/c) * Ratio(h)
            
            # Since we want to MAXIMIZE the ratio, and we can scale h arbitrarily (nonnegative heights),
            # we should make c as small as possible?
            # But the heights must be "nonnegative". There is no upper bound specified in the prompt 
            # "Choose 50 nonnegative heights". 
            # However, usually in such problems, if the scale is free, the optimum is at the boundary.
            # If c -> 0, Ratio -> infinity.
            # This implies there must be a normalization constraint I am missing or the "heights" imply a fixed L1 or L_inf?
            # Let's look at the seed program.
            # Seed program normalizes: h = h / np.max(h). 
            # And computes: ratio = l2_sq / (l1 * max_conv) with l_inf=1.
            # This suggests the problem implicitly assumes ||h||_inf = 1 or similar normalization.
            # The prompt says "score is the lower bound on C2". 
            # The seed program explicitly normalizes max(h) to 1.
            # I will follow the seed program's implicit normalization: ||h||_inf = 1.
            
            # So, in the objective, we should normalize h so max(h)=1.
            max_h = np.max(h)
            if max_h < 1e-15:
                return 1e10
            h_norm = h / max_h
            l2_sq = np.sum(h_norm**2)
            l1 = np.sum(h_norm)
            h_conv = np.convolve(h_norm, h_norm, mode='full')
            max_conv = np.max(h_conv)
            if max_conv < 1e-15:
                return 1e10
            ratio = l2_sq / (l1 * max_conv)
            return -np.log(ratio) # Minimize negative log ratio

        # Initial guess: all ones on support
        x0 = np.ones(n_active)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           bounds=[(1e-6, 1.0)] * n_active, # Heights between 0 and 1
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            h_opt = np.zeros(N)
            h_opt[support_indices] = res.x
            # Normalize to max=1
            max_h = np.max(h_opt)
            if max_h > 0:
                h_opt /= max_h
            
            ratio = compute_ratio(h_opt)
            return h_opt, ratio
        except Exception:
            return None, 0.0

    best_h = None
    best_ratio = 0.0
    
    # Generate candidate support sets
    candidate_supports = []
    
    # 1. Sidon sets (small differences)
    # Known good Sidon sets for N=50
    sidon_sets = [
        [0, 1, 4, 10, 18, 23, 30],
        [0, 1, 4, 10, 18, 23, 30, 39],
        [0, 2, 5, 11, 20, 26, 35],
        [0, 1, 3, 7, 12, 20, 29, 41],
        [0, 1, 4, 10, 18, 23, 30, 39, 49],
        [0, 1, 2, 5, 11, 21, 34, 46],
        [0, 1, 4, 9, 16, 25, 36, 49], # Squares
        [0, 1, 2, 4, 8, 16, 32], # Powers of 2
        [0, 1, 3, 7, 15, 31], # 2^k - 1
        [0, 1, 5, 11, 21, 35], # Linear growth
        [0, 2, 6, 12, 20, 30, 42], # Quadratic growth
    ]
    
    # 2. Random sparse sets
    np.random.seed(42)
    for size in range(3, 15):
        for _ in range(5):
            indices = np.sort(np.random.choice(N, size, replace=False))
            candidate_supports.append(list(indices))
            
    # 3. Structured patterns
    # Geometric
    for r in [1.5, 2.0, 2.5, 3.0]:
        idx = 0
        supp = []
        while idx < N:
            supp.append(idx)
            idx = int(idx * r) + 1
        if len(supp) > 1:
            candidate_supports.append(supp)
            
    # Arithmetic
    for step in range(2, 10):
        supp = list(range(0, N, step))
        candidate_supports.append(supp)
        
    # 4. Local refinements of best found so far
    # We will iterate through candidates, and if we find a good one, we will try adding/removing neighbors.
    
    evaluated_supports = set()
    
    # Sort candidates by size, prefer smaller (sparser) as they often have better properties for this ratio
    # Actually, let's just iterate.
    
    for supp in candidate_supports:
        if time.time() > deadline:
            break
            
        # Deduplicate
        key = tuple(sorted(supp))
        if key in evaluated_supports:
            continue
        evaluated_supports.add(key)
        
        h_opt, ratio = optimize_heights_for_support(supp)
        
        if h_opt is not None and ratio > best_ratio:
            best_ratio = ratio
            best_h = h_opt.copy()
            
            # If we found a good one, try local structural modifications
            # This is a heuristic: if a sparse set works, maybe adding a close neighbor helps?
            # Or removing a redundant one?
            # For now, just record. We will do a second pass with more complex structures if time permits.
            
    # Second pass: More complex optimization or different families if time remains
    if time.time() < deadline - 10:
        # Try to refine the best support found by adding/removing single elements
        if best_h is not None:
            current_support = np.where(best_h > 1e-6)[0]
            current_supp_list = list(current_support)
            
            # Try adding each possible new index
            for i in range(N):
                if i in current_supp_list:
                    continue
                new_supp = current_supp_list + [i]
                key = tuple(sorted(new_supp))
                if key in evaluated_supports:
                    continue
                evaluated_supports.add(key)
                
                h_opt, ratio = optimize_heights_for_support(new_supp)
                if h_opt is not None and ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_opt.copy()
                    current_supp_list = new_supp # Update current for next iterations? 
                    # Careful: this might change the base for subsequent additions.
                    # Let's stick to the original best to avoid drift, or update if significantly better.
                    # For safety, we'll just update best_h and continue checking against the *original* best support + i
                    # Actually, updating current_supp_list allows for greedy growth.
                    
            # Try removing each element from the best support
            current_support = np.where(best_h > 1e-6)[0]
            current_supp_list = list(current_support)
            
            for i in current_supp_list:
                new_supp = [j for j in current_supp_list if j != i]
                if not new_supp:
                    continue
                key = tuple(sorted(new_supp))
                if key in evaluated_supports:
                    continue
                evaluated_supports.add(key)
                
                h_opt, ratio = optimize_heights_for_support(new_supp)
                if h_opt is not None and ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_opt.copy()

    # Fallback if nothing found
    if best_h is None:
        # Use the Sidon set from seed
        S = [0, 1, 4, 10, 18, 23, 30]
        best_h = np.zeros(N)
        for idx in S:
            best_h[idx] = 1.0
        best_h /= np.max(best_h)
        best_ratio = compute_ratio(best_h)

    return list(best_h)
