import numpy as np

def construct():
    n = 50
    best_h = np.zeros(n)
    best_score = 0.0
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))

    # We will search for sparse supports and optimize weights
    # The "Atom" approach: atoms are single indices (width 1) or small blocks.
    # Given the discrete nature, width 1 is most efficient for L2/L1 ratio.
    
    for trial in range(2000):
        # Randomly select a support size
        # Sidon sets can have size up to ~sqrt(n). For n=50, max Sidon is around 7-8.
        # Let's try sizes from 3 to 10.
        k = np.random.randint(3, 11)
        
        # Randomly select k indices
        indices = np.random.choice(n, k, replace=False)
        
        # Initialize weights
        h = np.zeros(n)
        # Start with uniform weights
        h[indices] = 1.0 / k
        
        # Optimize weights
        # We want to maximize f(h) = ||h||_2^2 / (||h||_1 ||c||_inf)
        # Fix ||h||_1 = 1.
        
        # Simple coordinate descent / local search
        for _iter in range(50):
            # Try to adjust weights
            # Gradient of ||h||_2^2 is 2h.
            # Gradient of ||c||_inf is complex.
            # Use a heuristic: increase weight on indices that contribute to the peak?
            # Or simply try random perturbations.
            
            improved = False
            for i in indices:
                # Try shifting mass from i to j
                for j in indices:
                    if i == j:
                        continue
                    # Move small amount delta from i to j
                    # h_i' = h_i - delta, h_j' = h_j + delta
                    # Keep sum 1.
                    delta = 0.05 * h[i]
                    if delta >= h[i]:
                        delta = h[i] * 0.5
                    if delta < 1e-9:
                        continue
                    
                    h_new = h.copy()
                    h_new[i] -= delta
                    h_new[j] += delta
                    
                    # Ensure non-negative
                    if h_new[i] < 0:
                        continue
                        
                    score_new = calc_score(h_new)
                    if score_new > calc_score(h):
                        h = h_new
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try a larger random jump
                idx_to_change = np.random.choice(indices)
                h[idx_to_change] *= 1.5
                # Renormalize
                h /= np.sum(h)
                if calc_score(h) > calc_score(h): # Placeholder, need prev score
                    pass

        score = calc_score(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()

    if best_score == 0.0:
        best_h[0] = 1.0

    return list(best_h)
