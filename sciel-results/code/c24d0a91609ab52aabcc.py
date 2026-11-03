import numpy as np

def construct():
    n = 50
    
    def is_sidon(indices):
        """
        Check if the set of indices is a Sidon set.
        A Sidon set has distinct pairwise sums.
        """
        sums = set()
        k = len(indices)
        for i in range(k):
            for j in range(i, k):
                s = indices[i] + indices[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

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

    # Try to find a large Sidon set
    # Max possible k is around sqrt(n)
    # For n=50, try k=8, 7, 6...
    best_h = np.zeros(n)
    best_score = 0.0
    
    # Fallback: single spike has score 1.0
    best_h[0] = 1.0
    best_score = 1.0
    
    for k in range(8, 0, -1):
        found = False
        # Try many random subsets
        for _ in range(50000):
            indices = np.random.choice(n, k, replace=False)
            if is_sidon(indices):
                h = np.zeros(n)
                h[indices] = 1.0 / k
                score = calc_score(h)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
                found = True
                # We found one, but maybe a better one exists?
                # For a fixed k, all Sidon sets have similar score (k/2).
                # So we can break after finding one for this k.
                # However, different Sidon sets might have different max collision counts?
                # No, by definition max collision is 2 (or 1 if no off-diagonal).
                # So score is k/2.
                # So any Sidon set of size k is equally good.
                break
        
        if found:
            # We found a Sidon set of size k.
            # Since score increases with k, and we are iterating k downwards,
            # the first k we find (largest) is the best.
            # So we can return immediately.
            return list(best_h)
            
    # If no Sidon set found (should not happen for k=1), return best found
    return list(best_h)
