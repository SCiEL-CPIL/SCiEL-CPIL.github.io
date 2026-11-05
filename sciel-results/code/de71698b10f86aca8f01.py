import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 130

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # Target: all-ones tensor of shape (n, m, p)
    # Greedy forward selection over GF(2)
    # A rank-1 term is (U_mask, V_mask, W_mask) where each is a bitmask
    # It covers position (i,j,k) iff U_mask has bit i, V_mask has bit j, W_mask has bit k
    
    total = n * m * p  # 72
    
    # Precompute all possible masks
    all_u_masks = [1 << i for i in range(n)]  # single bits for efficiency
    all_v_masks = [1 << j for j in range(m)]
    all_w_masks = [1 << k for k in range(p)]
    
    # For greedy, we want to find the best (u_mask, v_mask, w_mask) that covers
    # the most uncovered positions. But the search space is huge.
    # Instead, use a smarter approach: try all combinations of small masks.
    
    # Actually, let's think about this more carefully.
    # The best rank-1 tensor for covering uncovered positions is one where
    # u_mask, v_mask, w_mask are chosen to maximize coverage of the residual.
    
    # Since we're over GF(2), adding a term toggles coverage.
    # We want to minimize the number of terms such that XOR of all terms = all-ones.
    
    # Greedy: at each step, find the term that covers the most currently-uncovered positions.
    
    uncovered = (1 << total) - 1  # bitmask of all positions
    terms = []
    
    # Position index: pos = (i * m + j) * p + k
    def pos_idx(i, j, k):
        return (i * m + j) * p + k
    
    # For efficiency, precompute for each mask combination the set of covered positions
    # But the space is 2^n * 2^m * 2^p = 8 * 16 * 64 = 8192, which is manageable
    
    # Precompute coverage for all (u_mask, v_mask, w_mask)
    # coverage[u_mask][v_mask][w_mask] = bitmask of covered positions
    # This is 8*16*64 = 8192 entries, each a 72-bit integer
    
    coverage = {}
    for u_mask in range(1 << n):
        for v_mask in range(1 << m):
            for w_mask in range(1 << p):
                cov = 0
                for i in range(n):
                    if u_mask & (1 << i):
                        for j in range(m):
                            if v_mask & (1 << j):
                                for k in range(p):
                                    if w_mask & (1 << k):
                                        cov |= (1 << pos_idx(i, j, k))
                coverage[(u_mask, v_mask, w_mask)] = cov
    
    # Greedy forward selection
    current_uncovered = (1 << total) - 1
    selected_terms = []
    
    while current_uncovered != 0 and check_deadline():
        best_cov = 0
        best_term = None
        for key, cov in coverage.items():
            # Count how many currently uncovered positions this term covers
            new_cov = cov & current_uncovered
            count = bin(new_cov).count('1')
            if count > best_cov:
                best_cov = count
                best_term = key
                if best_cov == bin(current_uncovered).count('1'):
                    break  # Perfect coverage
        
        if best_term is None:
            break
        
        u_mask, v_mask, w_mask = best_term
        selected_terms.append((u_mask, v_mask, w_mask))
        current_uncovered ^= coverage[best_term]  # XOR to toggle
    
    # Convert to U, V, W matrices
    rank = len(selected_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u_mask, v_mask, w_mask) in enumerate(selected_terms):
        for i in range(n):
            if u_mask & (1 << i):
                for j in range(m):
                    U[i * m + j][t] = 1
        for j in range(m):
            if v_mask & (1 << j):
                for k in range(p):
                    V[j * p + k][t] = 1
        for k in range(p):
            if w_mask & (1 << k):
                for i in range(n):
                    W[k * n + i][t] = 1
    
    return {"U": U, "V": V, "W": W}
