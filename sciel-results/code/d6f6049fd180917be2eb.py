"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    import sys
    sys.setrecursionlimit(10000)
    
    N = 512
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Greedy baseline
    greedy_set = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in greedy_set):
            greedy_set.append(a)
    greedy_sum = sum(weights[a] for a in greedy_set)
    
    best_sum = greedy_sum
    best_set = list(greedy_set)
    
    # Branch and bound
    # Candidates in order of decreasing weight
    candidates = list(range(511, 0, -1))
    n = len(candidates)
    
    # Precompute coprime check using bitsets
    # bit a corresponds to number a
    coprime_mask = [0] * N
    for a in range(1, N):
        mask = 0
        for b in range(1, N):
            if b != a and gcd(a, b) == 1:
                mask |= (1 << b)
        coprime_mask[a] = mask
    
    # Suffix sum of weights for upper bound
    suffix_w = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_w[i] = suffix_w[i + 1] + weights[candidates[i]]
    
    # B&B search
    def search(idx, selected_mask, current_sum):
        nonlocal best_sum, best_set
        
        if current_sum + suffix_w[idx] <= best_sum:
            return
        
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                s = [a for a in range(1, N) if selected_mask & (1 << a)]
                best_set = s
            return
        
        a = candidates[idx]
        bit = 1 << a
        
        # Try including a
        if (selected_mask & ~coprime_mask[a]) == 0:
            search(idx + 1, selected_mask | bit, current_sum + weights[a])
        
        # Try skipping a
        search(idx + 1, selected_mask, current_sum)
    
    search(0, 0, 0.0)
    
    return best_set
# EVOLVE-BLOCK-END
