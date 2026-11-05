import time
from math import gcd

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses branch-and-bound with suffix-sum pruning for exact rational arithmetic.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of all denominators for exact integer arithmetic
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Scale each reciprocal: 1/d becomes L/d (integer)
    scaled_recips = [L // d for d in denominators]
    
    # Sort indices by descending scaled reciprocal value (largest first)
    # This helps the branch-and-bound find good solutions early
    order = sorted(range(n), key=lambda i: -scaled_recips[i])
    
    # Precompute suffix sums of scaled_recips in the sorted order
    # suffix_sum[i] = sum of scaled_recips[order[j]] for j >= i
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + scaled_recips[order[i]]
    
    # Best solution tracking
    best_deficit = L  # Initial: deficit = L (i.e., sum = 0, deficit = 1)
    best_indices = None
    
    # Initial greedy solution to get a good starting bound
    # Greedily add denominators in descending order if sum stays < L
    greedy_sum = 0
    greedy_indices = []
    for idx in order:
        if greedy_sum + scaled_recips[idx] < L:
            greedy_sum += scaled_recips[idx]
            greedy_indices.append(idx)
    
    if greedy_indices:
        best_deficit = L - greedy_sum
        best_indices = greedy_indices[:]
    
    # Branch and bound DFS
    # State: current position in sorted order, current sum, current selected indices
    def dfs(pos, cur_sum, selected):
        nonlocal best_deficit, best_indices
        
        if time.time() > deadline:
            return
        
        # Pruning: even if we take all remaining, can we beat current best?
        # Max possible sum = cur_sum + suffix_sum[pos]
        # Deficit if we take all = L - (cur_sum + suffix_sum[pos])
        # If this deficit >= best_deficit, no need to explore
        if cur_sum + suffix_sum[pos] <= L - best_deficit:
            return
        
        # If we're at the end, check if current selection is valid and better
        if pos == n:
            if selected and cur_sum < L:
                deficit = L - cur_sum
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = selected[:]
            return
        
        # Try not taking order[pos]
        dfs(pos + 1, cur_sum, selected)
        
        # Try taking order[pos]
        new_sum = cur_sum + scaled_recips[order[pos]]
        if new_sum < L:  # Must stay strictly below L
            selected.append(order[pos])
            dfs(pos + 1, new_sum, selected)
            selected.pop()
    
    dfs(0, 0, [])
    
    if best_indices is None or not best_indices:
        # Fallback: return a simple valid solution
        return [0]
    
    return best_indices

if __name__ == "__main__":
    res = construct()
    pass
