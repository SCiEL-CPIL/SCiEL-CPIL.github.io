import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but we want to be safe)
    
    N = 512
    # Candidates: 1..511. 
    # Weight for candidate a is 1/(512 - a).
    # We want to maximize sum of weights subject to pairwise coprime constraint.
    
    # 1. Precompute prime factors for each number 1..511
    # Primes up to 511
    primes = []
    for i in range(2, 512):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = 0
                break
        if is_p:
            primes.append(i)
    
    # Map each prime to an index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Bitmask of prime factors for each number
    # We only care about primes < 512.
    factor_masks = [0] * 512
    for a in range(1, 512):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        factor_masks[a] = mask
    
    # 2. Sort candidates by weight (1/(512-a)) in descending order.
    # Since 1/(512-a) is increasing with a, we sort by a descending.
    candidates = list(range(511, 0, -1))
    
    # Precompute weights
    weights = [1.0 / (N - a) for a in candidates]
    
    # 3. Branch and Bound
    # State: (index, current_mask, current_score, current_selection)
    # To speed up, we can prune if current_score + sum(remaining_weights) <= best_score.
    # A tighter bound: Fractional Knapsack on remaining items that are compatible with current_mask.
    # Since the items are already sorted by weight, and the "capacity" is not a sum but a bitmask constraint,
    # the standard fractional knapsack bound doesn't directly apply in the same way as weight/capacity.
    # However, a simple bound is: current_score + sum(weights[i:] for i in range(idx, len(candidates)) if compatible).
    # This is O(N) per node. With N=511, this might be slow if the tree is deep.
    
    # Alternative: Use a recursive search with strong pruning.
    # Since we want pairwise coprime, the set size is limited by the number of distinct prime factors available.
    # Actually, the maximum size of a pairwise coprime set of integers > 1 is bounded by the number of primes.
    # But we can include 1.
    
    # Let's use a simple DFS with pruning.
    best_score = 0.0
    best_selection = []
    
    # Precompute suffix sums of weights for a quick upper bound (ignoring compatibility)
    # This is a weak bound but fast.
    suffix_sum = [0.0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    def dfs(idx, current_mask, current_score, current_selection):
        nonlocal best_score, best_selection
        
        if time.time() > deadline:
            return
        
        # Prune: if even taking all remaining items (ignoring conflicts) can't beat best, stop
        if current_score + suffix_sum[idx] <= best_score:
            return
            
        if idx == len(candidates):
            if current_score > best_score:
                best_score = current_score
                best_selection = list(current_selection)
            return
            
        a = candidates[idx]
        w = weights[idx]
        mask = factor_masks[a]
        
        # Option 1: Include a
        if (current_mask & mask) == 0:
            current_selection.append(a)
            dfs(idx + 1, current_mask | mask, current_score + w, current_selection)
            current_selection.pop()
            
        # Option 2: Exclude a
        dfs(idx + 1, current_mask, current_score, current_selection)
        
    # Start DFS
    dfs(0, 0, 0.0, [])
    
    # The result is a list of integers
    return best_selection

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Selected: {res}")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    # Verify pairwise coprime
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
    print("Done")
