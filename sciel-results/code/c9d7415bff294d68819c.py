"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import lcm
from functools import lru_cache
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Elements are 1..24
    N = 24
    elements = list(range(1, N + 1))
    
    # Precompute LCM of all elements 1..24
    # LCM(1..24) = 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23
    # Let's compute it
    global_LCM = 1
    for i in range(1, N + 1):
        global_LCM = lcm(global_LCM, i)
    
    # For element x, 1/x in integer terms (scaled by global_LCM) is global_LCM // x
    # We need: for selected set S, for each x in S,
    #   global_LCM // x != sum of (global_LCM // y) for y in T, where T is a subset of S \ {x}
    
    # This is equivalent to: 1/x != sum_{y in T} 1/y for T subset of S\{x}
    
    # Let's define weight[x] = global_LCM // x
    weight = [0] * (N + 1)
    for x in range(1, N + 1):
        weight[x] = global_LCM // x
    
    # For a candidate set S, check validity:
    # For each x in S, check if weight[x] is in the subset-sums of {weight[y] for y in S\{x}}
    
    def is_valid(S):
        """Check if set S (list of elements) is valid (reciprocal-subsum-free)."""
        if not S:
            return True
        S_set = set(S)
        for x in S:
            target = weight[x]
            others = [weight[y] for y in S if y != x]
            # Check if target is a subset sum of others
            if _subset_sum_exists(others, target):
                return False
        return True
    
    def _subset_sum_exists(weights, target):
        """Check if some subset of weights sums to exactly target.
        Uses bitset DP for efficiency."""
        if target == 0:
            return True
        # Bitset DP: bit i is set if sum i is achievable
        # But target can be very large (global_LCM is huge)
        # Let's use a different approach: meet-in-the-middle or just iterate subsets for small sets
        
        n = len(weights)
        if n == 0:
            return target == 0
        
        # For small n (<= 20), we can use meet-in-the-middle
        # For larger n, we need a smarter approach
        
        # Actually, let's use a set-based DP but with early termination
        # The weights are large, so the number of distinct sums might be manageable
        # if the set is small enough
        
        # Use a set of achievable sums, but cap the size
        achievable = {0}
        for w in weights:
            if w > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + w
                if ns == target:
                    return True
                if ns < target:
                    new_sums.add(ns)
            achievable.update(new_sums)
            if len(achievable) > 100000:  # cap to avoid blowup
                # Fallback: use a different method
                break
        
        return target in achievable
    
    # Greedy construction with multiple orderings
    best_set = None
    best_size = 0
    
    # Try different insertion orders
    import random
    random.seed(42)
    
    orders = [
        list(range(1, N + 1)),          # ascending
        list(range(N, 0, -1)),          # descending
        list(range(1, N + 1, 2)),       # odd first
        list(range(2, N + 1, 2)),       # even first
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]
    
    # Add some random permutations
    for _ in range(20):
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        orders.append(perm)
    
    for order in orders:
        if time.time() > deadline:
            break
        
        current = []
        for x in order:
            if time.time() > deadline:
                break
            # Try adding x
            trial = current + [x]
            if is_valid(trial):
                current = trial
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Local improvement: try to add more elements or swap
    if best_set is None:
        best_set = [N]  # fallback
        best_size = 1
    
    # Try to improve by adding elements not in best_set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            trial = best_set + [x]
            if is_valid(trial):
                best_set = trial
                best_size = len(best_set)
                improved = True
                break
    
    # Try swapping: remove one, try to add two
    if time.time() < deadline:
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            remaining = best_set[:i] + best_set[i+1:]
            # Try to add two elements
            for a in range(1, N + 1):
                if a == removed or a in remaining:
                    continue
                trial1 = remaining + [a]
                if not is_valid(trial1):
                    continue
                for b in range(1, N + 1):
                    if b == removed or b == a or b in remaining:
                        continue
                    trial2 = remaining + [a, b]
                    if is_valid(trial2):
                        best_set = trial2
                        best_size = len(best_set)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_set if best_set else [N]

# EVOLVE-BLOCK-END
