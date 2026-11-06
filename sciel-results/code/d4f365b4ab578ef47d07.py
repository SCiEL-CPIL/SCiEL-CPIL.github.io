"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 512
    MAX_LCM = 512
    
    # Precompute compatibility: compatible[i] = set of j such that lcm(i,j) <= 512
    compatible = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            l = i // gcd(i, j) * j
            if l <= MAX_LCM:
                compatible[i].add(j)
                compatible[j].add(i)
    
    # Greedy: start with all numbers, remove incompatible ones
    # Better: greedy clique construction
    def greedy_clique():
        remaining = set(range(1, N + 1))
        clique = []
        while remaining:
            # Pick vertex with max degree in remaining subgraph
            best_v = None
            best_deg = -1
            for v in remaining:
                deg = len(compatible[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is None or best_deg == 0 and len(clique) > 0:
                break
            clique.append(best_v)
            # Remove best_v and all its non-neighbors
            neighbors = compatible[best_v]
            remaining = remaining & neighbors
            remaining.discard(best_v)
        return set(clique)
    
    # Local search: try to improve by swaps
    def local_search(initial_set, max_iters=5000):
        current = set(initial_set)
        current_size = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            improved = False
            # Try removing one element and adding another
            for a in list(current):
                # Try adding each non-member that is compatible with all current members
                for b in range(1, N + 1):
                    if b in current:
                        continue
                    # Check if b is compatible with all elements in current except a
                    if all(compatible[b].get(x) for x in current if x != a):
                        # Try swap: remove a, add b
                        new_size = current_size
                        if b not in current:
                            new_size += 1
                        if new_size > current_size:
                            current.remove(a)
                            current.add(b)
                            current_size = new_size
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return current
    
    # Initial greedy solution
    best_set = greedy_clique()
    
    # Try multiple restarts with different orderings
    import random
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        # Shuffle and do greedy
        nums = list(range(1, N + 1))
        random.shuffle(nums)
        
        # Greedy with shuffled order
        clique = set()
        for v in nums:
            if all(v in compatible[x] for x in clique):
                clique.add(v)
        
        if len(clique) > len(best_set):
            best_set = clique
        
        # Local search from this solution
        if time.time() > deadline - 3:
            break
        improved = local_search(clique)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Final local search on best
    if time.time() < deadline - 3:
        final = local_search(best_set, max_iters=10000)
        if len(final) > len(best_set):
            best_set = final
    
    return list(best_set)
# EVOLVE-BLOCK-END
