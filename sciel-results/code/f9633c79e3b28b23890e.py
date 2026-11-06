"""Optimized construction for large pairwise noncoprime sets containing 255."""
import math
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Find largest subset of [1,255] containing 255 with gcd(a,b)>1 for all distinct pairs.
    
    # Generate all numbers in [1,255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency using bitsets
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy + local search for maximum clique containing 255
    # Start with all multiples of 3 (known valid clique of size 85)
    
    # Build initial clique: all multiples of 3
    initial_clique = set()
    for i in range(n):
        if candidates[i] % 3 == 0:
            initial_clique.add(i)
    
    best_clique = initial_clique.copy()
    best_size = len(initial_clique)
    
    # Try to improve with local search: try adding vertices, removing if needed
    # Also try different starting points
    
    def time_check():
        return time.time() < deadline
    
    def is_clique(clique_set):
        for i in clique_set:
            for j in clique_set:
                if i == j:
                    continue
                if not (adj[i] >> j) & 1:
                    return False
        return True
    
    def try_add_vertex(clique_set, v):
        """Try to add vertex v to clique. Return new clique if successful, else None."""
        for u in clique_set:
            if not (adj[u] >> v) & 1:
                return None
        return clique_set | {v}
    
    # Local search: try to add vertices one at a time
    improved = True
    while improved and time_check():
        improved = False
        for v in range(n):
            if v in best_clique:
                continue
            # Try adding v
            if try_add_vertex(best_clique, v) is not None:
                best_clique = best_clique | {v}
                best_size = len(best_clique)
                improved = True
    
    # Try different strategies: start from other bases
    # Strategy: start from all multiples of 5
    for base_prime in [5, 17]:
        if not time_check():
            break
        base_clique = set()
        for i in range(n):
            if candidates[i] % base_prime == 0:
                base_clique.add(i)
        
        # Try to expand
        cur_clique = base_clique.copy()
        improved = True
        while improved and time_check():
            improved = False
            for v in range(n):
                if v in cur_clique:
                    continue
                if try_add_vertex(cur_clique, v) is not None:
                    cur_clique = cur_clique | {v}
                    improved = True
        
        if len(cur_clique) > best_size:
            best_clique = cur_clique
            best_size = len(cur_clique)
    
    # Convert to result
    result = [candidates[i] for i in best_clique]
    result.sort()
    
    # Verify
    assert 255 in result
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert math.gcd(result[i], result[j]) > 1
    
    return result

# EVOLVE-BLOCK-END
