"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Returns a subset of [1, 512] with lcm(a, b) <= 512 for all pairs,
    maximizing cardinality.
    
    Strategy:
    1. Start with divisors of 360 (24 elements) as initial seed.
    2. Build a compatibility graph where nodes are numbers 1-512,
       and edges connect pairs with lcm <= 512.
    3. Use Branch and Bound to find the maximum clique.
    4. Return the largest valid set found within time limit.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute gcd table for speed
    max_n = 512
    gcd_table = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    for i in range(1, max_n + 1):
        for j in range(i, max_n + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Check compatibility: lcm(a,b) <= 512
    def compatible(a, b):
        return a * b // gcd_table[a][b] <= 512
    
    # Build adjacency: adj[i] = set of j where compatible(i,j)
    # Use bitmasks for efficient clique search
    adj = [0] * (max_n + 1)
    for i in range(1, max_n + 1):
        mask = 0
        for j in range(i + 1, max_n + 1):
            if compatible(i, j):
                mask |= (1 << j)
        adj[i] = mask
    
    # Start with divisors of 360 as initial solution
    L = 360
    initial = set()
    for i in range(1, int(L**0.5) + 1):
        if L % i == 0:
            initial.add(i)
            if i != L // i:
                initial.add(L // i)
    
    best_clique = list(initial)
    best_size = len(best_clique)
    
    # Branch and Bound for maximum clique
    # We'll search over candidates 1..512
    # Use a recursive approach with pruning
    
    def branch_bound(candidates, current_clique, depth):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
        
        # Upper bound: current size + number of candidates that can be added
        # Simple bound: len(current_clique) + len(candidates)
        if len(current_clique) + len(candidates) <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree within candidates
        v = None
        max_deg = -1
        for u in candidates:
            deg = bin(adj[u] & candidates).count('1')
            if deg > max_deg:
                max_deg = deg
                v = u
        
        # Branch 1: include v
        new_candidates = [u for u in candidates if u != v and compatible(v, u)]
        new_clique = list(current_clique) + [v]
        branch_bound(new_candidates, new_clique, depth + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude v
        candidates_without_v = [u for u in candidates if u != v]
        branch_bound(candidates_without_v, current_clique, depth + 1)
    
    # Initial candidates: all numbers 1..512
    # But we can prune: only consider numbers that are compatible with at least one in initial
    # Actually, let's just do a simpler greedy + local search approach for robustness
    
    # Greedy approach: start with initial, try to add more
    S = set(initial)
    changed = True
    while changed:
        changed = False
        for x in range(1, max_n + 1):
            if x in S:
                continue
            if all(compatible(x, a) for a in S):
                S.add(x)
                changed = True
                best_size = len(S)
                best_clique = list(S)
                break
        if time.time() > deadline:
            break
    
    # Local search: try swapping elements to improve
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for x in list(S):
            if x == 1:  # Keep 1 always
                continue
            S.remove(x)
            for y in range(1, max_n + 1):
                if y in S or y == x:
                    continue
                if all(compatible(y, a) for a in S):
                    S.add(y)
                    improved = True
                    break
            if not improved:
                S.add(x)
                break
        if not improved:
            break
    
    best_clique = list(S)
    best_size = len(best_clique)
    
    # Final verification
    valid = True
    for i in range(len(best_clique)):
        for j in range(i + 1, len(best_clique)):
            if not compatible(best_clique[i], best_clique[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to initial
        best_clique = list(initial)
    
    return best_clique
# EVOLVE-BLOCK-END
