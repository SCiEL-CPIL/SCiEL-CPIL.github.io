import sys
from math import lcm
from itertools import combinations

def construct():
    N = 128
    
    # 1. Precompute bad triples
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if lcm(a, b) == lcm(a, c) == lcm(b, c):
                    bad_triples.add((a, b, c))
    
    # 2. Precompute blockers for each vertex
    # blockers[x] is a list of pairs (a, b) such that {a, b, x} is a bad triple
    blockers = [[] for _ in range(N + 1)]
    for a, b, c in bad_triples:
        blockers[a].append((b, c))
        blockers[b].append((a, c))
        blockers[c].append((a, b))
        
    # 3. Calculate degree for ordering
    degree = [0] * (N + 1)
    for a, b, c in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
        
    # Order vertices by degree (descending) to prune earlier
    order = list(range(1, N + 1))
    order.sort(key=lambda x: degree[x], reverse=True)
    
    # 4. Get an initial good solution using greedy
    # We use the seed's greedy logic but optimized with precomputed checks if possible?
    # Actually, just run the simple greedy to set a high bar for B&B
    seed_selected = []
    for x in range(1, N + 1):
        valid = True
        for a, b in combinations(seed_selected, 2):
            if lcm(a, b) == lcm(a, x) == lcm(b, x):
                valid = False
                break
        if valid:
            seed_selected.append(x)
            
    best_size = len(seed_selected)
    best_set = seed_selected
    
    sys.setrecursionlimit(10000)
    
    def dfs(pos, S_bits, size):
        nonlocal best_size, best_set
        
        # Pruning: Even if we take all remaining, we can't beat best
        if size + (N - pos) <= best_size:
            return
            
        if pos == N:
            if size > best_size:
                best_size = size
                s = []
                for i in range(1, N + 1):
                    if (S_bits >> (i - 1)) & 1:
                        s.append(i)
                best_set = s
            return
            
        x = order[pos]
        
        # Option 1: Include x
        can_add = True
        for a, b in blockers[x]:
            if ((S_bits >> (a - 1)) & 1) and ((S_bits >> (b - 1)) & 1):
                can_add = False
                break
                
        if can_add:
            dfs(pos + 1, S_bits | (1 << (x - 1)), size + 1)
            
        # Option 2: Exclude x
        dfs(pos + 1, S_bits, size)
        
    dfs(0, 0, 0)
    
    return best_set

# EVOLVE-BLOCK-END
