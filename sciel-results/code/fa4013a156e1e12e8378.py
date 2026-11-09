import time
from math import lcm, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (90s kill)
    
    def is_valid_set(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if lcm(S[i], S[j]) <= 256:
                    return False
        return True

    def get_score(S):
        return sum(1.0/x for x in S)

    # Precompute compatibility matrix
    # compat[i][j] = True if lcm(i+2, j+2) > 256
    # Numbers are 2..256, index 0..254
    n = 255  # numbers 2 to 256
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = i+2, j+2
            compat[i][j] = compat[j][i] = (lcm(a, b) > 256)
    
    # Weight of each number
    weights = [1.0/(i+2) for i in range(n)]
    
    # Sort numbers by weight descending (larger reciprocal first)
    # But we need to maintain compatibility, so let's work with the original ordering
    # and use branch and bound
    
    # Branch and bound for maximum weight clique
    # We want a clique in the compat graph (all pairs compatible)
    
    best_score = 0.0
    best_clique = []
    
    # Upper bound: sum of weights of all remaining candidates
    # Better bound: for each candidate, it can only connect to compatible nodes
    # Use a greedy coloring bound on the complement of compat (incompatibility graph)
    
    # For clique finding, we use the standard algorithm:
    # At each step, we have a candidate set R and current clique C
    # Upper bound = sum of weights in C + sum of weights in R
    # But tighter: we can use the fact that within R, we can only pick a clique
    
    # Let's implement a simple branch and bound with good ordering
    
    def branch_and_bound(candidates, current_clique, current_weight):
        nonlocal best_score, best_clique
        
        if time.time() > deadline:
            return
        
        # Upper bound: current weight + sum of all candidate weights
        upper = current_weight + sum(weights[i] for i in candidates)
        if upper <= best_score + 1e-12:
            return
        
        if not candidates:
            if current_weight > best_score:
                best_score = current_weight
                best_clique = list(current_clique)
            return
        
        # Try adding each candidate to the clique
        # Order candidates by weight descending
        sorted_candidates = sorted(candidates, key=lambda i: -weights[i])
        
        for idx, node in enumerate(sorted_candidates):
            if time.time() > deadline:
                return
            
            # Check if node is compatible with all in current_clique
            compatible = True
            for c in current_clique:
                if not compat[node][c]:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # New candidates: those compatible with node and not yet considered
            new_candidates = [c for c in sorted_candidates[idx+1:] if compat[node][c]]
            
            branch_and_bound(new_candidates, current_clique + [node], current_weight + weights[node])
    
    # Start branch and bound
    all_nodes = list(range(n))
    branch_and_bound(all_nodes, [], 0.0)
    
    if best_clique:
        return [i+2 for i in best_clique]
    
    # Fallback: return seed set
    S = [2] + [x for x in range(3, 257) if x % 2 != 0]
    return S
