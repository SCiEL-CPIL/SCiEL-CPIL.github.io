"""Maximize pairwise-noncoprime subset of [1,255] containing 255."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Candidates: x in [1, 255] such that gcd(x, 255) > 1
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
            
    n = len(candidates)
    
    # 255 is at index 254? No, candidates are sorted.
    # 255 is the largest candidate.
    idx_255 = candidates.index(255)
    
    # We need max clique in G[U \ {255}]
    # Let V be indices of candidates excluding 255
    V_indices = [i for i in range(n) if i != idx_255]
    m = len(V_indices)
    
    # Build adjacency matrix for V
    # adj[i][j] is True if gcd(candidates[V_indices[i]], candidates[V_indices[j]]) > 1
    adj = [[False] * m for _ in range(m)]
    for i in range(m):
        vi = candidates[V_indices[i]]
        for j in range(i + 1, m):
            vj = candidates[V_indices[j]]
            if math.gcd(vi, vj) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Max Clique Algorithm
    # We use a Branch and Bound with a greedy coloring heuristic (Tomita's algorithm simplified)
    # Since m=126, it's small enough.
    
    best_clique = []
    
    # Order vertices by degree (descending) to improve pruning
    degrees = [sum(adj[i]) for i in range(m)]
    order = sorted(range(m), key=lambda x: -degrees[x])
    
    # Remap adjacency to ordered indices?
    # It's easier to just use the original indices but visit in order.
    # However, Tomita's algorithm relies on ordering for coloring.
    # Let's create a new adjacency matrix based on the order.
    
    new_adj = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            new_adj[i][j] = adj[order[i]][order[j]]
            
    def max_clique_bb(P, X, clique):
        """
        P: list of vertices in the candidate set (ordered)
        X: list of vertices excluded (not used in standard simple BB, but in Tomita)
        clique: current clique
        """
        nonlocal best_clique
        
        if not P:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return
        
        # Greedy coloring of P
        # Color vertices in P such that adjacent vertices have different colors
        # The max color is an upper bound on the clique size in P
        colors = [0] * len(P)
        color_class = [[] for _ in range(len(P) + 1)]
        
        # Simple greedy coloring
        for i in range(len(P)):
            v = P[i]
            # Find available color
            used_colors = set()
            for j in range(i):
                if new_adj[v][P[j]]:
                    used_colors.add(colors[j])
            
            c = 1
            while c in used_colors:
                c += 1
            colors[i] = c
            color_class[c].append(v)
            
        max_color = max(colors) if colors else 0
        
        # Prune if current size + max_color <= best size
        if len(clique) + max_color <= len(best_clique):
            return
        
        # Iterate through vertices in P in reverse color order (highest color first)
        # This is a heuristic to find large cliques early
        # We process vertices from end of P to start, but we need to respect colors?
        # Standard Tomita:
        # We maintain P and X.
        # Here we simplify: we just iterate P in reverse order.
        
        for i in range(len(P) - 1, -1, -1):
            v = P[i]
            
            # Update upper bound
            # The max color in P[:i+1] is colors[i] (if we colored in order)
            # Actually, the upper bound for the subproblem rooted at v is len(clique) + colors[i]
            if len(clique) + colors[i] <= len(best_clique):
                continue
                
            # Recurse
            new_P = [u for u in P[:i] if new_adj[v][u]]
            new_clique = clique + [v]
            
            max_clique_bb(new_P, [], new_clique)
            
            # Remove v from P (conceptually, handled by loop)
            
    # Initial call
    # P is all vertices in order
    P_initial = list(range(m))
    max_clique_bb(P_initial, [], [])
    
    # Map back to original candidates
    # best_clique contains indices in the 'order' permutation
    clique_vals = [candidates[V_indices[idx]] for idx in best_clique]
    
    # Add 255
    clique_vals.append(255)
    
    # Verify
    # Check if all pairs have gcd > 1
    # For safety, if verification fails, fall back to multiples of 3
    valid = True
    for i in range(len(clique_vals)):
        for j in range(i+1, len(clique_vals)):
            if math.gcd(clique_vals[i], clique_vals[j]) == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Multiples of 3
        clique_vals = [x for x in range(1, 256) if x % 3 == 0]
    
    return sorted(clique_vals)

# EVOLVE-BLOCK-END
