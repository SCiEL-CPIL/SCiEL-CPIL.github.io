"""Rewritten program: construct largest pairwise non-coprime subset of [1,255] containing 255."""
import time
import math
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find the maximum subset S ⊆ [1, 255] containing 255 such that gcd(a,b) > 1 
    for all distinct pairs a,b in S.
    
    Since 255 ∈ S, for any x ∈ S, gcd(x, 255) > 1.
    Let C = {x ∈ [1, 255] | gcd(x, 255) > 1}.
    Then S ⊆ C.
    The condition "pairwise gcd > 1" means S is a clique in the graph G_C where 
    vertices are C and edges connect u,v if gcd(u,v) > 1.
    Since 255 is connected to all vertices in C (by definition of C), 
    finding the max S containing 255 is equivalent to finding the max clique 
    in the subgraph induced by C \ {255}, and adding 255 to it.
    """
    
    # 1. Generate candidates C
    C = [x for x in range(1, 256) if gcd(x, 255) > 1]
    
    # 2. We need max clique in G[C \ {255}]
    # Let V = C \ {255}
    V = [x for x in C if x != 255]
    n = len(V)
    
    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(V)}
    
    # Build adjacency list
    # adj[i] is a set of indices j such that gcd(V[i], V[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = V[i]
        for j in range(i + 1, n):
            vj = V[j]
            if gcd(vi, vj) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Maximum Clique Algorithm (Branch and Bound)
    # We use a simple recursive branch and bound with ordering by degree.
    # For n=110, this is fast.
    
    best_clique = []
    best_size = 0
    
    # Order vertices by degree descending to help pruning? 
    # Standard Bron-Kerbosch with pivot is better.
    # Let's implement Bron-Kerbosch with pivot (Tomita style).
    
    # Pre-compute degrees for initial ordering? 
    # The algorithm will handle ordering via the candidate set.
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # If current clique size + candidates size <= best_size, prune
        if len(clique) + len(candidates) <= best_size:
            return
            
        # Pivot selection: choose v in candidates with max degree in candidates
        # This reduces the number of branches
        pivot_v = None
        max_common_neighbors = -1
        for v in candidates:
            common = len(adj[v] & candidates)
            if common > max_common_neighbors:
                max_common_neighbors = common
                pivot_v = v
                
        # Branch on candidates \ N(pivot_v)
        # We iterate through candidates that are NOT neighbors of pivot_v
        # This ensures we cover all maximal cliques
        candidates_to_branch = candidates - adj[pivot_v]
        
        for v in candidates_to_branch:
            new_clique = clique + [v]
            new_candidates = candidates & adj[v]
            expand(new_clique, new_candidates)
            candidates.remove(v)
            # Pruning check after removing v
            if len(clique) + len(candidates) <= best_size:
                break
                
    # Start with empty clique and all V as candidates
    # Note: The recursion depth is at most n=110, which is safe.
    # However, Python recursion limit might be an issue if the clique is large, 
    # but max clique size is likely < 50. 110 depth is fine.
    
    # To be safe with recursion limit, we can increase it or use iterative.
    # 110 is small, default 1000 is fine.
    
    expand([], set(range(n)))
    
    # 4. Construct final set
    # best_clique contains indices into V
    result_indices = best_clique
    result_values = [V[i] for i in result_indices]
    result_values.append(255)
    
    # 5. Verification
    s = set(result_values)
    assert 255 in s
    assert len(s) == len(result_values)
    for a in s:
        assert 1 <= a <= 255
    for i in range(len(result_values)):
        for j in range(i + 1, len(result_values)):
            assert gcd(result_values[i], result_values[j]) > 1
            
    return result_values
# EVOLVE-BLOCK-END
