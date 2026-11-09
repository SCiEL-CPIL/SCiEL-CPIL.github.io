import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    """
    deadline = time.time() + 85
    
    # Candidates must share a factor with 255 (3, 5, 17)
    # 255 = 3 * 5 * 17
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)
    idx_map = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # We must include 255.
    # 255 is a "hub" connected to all candidates (by definition of candidates).
    # Therefore, any clique C can be extended to C U {255} if 255 not in C.
    # So the max clique containing 255 is the global max clique.
    
    # Tomita's Algorithm for Maximum Clique
    # Uses greedy coloring for upper bound
    
    # Order vertices by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    
    best_clique = []
    best_size = 0
    
    # Greedy coloring bound
    # We color the candidate set. The number of colors is an upper bound on clique size.
    def color_bound(cand_set):
        if not cand_set:
            return 0
        colors = {}
        max_color = 0
        # Sort by degree in cand_set for better coloring
        sorted_cands = sorted(cand_set, key=lambda x: -len(adj[x] & cand_set))
        for v in sorted_cands:
            used = set()
            for u in adj[v] & cand_set:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            max_color = max(max_color, c + 1)
        return max_color

    def expand(clique, cand_set):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        
        if not cand_set:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return
        
        ub = len(clique) + color_bound(cand_set)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick the vertex with highest degree in cand_set
        sorted_cands = sorted(cand_set, key=lambda x: -len(adj[x] & cand_set))
        
        for v in sorted_cands:
            if time.time() > deadline:
                return
            
            new_clique = clique + [v]
            new_cand = cand_set & adj[v]
            
            expand(new_clique, new_cand)
            
            cand_set.discard(v)
            
            # Prune: if remaining cand_set + current clique size <= best_size, break
            if len(clique) + len(cand_set) <= best_size:
                break

    # Initial candidate set is all vertices
    initial_cand = set(range(n))
    expand([], initial_cand)
    
    # Map indices back to values
    result_values = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in the result.
    if 255 not in result_values:
        # If 255 is not in the found clique, we can add it because it's connected to all.
        result_values.append(255)
        
    return sorted(result_values)
