import math
import time

def construct():
    """
    Finds the maximum cardinality subset of [1, 255] containing 255 
    such that gcd(a, b) > 1 for all distinct pairs a, b.
    """
    N = 255
    target = 255
    
    # Candidate pool: numbers in [1, 255] sharing a factor with 255 (3, 5, 17)
    # Since 255 is in the set, all other elements must share a factor with 255.
    pool = [x for x in range(1, N + 1) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    m = len(pool)
    
    # Map value to index
    idx = {v: i for i, v in enumerate(pool)}
    
    # Build adjacency matrix
    # adj[i] is a list of indices j such that gcd(pool[i], pool[j]) > 1
    adj = [[] for _ in range(m)]
    for i in range(m):
        vi = pool[i]
        for j in range(i + 1, m):
            vj = pool[j]
            if math.gcd(vi, vj) > 1:
                adj[i].append(j)
                adj[j].append(i)
                
    # Sort neighbors by degree (descending) for better branching in MCQ
    # Degree is len(adj[i])
    # We will use a recursive branch and bound approach
    
    # MCQ Algorithm
    # R: current clique
    # P: candidate set (vertices adjacent to all in R)
    
    best_clique = []
    best_size = 0
    
    # Initial candidate set: all vertices
    # Since 255 is connected to all, and we proved max clique contains 255,
    # we can start with R = [255] and P = neighbors of 255.
    # However, standard MCQ starts with P = all vertices.
    # Let's start with P = all vertices.
    
    # To optimize, we can order vertices by degree.
    # But MCQ usually works well with dynamic ordering.
    
    # We will use a stack-based or recursive approach.
    # Given m=127, recursion depth is fine.
    
    # Precompute neighbor lists as sets for faster intersection?
    # Or just lists.
    
    def mcq(R, P):
        nonlocal best_clique, best_size
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        if len(R) + len(P) <= best_size:
            return
        
        # Greedy coloring to get upper bounds
        # Sort P by degree? Or use the greedy coloring heuristic.
        # Standard MCQ:
        # 1. Color P greedily.
        # 2. Order vertices in P by color.
        # 3. Branch on vertices in reverse color order.
        
        # Greedy Coloring
        # We want to color the subgraph induced by P.
        # Colors are 1, 2, ...
        # We process vertices in some order. A good order is by degree in P.
        
        # Get degrees in P
        P_set = set(P)
        deg_in_P = {v: len([u for u in adj[v] if u in P_set]) for v in P}
        
        # Order P by degree descending
        ordered_P = sorted(P, key=lambda v: deg_in_P[v], reverse=True)
        
        # Color the graph
        colors = {}
        color = 0
        # We will assign colors such that no two adjacent vertices have the same color.
        # This is an independent set problem on the complement, or just greedy coloring.
        # Greedy coloring:
        # For each vertex in ordered_P, assign the smallest color not used by its neighbors in P.
        # Actually, standard MCQ uses a specific greedy coloring that produces an upper bound.
        # The number of colors is an upper bound on the clique size in P.
        
        # Let's implement a simple greedy coloring.
        # We need to be careful with the bound.
        
        # Alternative: Use a simpler bound.
        # If we just use len(P) as bound, it's weak.
        # Let's try to implement the standard MCQ coloring.
        
        # We'll store the color of each vertex in P.
        # And we'll store the vertices grouped by color.
        
        # To make it fast, we can use a list of available colors for each vertex.
        # But for m=127, simple checks are fine.
        
        color_map = {}
        max_color = 0
        # We process vertices in reverse order of the greedy coloring algorithm?
        # No, usually we process in the order defined, and assign smallest available color.
        
        # Let's just use a simple greedy coloring and take the max color as bound.
        # This is an upper bound.
        
        for v in ordered_P:
            used_colors = set()
            for u in adj[v]:
                if u in P_set:
                    if u in color_map:
                        used_colors.add(color_map[u])
            # Find smallest color not in used_colors
            c = 1
            while c in used_colors:
                c += 1
            color_map[v] = c
            if c > max_color:
                max_color = c
                
        # Upper bound for this node is len(R) + max_color
        if len(R) + max_color <= best_size:
            return
            
        # Now branch
        # We should branch on vertices in reverse order of color (highest color first)
        # This is a heuristic to find large cliques early.
        
        # Group vertices by color
        # Or just sort by color descending
        # We need to maintain P for the next recursion.
        
        # We will iterate through vertices in P sorted by color descending.
        # For each vertex v:
        #   Add v to R
        #   New P = P \ {v} intersect neighbors(v)
        #   Recurse
        #   Remove v from P
        
        # To avoid copying P (list) too much, we can use a set or careful list manipulation.
        # Given small size, copying is okay.
        
        # Sort P by color descending
        P_sorted = sorted(P, key=lambda v: color_map[v], reverse=True)
        
        for v in P_sorted:
            # Check bound again?
            # The bound for the sub-problem including v is len(R) + 1 + (max color in new P)
            # But we can just recurse and let the base case handle it.
            
            # New R
            R.append(v)
            
            # New P
            # P' = P \ {v} \ {vertices not adjacent to v}
            # P' = {u in P \ {v} : u is adjacent to v}
            
            # We can filter P
            # Since P is a list, we create a new list.
            # Optimization: only keep u if u in adj[v]
            
            new_P = [u for u in P if u != v and u in adj[v]]
            
            # Recurse
            mcq(R, new_P)
            
            # Backtrack
            R.pop()
            
            # Update P for next iteration?
            # In the standard algorithm, P is modified in place.
            # But here we pass new_P.
            # The loop iterates over P_sorted, which is a static list.
            # This is correct.
            
    # Start with empty R and all vertices in P
    # However, we know 255 must be in the clique.
    # So we can start with R = [idx[255]] and P = neighbors of 255.
    # This reduces the search space.
    
    start_v = idx[target]
    R = [start_v]
    P = list(adj[start_v])
    
    # Run MCQ
    mcq(R, P)
    
    # best_clique contains the indices
    # Convert back to values
    result_values = [pool[i] for i in best_clique]
    
    # Ensure 255 is in the result (it should be, as we started with it)
    if target not in result_values:
        # Fallback just in case
        result_values.append(target)
        
    return result_values

# EVOLVE-BLOCK-END
