import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    time_limit = 100  # leave buffer for overhead
    
    # Generate support-size-2 candidates (baseline: 220)
    candidates2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates2.append(v)
    candidates2 = np.array(candidates2, dtype=np.int64)
    baseline = len(candidates2)  # 220
    
    # Generate support-size-3 candidates
    candidates3 = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates3.append(v)
    candidates3 = np.array(candidates3, dtype=np.int64)
    n3 = len(candidates3)  # 1320
    
    # Build conflict matrix
    # Conflict if dot product > 1 (since norms are sqrt(3), min distance >= max norm implies
    # ||v_i - v_j||^2 >= ||v_k||^2 for all k. Max norm is sqrt(3).
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j = 3 + 3 - 2 dot = 6 - 2 dot.
    # We need 6 - 2 dot >= 3  => 3 >= 2 dot => dot <= 1.5.
    # Since dot is integer, dot <= 1.
    # So conflict if dot > 1, i.e., dot >= 2.
    dots3 = candidates3 @ candidates3.T
    conflict = (dots3 > 1)
    np.fill_diagonal(conflict, False)
    
    # Build complement graph for Maximum Clique
    # Edge in complement if NOT conflict
    complement = ~conflict
    np.fill_diagonal(complement, False)
    
    # Tomita Maximum Clique Algorithm
    # We want to find the maximum clique in the complement graph.
    
    # Precompute neighbors as bitsets or lists? For 1320 nodes, lists of ints are fine.
    # But for speed, we can use numpy or lists.
    # Let's use lists of ints for neighbors.
    
    neighbors = []
    for i in range(n3):
        neighbors.append(np.where(complement[i])[0].tolist())
    
    best_clique = []
    
    def expand(R, P):
        nonlocal best_clique
        if len(R) + len(P) <= len(best_clique):
            return
        
        if not P:
            if len(R) > len(best_clique):
                best_clique = R.copy()
            return
        
        # Coloring step
        # We order P such that the color of each vertex is an upper bound on the clique size
        # that can be formed by it and its successors.
        
        # Simple greedy coloring for upper bound
        # We want to color P such that vertices with same color are not adjacent.
        # This gives an upper bound: if we have colors 1..k, the max clique in P is <= k.
        
        # To make it tighter, we use the pivot selection from Tomita.
        # However, standard Tomita uses a specific ordering.
        
        # Let's implement a simpler version first: 
        # 1. Color P greedily.
        # 2. Iterate through P in reverse order of color.
        # 3. For each vertex, if len(R) + color <= best_size, break.
        
        # Greedy coloring of P
        # We assign colors to vertices in P.
        # We want to minimize the number of colors, but the order matters for the bound.
        # A common heuristic is to order by degree in P.
        
        # For efficiency, let's just use a simple greedy coloring on the current P set.
        # We'll maintain a list of colors for vertices in P.
        
        # To speed up, we can use the "pivot" technique to select the next vertex.
        # But let's stick to a robust implementation.
        
        # We will iterate through vertices in P.
        # We need to handle the set P efficiently.
        
        # Let's use a list for P to allow easy indexing and removal.
        # But P is passed as a list.
        
        # Coloring:
        # We want to assign a color to each vertex in P.
        # The color of a vertex v is 1 + max(color(u) for u in P that are not adjacent to v and have been colored).
        # Actually, for the upper bound, we want the colors to be as small as possible?
        # No, the bound is: if we process vertices in decreasing order of color, 
        # and we have a vertex v with color c, then the max clique size in the subgraph induced by v and its successors is <= c.
        # So if len(R) + c <= best_size, we can prune.
        
        # To get a tight bound, we want to color the graph such that the max color is small.
        # This is the graph coloring problem, which is NP-hard, but greedy is good enough for bounds.
        
        # Let's implement a greedy coloring that returns the color of each vertex in P.
        # We order P by degree (descending) to get better coloring? Or just arbitrary?
        # Tomita's algorithm uses a specific ordering based on the coloring itself.
        
        # Let's try a simple approach:
        # 1. Create a list of vertices in P.
        # 2. Color them greedily.
        # 3. Sort by color descending.
        # 4. Iterate.
        
        # However, we need to be careful with the time.
        
        # Let's use the standard Tomita coloring:
        # We assign colors to vertices in P.
        # We maintain a set of "uncolored" vertices.
        # We pick a vertex, assign it a color, and remove its neighbors from the "uncolored" set for that color.
        
        # This is getting complex. Let's use a simpler bound:
        # Just use the size of P as the bound? No, that's weak.
        # Let's use a greedy coloring where we just assign the smallest possible color.
        
        # Implementation of greedy coloring for P:
        # p_list = list(P)
        # colors = {}
        # uncolored = set(p_list)
        # color_idx = 1
        # while uncolored:
        #     # Pick a vertex
        #     v = uncolored.pop() # or pick max degree
        #     colors[v] = color_idx
        #     # Remove neighbors of v from uncolored
        #     for u in neighbors[v]:
        #         if u in uncolored:
        #             uncolored.remove(u)
        #     color_idx += 1
        
        # This gives a valid coloring. The max color is an upper bound on the clique size.
        
        # But we need to iterate.
        # We will sort p_list by color descending.
        # Then for each v in sorted_p_list:
        #     if len(R) + colors[v] <= len(best_clique): break
        #     expand(R + [v], P ∩ N(v))
        #     P.remove(v)
        
        # This is the standard algorithm.
        
        # Let's implement it.
        
        p_list = list(P)
        # To improve coloring, we can order p_list by degree in P (descending).
        # Degree in P: len([u for u in p_list if u in neighbors[v]])
        # This might be slow. Let's just use the order they are in P.
        
        # Actually, a better heuristic is to order by the number of neighbors in P.
        # Let's compute degrees in P for better coloring.
        # But for 1320 nodes, this might be okay if we do it once per expand call?
        # No, expand is called many times.
        
        # Let's just use a simple greedy coloring without reordering P initially,
        # but during coloring, we pick the vertex with max degree in the remaining uncolored set.
        
        uncolored = set(p_list)
        colors = {}
        color_idx = 0
        
        # We need to be efficient.
        # Let's use a list for uncolored to avoid set removal overhead?
        # Sets are fine for 1320.
        
        while uncolored:
            # Pick vertex with max degree in uncolored
            # This is O(|uncolored|^2) worst case.
            # Maybe just pick arbitrary?
            # Let's pick arbitrary first to see if it's fast enough.
            v = next(iter(uncolored))
            uncolored.remove(v)
            colors[v] = color_idx
            # Remove neighbors
            to_remove = []
            for u in uncolored:
                if u in neighbors[v]: # This check is O(1) if neighbors[v] is a set
                    to_remove.append(u)
            for u in to_remove:
                uncolored.remove(u)
            color_idx += 1
        
        # Now we have colors for all vertices in P.
        # Sort p_list by color descending
        p_list.sort(key=lambda x: -colors[x])
        
        # Iterate
        for v in p_list:
            if len(R) + colors[v] + 1 <= len(best_clique): # +1 because colors are 0-indexed?
                # If colors are 0..k-1, then max clique size is <= k.
                # So bound is len(R) + colors[v] + 1.
                break
            
            # Expand
            # New P is P ∩ N(v)
            # We need to remove v from P for the next iteration
            # But we are iterating over p_list, which is a copy.
            # We should maintain a set P for the current state.
            
            # Wait, the standard algorithm modifies P.
            # Let's re-structure.
            
            pass # This block is a placeholder for the logic above, let's write it properly in the code.
        
        # The logic above is a bit messy to implement directly in a nested function with nonlocal P.
        # Let's rewrite the expand function to be cleaner.
        
        pass

    # Let's implement the Tomita algorithm properly in the code block below.
    # I will replace the entire construct function.
    
    return candidates2 # Placeholder, will be replaced
