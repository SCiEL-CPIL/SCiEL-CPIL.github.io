import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build Conflict Graph Adjacency Matrix (1 if conflict)
    # Conflict[i][j] is 1 if (i-j) in forbidden or (j-i) in forbidden
    # Since forbidden is symmetric (if d is square, -d is square? 
    # Squares mod 205: if x^2 = d, then (-x)^2 = x^2 = d. 
    # So d is a square iff -d is a square? 
    # Wait. The set of squares is symmetric? 
    # Let S = {x^2 mod 205}. If d in S, is -d in S?
    # 205 = 5 * 41.
    # By Chinese Remainder Theorem, x^2 mod 205 corresponds to (x^2 mod 5, x^2 mod 41).
    # Squares mod 5: {0, 1, 4}. -1 = 4 (square), -4 = 1 (square).
    # Squares mod 41: {0, 1, 4, 5, 9, 16, 25, 33, 36, 39}.
    # -1 mod 41 = 40. Is 40 a square mod 41? 
    # Legendre symbol (40/41) = (-1/41) = (-1)^((41-1)/2) = -1. So -1 is NOT a square mod 41.
    # So the set of squares is NOT symmetric.
    # However, the condition is: neither (a-b) nor (b-a) is in F.
    # This means (a-b) mod 205 not in F AND (b-a) mod 205 not in F.
    # Note that (b-a) mod 205 = (205 - (a-b) mod 205) mod 205.
    # So we need to check both directions.
    
    # Let's build the "Good" graph (Non-conflict graph) directly.
    # Good[i][j] = 1 if i and j can coexist.
    # i, j coexist if (i-j) % m not in F and (j-i) % m not in F.
    
    good = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 not in forbidden and d2 not in forbidden:
                good[i][j] = True
                good[j][i] = True
    
    # 3. Max Clique on Good Graph
    # Tomita Algorithm
    
    best_clique = []
    best_size = 0
    deadline = time.time() + 550 # 550s limit
    
    def greedy_coloring(P, H):
        """
        Greedily color the subgraph induced by P.
        Returns (order, colors) where order is list of vertices sorted by color descending,
        and colors is list of color numbers for each vertex in order.
        """
        if not P:
            return [], []
        
        uncolored = set(P)
        colors = []
        order = []
        color = 0
        
        while uncolored:
            color += 1
            # Find a maximal independent set in the remaining uncolored graph
            # This is equivalent to finding a clique in the complement? No.
            # Greedy coloring: pick a vertex, assign color, remove it and its neighbors from uncolored.
            # We want to maximize the number of colors? No, we want a valid coloring.
            # The standard Tomita approach:
            # Iterate through vertices in P.
            # Assign smallest possible color.
            
            # Let's do a simple greedy coloring:
            # Sort P by degree?
            # For each vertex v in P:
            #   Find smallest color c such that no neighbor of v in P has color c.
            
            # This O(N^2) coloring is fine for N=205.
            vertex_colors = {}
            # To speed up, we can process in a specific order.
            # Let's just iterate through P.
            for v in P:
                used_colors = set()
                for u in P:
                    if u == v: continue
                    if H[v][u]: # u is neighbor of v
                        if u in vertex_colors:
                            used_colors.add(vertex_colors[u])
                c = 1
                while c in used_colors:
                    c += 1
                vertex_colors[v] = c
                
            # Now sort vertices by color descending
            # Vertices with higher colors are more likely to be in the max clique?
            # Actually, the bound is the max color used.
            # We sort by color descending to try high color vertices first.
            sorted_vertices = sorted(P, key=lambda x: vertex_colors[x], reverse=True)
            return sorted_vertices, [vertex_colors[v] for v in sorted_vertices]

    def expand(R, P):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        # Bound
        if len(R) + len(P) <= best_size:
            return
            
        # Greedy Coloring
        order, colors = greedy_coloring(P, good)
        
        # The max color is colors[0]
        # If len(R) + colors[0] <= best_size, prune
        if len(R) + colors[0] <= best_size:
            return
            
        # Iterate in reverse order (highest color first)
        # We need to maintain P as a set or list for intersection
        P_set = set(P)
        
        for i in range(len(order) - 1, -1, -1):
            v = order[i]
            
            # Update bound
            # The max color available for the remaining vertices (order[i+1:]) is colors[i+1] if i < len-1 else 0
            # Actually, the coloring was done on the original P.
            # When we remove v, the colors of remaining vertices might change, but the original color is a valid upper bound?
            # Yes, removing vertices cannot increase the chromatic number.
            # So the max color of the remaining set is at most the max color of the original set.
            # Specifically, if we process in decreasing color order, the max color of the remaining set is colors[i] (current vertex's color is removed, but others <= colors[i] remain? No.)
            # Let's just use the simple bound: len(R) + len(remaining_P)
            # Or better: len(R) + max_color_of_remaining.
            
            # To be safe and simple, let's just use len(R) + len(remaining)
            # But Tomita's efficiency comes from the coloring bound.
            # Let's re-implement the loop properly.
            
            pass # Placeholder, let's restructure

    # The greedy_coloring above is a bit messy for the loop.
    # Let's use a cleaner implementation of Tomita.
    
    def max_clique_tomita():
        nonlocal best_clique, best_size
        # Initial P is all vertices
        P = list(range(m))
        R = []
        
        # We need a helper to get neighbors in P
        # H is the 'good' matrix.
        
        def expand(R, P):
            nonlocal best_clique, best_size
            if time.time() > deadline:
                return
                
            if not P:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = R[:]
                return
            
            # Bound
            if len(R) + len(P) <= best_size:
                return
            
            # Greedy Coloring
            # We want to color the subgraph induced by P.
            # We will assign colors such that adjacent vertices have different colors.
            # We process vertices in a specific order to make the coloring tight.
            # Order by degree in P?
            
            P_list = P[:]
            # Sort by degree in the induced subgraph (descending)
            def degree_in_P(v):
                count = 0
                for u in P_list:
                    if u != v and good[v][u]:
                        count += 1
                return count
            
            P_sorted = sorted(P_list, key=degree_in_P, reverse=True)
            
            vertex_colors = {}
            for v in P_sorted:
                used = set()
                for u in P_sorted:
                    if u == v: continue
                    if good[v][u]:
                        if u in vertex_colors:
                            used.add(vertex_colors[u])
                c = 1
                while c in used:
                    c += 1
                vertex_colors[v] = c
                
            # Sort by color descending
            P_by_color = sorted(P_sorted, key=lambda x: vertex_colors[x], reverse=True)
            
            # Max color is vertex_colors[P_by_color[0]]
            if len(R) + vertex_colors[P_by_color[0]] <= best_size:
                return
            
            # Iterate
            for i in range(len(P_by_color) - 1, -1, -1):
                v = P_by_color[i]
                
                # New P is intersection of current P and neighbors of v
                # P_new = {u in P | good[v][u]}
                P_new = [u for u in P if u != v and good[v][u]]
                
                # Check bound with coloring
                # The max color in P_new is at most vertex_colors[v] (since v is the current one, and we are going down in color)
                # Actually, the colors assigned were for the whole P.
                # The max color in P_new is at most the max color in P.
                # A tighter bound: The max color in P_new is at most vertex_colors[v]?
                # Not necessarily. But we can just use len(R) + len(P_new) for simplicity,
                # or re-color P_new? Re-coloring is expensive.
                # Tomita's algorithm uses the fact that if we order by color,
                # the max color of the remaining set (indices i+1 to end) is colors[i+1].
                # But we are removing v and its non-neighbors.
                
                # Let's just use the simple bound for now, it should be fast enough for N=205.
                if len(R) + len(P_new) <= best_size:
                    continue
                
                expand(R + [v], P_new)
                
        expand(R, P)

    # Run Tomita
    max_clique_tomita()
    
    # If Tomita didn't find anything (shouldn't happen, size 1 is always possible), fallback to greedy
    if not best_clique:
        # Greedy fallback
        selected = []
        for i in range(m):
            ok = True
            for s in selected:
                d1 = (i - s) % m
                d2 = (s - i) % m
                if d1 in forbidden or d2 in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(i)
        best_clique = selected
        best_size = len(selected)

    return np.array(best_clique, dtype=np.int64)

# EVOLVE-BLOCK-END
