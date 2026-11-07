import math
import sys

def construct():
    # Vertices are 1..256
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute gcd or compatibility
    # compat[u][v] is True if gcd(u, v) == 1
    # We only need this for u < v to save memory/time, or just compute on fly?
    # 256*256 is small.
    gcd_matrix = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_matrix[i][j] = math.gcd(i, j)
            gcd_matrix[j][i] = gcd_matrix[i][j]
            
    # Sort vertices by value descending to prioritize high weights
    # But for the "index > current index" constraint to avoid duplicates,
    # we need a fixed ordering.
    # Let's keep the natural order 1..256 for the "index" constraint?
    # Or sort by value descending and use that order?
    # If we sort by value descending, $v_1 > v_2 > \dots$.
    # We pick $v_i$. Next candidates must be $v_j$ with $j > i$ (lower value).
    # This ensures we don't pick $\{10, 5\}$ and later $\{5, 10\}$.
    # It also biases towards high values, which is good for B&B.
    
    sorted_vertices = sorted(vertices, reverse=True)
    # Map value to index in sorted_vertices
    val_to_idx = {v: i for i, v in enumerate(sorted_vertices)}
    
    # Precompute neighbors in G^c (coprime graph) for each vertex
    # neighbors[v] = list of u in sorted_vertices such that u < v (in sorted order? No, just u != v)
    # And gcd(u, v) == 1
    # For the B&B, when we pick v, the new candidates are 
    # { u in current_candidates | u is after v in sorted_vertices AND gcd(u,v)==1 }
    
    # Let's precompute for each v, the list of u (values) such that gcd(u,v)==1.
    # Then filter by "index > idx_v" during recursion.
    
    coprime_neighbors = [[] for _ in range(N + 1)]
    for v in sorted_vertices:
        for u in sorted_vertices:
            if u == v:
                continue
            if gcd_matrix[v][u] == 1:
                coprime_neighbors[v].append(u)
                
    best_weight = 0
    best_set = []
    
    def color_candidates(candidates):
        """
        Greedily color the subgraph induced by 'candidates' in G^c.
        Returns a dict {value: color} and the bound (sum of max weights per color).
        Vertices in 'candidates' are assumed to be in descending value order?
        We should process them in an order that helps coloring.
        Usually, processing high degree or high weight first is good.
        Here, 'candidates' is passed as a list. 
        We will sort them by weight descending inside this function? 
        The list 'candidates' comes from the previous step.
        To maintain the "index > prev" constraint, the order in 'candidates' 
        must be consistent with the global sorted order.
        So 'candidates' should always be a subsequence of sorted_vertices.
        """
        if not candidates:
            return {}, 0
            
        # We want to assign colors to vertices in 'candidates'.
        # Constraint: If u, v are in candidates and gcd(u,v)==1, they must have different colors.
        # We process candidates in descending value order (which is their order in sorted_vertices).
        color_of = {}
        # We need to know which colors are forbidden for a given v.
        # Forbidden colors are those used by neighbors u (gcd(u,v)==1) that are already colored.
        
        # Optimization: 
        # Since we process in descending value, and 'candidates' is a subsequence of sorted_vertices,
        # the vertices in 'candidates' are already sorted descending.
        
        max_weight_per_color = []
        
        for v in candidates:
            # Find available color
            forbidden = set()
            for u in candidates:
                if u == v:
                    continue
                if u not in color_of:
                    continue # Not colored yet
                if gcd_matrix[v][u] == 1:
                    forbidden.add(color_of[u])
            
            # Find smallest non-negative integer not in forbidden
            c = 0
            while c in forbidden:
                c += 1
                
            color_of[v] = c
            if c >= len(max_weight_per_color):
                max_weight_per_color.append(v)
            else:
                if v > max_weight_per_color[c]:
                    max_weight_per_color[c] = v
                    
        bound = sum(max_weight_per_color)
        return color_of, bound

    def dfs(candidates, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = current_set.copy()
            return
            
        # Calculate bound
        # Note: coloring is expensive. 
        # We can skip coloring if current_weight + sum(candidates) <= best_weight?
        # sum(candidates) is a very loose bound.
        # Let's try the coloring bound.
        
        # Optimization: If the sum of all remaining weights + current <= best, prune.
        # sum(candidates) is O(N).
        rem_sum = sum(candidates)
        if current_weight + rem_sum <= best_weight:
            return
            
        color_of, bound = color_candidates(candidates)
        
        if current_weight + bound <= best_weight:
            return
            
        # Process candidates in order (descending value)
        # To improve pruning, we can order them by color?
        # Standard Tomita: order by color, then weight.
        # Here, we just iterate through candidates.
        
        for i, v in enumerate(candidates):
            # New candidates: 
            # 1. Must be in candidates[i+1:] (to maintain order and avoid duplicates)
            # 2. Must be coprime with v
            new_candidates = []
            for u in candidates[i+1:]:
                if gcd_matrix[v][u] == 1:
                    new_candidates.append(u)
                    
            dfs(new_candidates, current_weight + v, current_set + [v])
            
    # Initial candidates: all vertices
    # We must ensure the initial list is sorted descending.
    initial_candidates = sorted_vertices
    
    dfs(initial_candidates, 0, [])
    
    return best_set
