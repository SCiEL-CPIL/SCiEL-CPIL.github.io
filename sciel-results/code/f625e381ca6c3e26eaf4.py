import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors with exactly 4 non-zero entries (+/- 1)
    all_vectors = []
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(comb, signs):
                v[k] = s
            all_vectors.append(v)
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph
    # Two vectors conflict if their distance squared is < 8 (since max norm sq is 4, dist^2 >= 4 required? 
    # Wait, "minimum pairwise distance is at least the maximum vector norm".
    # Max norm is sqrt(4) = 2.
    # So min dist >= 2.
    # dist^2 >= 4.
    # dist^2 = |u|^2 + |v|^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4  => 4 >= 2 u.v => u.v <= 2.
    # So we need u.v <= 2.
    # If u.v > 2, they conflict.
    # u.v is sum of products of entries.
    
    adj = [set() for _ in range(N)]
    
    # Optimized conflict check
    # u.v > 2 means u.v >= 3 (integer dot product).
    # Possible dot products for weight 4 vectors:
    # Intersection size k.
    # If k=4, u.v = sum(signs match) can be -4, -2, 0, 2, 4.
    # If k=3, u.v = sum of 3 terms. Range -3 to 3.
    # If k=2, u.v = sum of 2 terms. Range -2 to 2.
    # If k=1, u.v = +/- 1.
    # If k=0, u.v = 0.
    
    # Conflict if u.v >= 3.
    # This happens if:
    # 1. Intersection size 4 and u.v = 4 (identical vectors, but we have unique vectors so u!=v, so not possible unless duplicates? No, all_vectors are unique). Wait, if u and v are distinct but have same support and same signs, they are the same vector. We generated unique vectors. So u.v=4 only if u==v.
    # 2. Intersection size 4 and u.v = 2? No, 2 is allowed.
    # 3. Intersection size 3 and u.v = 3. This means all 3 intersecting entries have matching signs.
    # 4. Intersection size 4 and u.v = 4 is u=v.
    # 5. Intersection size 4 and u.v = 2 is allowed.
    # 6. Intersection size 3 and u.v = 3 is conflict.
    # 7. Intersection size 3 and u.v = 1 is allowed.
    
    # So conflict iff:
    # - Supports are identical and signs are identical (u=v, not in graph as self-loop)
    # - Supports intersect in 3, and signs match on those 3.
    # - Supports intersect in 4, and signs match in 4 (u=v).
    # - Supports intersect in 4, and signs match in 3? No, if support is same, intersection is 4.
    
    # Let's re-verify:
    # u.v > 2.
    # If |supp(u) \cap supp(v)| = 4: u and v have same support. u.v = sum s_i t_i.
    # Since s_i, t_i in {1, -1}, u.v is even. Possible: -4, -2, 0, 2, 4.
    # u.v > 2 implies u.v = 4. This means s_i = t_i for all i. So u = v.
    # Since we consider distinct vectors, this case doesn't create edges between distinct nodes.
    
    # If |supp(u) \cap supp(v)| = 3:
    # u.v = sum_{i in inter} s_i t_i.
    # Possible values: -3, -1, 1, 3.
    # u.v > 2 implies u.v = 3.
    # This means s_i = t_i for all i in inter.
    
    # If |supp(u) \cap supp(v)| = 2:
    # u.v = sum of 2 terms. Possible: -2, 0, 2.
    # u.v > 2 is impossible.
    
    # So, conflict iff:
    # 1. Same support, same signs (u=v)
    # 2. Intersection size 3, signs match on intersection.
    
    # We can build the graph efficiently by grouping by support.
    
    support_map = {}
    for i, v in enumerate(all_vectors):
        support = tuple(j for j in range(D) if v[j] != 0)
        if support not in support_map:
            support_map[support] = []
        support_map[support].append(i)
        
    # For each pair of supports that intersect in 3, add edges.
    supports = list(support_map.keys())
    sup_list = list(supports)
    
    for i in range(len(sup_list)):
        s1 = sup_list[i]
        for j in range(i + 1, len(sup_list)):
            s2 = sup_list[j]
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                # Find indices in s1 and s2
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs1 = support_map[s1]
                vecs2 = support_map[s2]
                
                # Group vecs2 by sign pattern on inter
                # Sign pattern is a tuple of 3 signs
                sig_map2 = {}
                for v2_idx in vecs2:
                    v2 = all_vectors[v2_idx]
                    sig = tuple(v2[x] for x in inter)
                    if sig not in sig_map2:
                        sig_map2[sig] = []
                    sig_map2[sig].append(v2_idx)
                    
                for v1_idx in vecs1:
                    v1 = all_vectors[v1_idx]
                    sig1 = tuple(v1[x] for x in inter)
                    if sig1 in sig_map2:
                        for v2_idx in sig_map2[sig1]:
                            adj[v1_idx].add(v2_idx)
                            adj[v2_idx].add(v1_idx)
                            
    # 3. Solve MIS using Tomita Algorithm (Branch and Bound with Coloring)
    
    # Convert adj to list of lists for faster access if needed, or keep as sets
    # Tomita requires ordering and coloring.
    
    best_size = 0
    best_set = []
    
    # Greedy coloring for upper bound
    def greedy_color(U):
        # U is a list of vertices
        # Returns (colors, color_classes)
        # colors[u] is the color index (0 to k-1)
        # We want to color such that no two adjacent vertices have the same color.
        # This gives an upper bound on MIS size as the number of colors.
        
        if not U:
            return {}, []
            
        colors = {}
        color_classes = []
        remaining = set(U)
        
        # Order vertices by degree in U (descending) for better coloring?
        # Or just process in given order.
        
        while remaining:
            current_color_class = []
            # Pick a vertex with max degree in remaining?
            # Simple greedy: pick first available
            # To get a good bound, we want to minimize colors.
            # Standard greedy: process vertices in some order, assign smallest available color.
            
            # Let's use a simpler coloring:
            # Iterate through U, assign smallest color not used by neighbors.
            # This is O(V*E) worst case, but V is small in subproblems.
            
            # Actually, for Tomita, we order the vertices in U based on the coloring.
            
            # Let's implement a simple greedy coloring that returns the color of each vertex.
            # And the max color used is the bound.
            
            # Reset colors for U
            for u in U:
                colors[u] = -1
                
            # Order U by degree descending
            sorted_U = sorted(U, key=lambda x: len(adj[x] & set(U)), reverse=True)
            
            current_max_color = 0
            for u in sorted_U:
                # Find smallest color not used by neighbors in U
                used_colors = set()
                for v in adj[u]:
                    if v in colors and colors[v] != -1:
                        used_colors.add(colors[v])
                        
                c = 0
                while c in used_colors:
                    c += 1
                colors[u] = c
                if c > current_max_color:
                    current_max_color = c
                    
            return colors, current_max_color + 1

    # Tomita Search
    # U: list of available vertices
    # C: list of vertices in current independent set
    
    def tomita_search(U, C):
        nonlocal best_size, best_set
        
        if not U:
            if len(C) > best_size:
                best_size = len(C)
                best_set = C[:]
            return
            
        # Get upper bound
        colors, bound = greedy_color(U)
        
        if len(C) + bound <= best_size:
            return
            
        # Order U by color descending
        # Vertices with higher color are more "constrained" or later in the order
        # We branch on the last vertex in the sorted list (highest color)
        
        # Sort U by color
        sorted_U = sorted(U, key=lambda x: colors[x], reverse=True)
        
        # Branch on sorted_U[0] (highest color)
        # Actually, standard Tomita branches on the last vertex in the sequence produced by coloring.
        # If we sort by color descending, the first one has the highest color.
        
        # Let's pick the first vertex in sorted_U (highest color)
        v = sorted_U[0]
        
        # Branch 1: Include v
        # New U = U \ (N(v) U {v})
        new_U_1 = [u for u in sorted_U[1:] if u not in adj[v] and u != v]
        # Note: sorted_U[1:] already excludes v.
        # We need to exclude neighbors of v.
        
        tomita_search(new_U_1, C + [v])
        
        # Branch 2: Exclude v
        # New U = U \ {v}
        new_U_2 = sorted_U[1:]
        
        tomita_search(new_U_2, C)
        
    # Initial U is all vertices
    # To speed up, we can start with a greedy solution to get a good lower bound
    def greedy_mis_init():
        U = set(range(N))
        C = []
        while U:
            # Pick vertex with min degree in U
            min_deg = float('inf')
            v = -1
            for u in U:
                deg = len(adj[u] & U)
                if deg < min_deg:
                    min_deg = deg
                    v = u
            C.append(v)
            U -= adj[v]
            U.discard(v)
        return C
        
    init_sol = greedy_mis_init()
    best_size = len(init_sol)
    best_set = init_sol[:]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Run Tomita
    # To avoid recursion depth issues, we might need to be careful, but N=5280 is deep.
    # However, the branching factor is reduced by pruning.
    
    # One issue: greedy_color is called at every node. This is expensive.
    # But it's the standard way to get strong bounds.
    
    # Let's run it
    tomita_search(list(range(N)), [])
    
    if time.time() - start_time > time_limit:
        # If we timed out, we return the best found so far
        pass
        
    result_vectors = [all_vectors[i] for i in best_set]
    
    if not result_vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(result_vectors, dtype=np.int64)
