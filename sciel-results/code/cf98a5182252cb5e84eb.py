import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    n = limit
    # Set a deadline slightly below the 90s hard limit to be safe
    # The evaluator kills at 90s. We want to return well before.
    deadline = start_time + 85.0
    
    # 1. Precompute neighbors as bitmasks
    # neighbors[i] is an integer where bit j is 1 if lcm(i+1, j+1) <= 512
    neighbors = [0] * n
    
    # Optimization: Compute LCMs efficiently
    # lcm(a, b) = a * b // gcd(a, b)
    for i in range(n):
        num_i = i + 1
        mask = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            lcm_val = (num_i * num_j) // g
            if lcm_val <= limit:
                mask |= (1 << j)
        neighbors[i] = mask

    # Degrees for ordering
    degrees = [bin(neighbors[i]).count('1') for i in range(n)]
    
    # Static order for vertices: Sort by degree descending (which roughly means index ascending for this graph)
    # This order is used for coloring and branching.
    # Pre-compute the list of vertices in this order
    vertex_order = list(range(n))
    vertex_order.sort(key=lambda x: -degrees[x])
    
    best_size = 0
    best_clique_mask = 0
    best_clique_list = []
    
    def update_best(mask):
        nonlocal best_size, best_clique_mask, best_clique_list
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_clique_mask = mask
            # Extract list for local search
            clique = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                clique.append(v)
                m ^= lsb
            best_clique_list = clique

    # Greedy Initialization to get a good lower bound quickly
    def greedy_init():
        current_mask = 0
        # Iterate in vertex_order (high degree first)
        for v in vertex_order:
            # Check if v is compatible with all in current_mask
            # v is compatible with all in current_mask if (neighbors[v] & current_mask) == current_mask
            if (neighbors[v] & current_mask) == current_mask:
                current_mask |= (1 << v)
        return current_mask

    init_mask = greedy_init()
    update_best(init_mask)
    
    # Tomita's Algorithm for Maximum Clique
    # Uses greedy coloring to provide upper bounds for pruning
    
    def color_sort(vertices_mask):
        """
        Greedily colors the vertices in vertices_mask.
        Returns a list of vertices sorted by color (ascending) and the max color used.
        vertices_mask: bitmask of vertices to color.
        """
        if vertices_mask == 0:
            return [], 0
            
        # Extract vertices in the order of vertex_order (pre-sorted by degree)
        # We filter vertex_order to only those in vertices_mask
        verts = []
        for v in vertex_order:
            if vertices_mask & (1 << v):
                verts.append(v)
                
        if not verts:
            return [], 0

        # Greedy Coloring
        # colors[v] is the color assigned to v
        # We want to assign min color not used by neighbors.
        
        # To speed up "find min color not used", we can use a small array of booleans.
        # Max color is at most len(verts).
        
        # We iterate through verts.
        # For each v, we check neighbors in `colored_set`.
        
        # Optimization: Use a list for colors to avoid dict overhead?
        # Since v is 0..511, we can use a list of size n.
        # But we need to reset it? No, we only look at colored nodes.
        # We'll use a dict for clarity, or a list with a "timestamp" trick?
        # Dict is fine for N=512.
        
        color_of = {}
        colored_mask = 0
        max_color = 0
        colored_list = [] # List of (color, vertex)
        
        for v in verts:
            # Find min color not used by colored neighbors
            used_colors = set()
            neigh_mask = neighbors[v] & colored_mask
            nm = neigh_mask
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                used_colors.add(color_of[u])
                nm ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
                
            color_of[v] = c
            if c > max_color:
                max_color = c
            colored_list.append((c, v))
            colored_mask |= (1 << v)
            
        # Sort by color
        colored_list.sort(key=lambda x: x[0])
        # Extract vertices in sorted order
        sorted_verts = [v for c, v in colored_list]
        
        return sorted_verts, max_color

    # Recursive function
    # R: bitmask of current clique
    # P: bitmask of candidates
    def expand(R_mask, P_mask):
        nonlocal best_size, best_clique_mask, best_clique_list
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            update_best(R_mask)
            return
            
        # Color sort P
        sorted_verts, max_c = color_sort(P_mask)
        
        # If |R| + max_c <= best_size, prune
        if bin(R_mask).count('1') + max_c <= best_size:
            return
            
        current_size = bin(R_mask).count('1')
        
        # Tomita's Loop
        # Iterate from the end (highest color) to the start
        # Standard Tomita iterates from the last vertex (highest color) to the first.
        # However, we need to maintain the set of available vertices for each step.
        # In the standard algorithm:
        # P is the set of candidates.
        # We sort P by color.
        # For i from len(P)-1 down to 0:
        #   v = P[i]
        #   if |R| + color[v] <= best: break
        #   R.add(v)
        #   P_new = P[0..i-1] intersect N(v)
        #   expand(R, P_new)
        #   R.remove(v)
        
        # We need to be careful with P_mask updates.
        # In the loop, we effectively remove vertices as we go?
        # No, P_new is calculated from the *original* P (restricted to indices < i) intersected with N(v).
        
        for i in range(len(sorted_verts) - 1, -1, -1):
            v = sorted_verts[i]
            c = color_of[v] # Wait, color_of is local to color_sort. We need it here.
            
            # We need to return color_of from color_sort or re-calculate?
            # Better: return the list of (color, vertex) pairs.
            # Let's modify color_sort to return the list of tuples.
            
            pass # Placeholder, will fix in final code

    # Refactored color_sort to return list of (color, vertex)
    def color_sort_ret(vertices_mask):
        if vertices_mask == 0:
            return []
            
        verts = []
        for v in vertex_order:
            if vertices_mask & (1 << v):
                verts.append(v)
                
        if not verts:
            return []

        color_of = {}
        colored_mask = 0
        max_color = 0
        colored_list = [] # List of (color, vertex)
        
        for v in verts:
            used_colors = set()
            neigh_mask = neighbors[v] & colored_mask
            nm = neigh_mask
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                used_colors.add(color_of[u])
                nm ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
                
            color_of[v] = c
            if c > max_color:
                max_color = c
            colored_list.append((c, v))
            colored_mask |= (1 << v)
            
        colored_list.sort(key=lambda x: x[0])
        return colored_list

    # Refactored expand
    def expand(R_mask, P_mask):
        nonlocal best_size, best_clique_mask, best_clique_list
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            update_best(R_mask)
            return
            
        colored_list = color_sort_ret(P_mask)
        
        if not colored_list:
            return
            
        max_c = colored_list[-1][0]
        
        # If |R| + max_c <= best_size, prune
        if bin(R_mask).count('1') + max_c <= best_size:
            return
            
        current_size = bin(R_mask).count('1')
        
        # We need to know the color of each vertex in the list for pruning.
        # The list is sorted by color.
        # Iterate from last to first.
        
        for i in range(len(colored_list) - 1, -1, -1):
            c, v = colored_list[i]
            
            if time.time() > deadline:
                return

            if current_size + c <= best_size:
                break
                
            # Add v to R
            new_R = R_mask | (1 << v)
            
            # Calculate P_new
            # P_new consists of vertices in colored_list[0..i-1] that are neighbors of v
            P_new = 0
            # Optimization: We can just iterate through all j < i
            # Since i can be up to 512, this loop is small.
            for j in range(i):
                c_u, u = colored_list[j]
                if neighbors[v] & (1 << u):
                    P_new |= (1 << u)
                    
            expand(new_R, P_new)
            
    # Initial call
    initial_P = (1 << n) - 1
    expand(0, initial_P)
    
    # Local Search Polish
    # Try to improve the best clique found
    if time.time() < deadline:
        local_search()
        
    # Convert best_clique to list of 1-based indices
    result = []
    m = best_clique_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    result.sort()
    
    # Ensure nonempty
    if not result:
        return [1]
        
    return result

def local_search():
    pass # Implemented inside construct in the final code to access variables
