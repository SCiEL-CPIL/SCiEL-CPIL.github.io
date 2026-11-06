import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    n = limit
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # 1. Build Conflict Graph
    # Two numbers a, b are in conflict if lcm(a,b) > limit.
    # We want the Maximum Independent Set in this graph, which is the Maximum Clique in the Compatibility Graph.
    # Let's work with the Conflict Graph (E_conflict) and find Max Independent Set (MIS).
    # Or work with Compatibility Graph (E_compat) and find Max Clique (MC).
    # MC is often easier to bound with coloring. Let's stick to MC on Compat Graph.
    
    # neighbors[i] is bitmask of j such that lcm(i+1, j+1) <= limit
    neighbors = [0] * n
    
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
        
    # Degrees
    degrees = [bin(neighbors[i]).count('1') for i in range(n)]
    
    # 2. Exact Maximum Clique via Branch and Bound
    # Using a recursive algorithm with greedy coloring bound.
    # To handle recursion depth and Python overhead, we will use a stack or careful recursion.
    # Given N=512, recursion depth is at most 512. Python default is 1000, so we are safe.
    
    best_size = 0
    best_mask = 0
    
    # Greedy coloring of a set of vertices P.
    # Returns: max_color, and a list of vertices ordered by color (ascending).
    # Vertices with lower colors are processed first in the branch (they are more constrained? No, usually high color = high bound, so we process high color last? 
    # Standard Tomita: Sort by color. Branch on vertex with max color first? 
    # Actually, the bound is |R| + c. If we sort by color, we can prune if |R| + c[v] <= best.
    
    def color_sort(P_mask):
        # P_mask is a bitmask of vertices to color.
        # We need to assign colors 1, 2, ... such that adjacent vertices have different colors.
        # Greedy: process vertices in some order.
        # Order: by degree in the subgraph induced by P? Or just natural order?
        # Let's try natural order of indices in P for simplicity, or sorted by degree in P.
        
        # Extract vertices
        verts = []
        m = P_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        # Sort by degree in P (descending) to get better coloring?
        # Degree in P for v: bin(neighbors[v] & P_mask).count('1')
        # This is expensive if done for every sort. 
        # Let's just use natural order for now, or precomputed global degree.
        # Using global degree as a proxy is faster.
        verts.sort(key=lambda x: -degrees[x])
        
        colors = {}
        max_c = 0
        color_classes = [] # list of lists of vertices
        
        for v in verts:
            # Find min color not used by neighbors in P
            used_colors = set()
            # Check neighbors of v that are in P and already colored
            nb_mask = neighbors[v] & P_mask
            m_nb = nb_mask
            while m_nb:
                lsb = m_nb & -m_nb
                u = lsb.bit_length() - 1
                m_nb ^= lsb
                if u in colors:
                    used_colors.add(colors[u])
                    
            # Find min available color
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_c:
                max_c = c
                color_classes.append([])
            color_classes[c-1].append(v)
            
        # Return max_c and list of vertices sorted by color (low to high)
        # For branching, we typically iterate through vertices in reverse order of color assignment?
        # Or we use the bound: if |R| + color[v] <= best, prune.
        # So we need to know the color of each v in the current P.
        # Return: max_c, dict {v: color}
        
        return max_c, colors, verts

    # Branch and Bound
    # R: current clique (mask)
    # P: candidate vertices (mask)
    
    def expand(R_mask, P_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if bin(R_mask).count('1') > best_size:
                best_size = bin(R_mask).count('1')
                best_mask = R_mask
            return
            
        # Coloring bound
        max_c, colors, verts = color_sort(P_mask)
        
        # If |R| + max_c <= best_size, prune
        if bin(R_mask).count('1') + max_c <= best_size:
            return
            
        # Process vertices in reverse order of color (highest color first)
        # Why? Because high color implies larger potential. 
        # Actually, standard Tomita processes in order of color assignment, but uses the color value for pruning.
        # Let's iterate through verts. 
        # To be efficient, we should pick a vertex v from P.
        # If we pick v, new P = P & neighbors[v].
        
        # Optimization: Pivot selection? 
        # Let's stick to simple B&B first.
        
        # Iterate through vertices in P
        # We can iterate through verts in reverse order of their color?
        # Let's just iterate through verts.
        
        for v in verts:
            if time.time() > deadline:
                return
            
            # Pruning check: if |R| + color[v] <= best_size, skip v
            # Note: colors[v] is the color assigned in the greedy coloring of the *current* P.
            if bin(R_mask).count('1') + colors[v] <= best_size:
                continue
                
            # Include v in R
            new_R = R_mask | (1 << v)
            
            # New P: vertices in P that are compatible with v
            # Must be in P and in neighbors[v]
            new_P = P_mask & neighbors[v]
            
            expand(new_R, new_P)
            
            # Remove v from P for subsequent iterations
            P_mask &= ~(1 << v)
            
            # Update bound? 
            # The coloring was done for the initial P. As P shrinks, the bound max_c might be loose.
            # Recomputing coloring at every step is expensive.
            # Tomita's algorithm re-colors or updates. 
            # For simplicity and given N=512, we might need to be careful.
            # If this is too slow, we can add a heuristic: if P is small, just brute force.
            
            # Early termination if we found a clique of size limit? Unlikely.
            
    # Initial P is all nodes
    initial_P = (1 << n) - 1
    expand(0, initial_P)
    
    # If exact B&B didn't finish or was too slow, we might have a suboptimal result.
    # But the problem size 512 is small enough for a well-implemented B&B in Python?
    # 2^512 is huge. But the graph is structured.
    # Let's add a heuristic fallback if time runs out.
    
    # Check if we have a valid result
    if best_size == 0:
        # Fallback: Greedy
        R = 0
        P = initial_P
        # Sort by degree
        order = sorted(range(n), key=lambda x: -degrees[x])
        for v in order:
            if P & (1 << v):
                if (neighbors[v] & R) == R:
                    R |= (1 << v)
        best_mask = R
        best_size = bin(R).count('1')
        
    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)

construct()
