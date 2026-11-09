"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Build conflict graph: edge between i and j if they form a 3-AP with some k
    # Two elements i, j conflict if there exists k such that {i,j,k} is a 3-AP
    # This means one of i,j,k is the midpoint of the other two.
    # For pair (i,j) with i<j:
    #   - k = 2*j - i (i,j,k AP) - valid if k <= N and k != i,j
    #   - k = i + j - i... wait, let me think again
    #   - (i,j,k): k = 2j - i, need k > j and k <= N
    #   - (i,k,j): k = (i+j)/2, need i < k < j and (i+j) even
    #   - (k,i,j): k = 2i - j, need k < i and k >= 1
    
    # Precompute adjacency as bitmasks for fast operations
    # adj[i] is a bitmask where bit j is set if i and j are in conflict
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Check if {i, j, k} forms a 3-AP for some k in [1, N], k != i, k != j
            # Case 1: i, j, k with i < j < k => k = 2j - i
            k = 2 * j - i
            if k <= N and k != i and k != j:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
                continue
            # Case 2: i, k, j with i < k < j => k = (i+j)/2
            if (i + j) % 2 == 0:
                k = (i + j) // 2
                if 1 <= k <= N and k != i and k != j:
                    adj[i] |= (1 << (j - 1))
                    adj[j] |= (1 << (i - 1))
                    continue
            # Case 3: k, i, j with k < i < j => k = 2i - j
            k = 2 * i - j
            if k >= 1 and k <= N and k != i and k != j:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
    
    # We want to find a maximum weight independent set in this conflict graph
    # where weight of vertex i is 1/i
    # Use branch and bound with greedy coloring upper bound
    
    # Weight of each vertex as float for quick comparison
    weights = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Greedy coloring of a set of vertices (represented as bitmask)
    # Returns an upper bound on the size (or weighted size) of an independent set
    # We use a simple greedy approach: color vertices one by one
    
    def greedy_color_bound(vertices_mask):
        """Compute an upper bound on the weighted independent set size using greedy coloring."""
        if vertices_mask == 0:
            return 0.0
        
        # Get list of vertices in the mask
        vertices = []
        for i in range(1, N + 1):
            if (vertices_mask >> (i - 1)) & 1:
                vertices.append(i)
        
        if not vertices:
            return 0.0
        
        # Sort by weight descending (higher weight first)
        vertices.sort(key=lambda x: -weights[x])
        
        # Greedy coloring: assign colors such that adjacent vertices have different colors
        # The number of colors used is an upper bound on the independence number
        # For weighted case, we sum weights of the first vertex in each color class
        
        color_of = {}  # vertex -> color
        color_classes = []  # list of lists of vertices
        
        for v in vertices:
            # Find the smallest color that v can take
            # v cannot share a color with any of its neighbors that are already colored
            used_colors = set()
            for u in color_of:
                if (adj[v] >> (u - 1)) & 1:
                    used_colors.add(color_of[u])
            
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[v] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(v)
        
        # Upper bound: sum of max weight in each color class
        # Actually, a better bound: for each color class, at most one vertex can be in the independent set
        # So upper bound = sum over color classes of max weight in that class
        
        upper_bound = 0.0
        for cc in color_classes:
            if cc:
                upper_bound += max(weights[v] for v in cc)
        
        return upper_bound
    
    # Better: use a simpler but effective bound
    # For a set of vertices, the maximum weight independent set has weight <= sum of all weights
    # But we can do better with coloring
    
    def compute_bound(vertices_mask):
        """Compute an upper bound on the weight of an independent set within vertices_mask."""
        if vertices_mask == 0:
            return 0.0
        
        # Simple bound: sum of weights of all vertices
        # This is loose but fast
        total = 0.0
        for i in range(1, N + 1):
            if (vertices_mask >> (i - 1)) & 1:
                total += weights[i]
        
        # Try greedy coloring for a tighter bound
        # But this might be slow, so use it selectively
        
        # For now, use the simple sum bound
        return total
    
    # Branch and bound
    # State: (current_set_mask, available_vertices_mask, current_weight)
    # We explore by either including or excluding the next vertex
    
    best_weight = 0.0
    best_set_mask = 0
    
    # Order vertices by weight descending (small numbers first since 1/i is larger for small i)
    vertex_order = list(range(1, N + 1))
    vertex_order.sort(key=lambda x: -weights[x])
    
    def branch_and_bound(index, current_mask, available_mask, current_weight):
        nonlocal best_weight, best_set_mask
        
        if time.time() > deadline:
            return
        
        # Update best if we have a better solution
        if current_weight > best_weight:
            best_weight = current_weight
            best_set_mask = current_mask
        
        # If no more vertices to consider
        if index >= len(vertex_order):
            return
        
        v = vertex_order[index]
        
        # Option 1: Exclude v
        branch_and_bound(index + 1, current_mask, available_mask, current_weight)
        
        # Option 2: Include v (if it doesn't conflict with current set)
        # v can be included if it's not adjacent to any vertex in current_mask
        if (adj[v] & current_mask) == 0:
            # Remove v and all its neighbors from available
            new_available = available_mask & ~(adj[v] | (1 << (v - 1)))
            branch_and_bound(index + 1, current_mask | (1 << (v - 1)), new_available, current_weight + weights[v])
    
    # Start branch and bound
    # Initial available: all vertices
    all_vertices_mask = (1 << N) - 1
    branch_and_bound(0, 0, all_vertices_mask, 0.0)
    
    # Extract final list from best_set_mask
    final_list = []
    for i in range(1, N + 1):
        if (best_set_mask >> (i - 1)) & 1:
            final_list.append(i)
    
    # If the result is empty (shouldn't happen), return a valid default
    if not final_list:
        final_list = [1]
    
    return final_list
# EVOLVE-BLOCK-END
