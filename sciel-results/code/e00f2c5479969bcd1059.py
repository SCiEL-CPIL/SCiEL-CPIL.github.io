import time
from math import gcd

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses Tomita's Maximum Clique algorithm with bitmask acceleration.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 512
    
    # 1. Build adjacency bitmasks
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Initial Heuristic Solution
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    best_clique = []
    best_size = 0
    
    # Try many highly composite numbers and their divisors
    highly_composite = [120, 180, 240, 360, 420, 480, 504, 432, 384, 448, 462, 486]
    candidate_sets = []
    candidate_sets.append(list(range(1, 24)))  # {1,...,23}, 23*22=506<=512
    
    for hc in highly_composite:
        if hc <= N:
            candidate_sets.append(get_divisors(hc))
    
    # Verify and update best
    for c_set in candidate_sets:
        valid = True
        for i in range(len(c_set)):
            for j in range(i+1, len(c_set)):
                a, b = c_set[i], c_set[j]
                if (a * b) // gcd(a, b) > N:
                    valid = False
                    break
            if not valid:
                break
        if valid and len(c_set) > best_size:
            best_size = len(c_set)
            best_clique = c_set[:]
    
    # 3. Tomita Algorithm with bitmasks for speed
    # Precompute degree for initial sorting
    global_degrees = [(adj[i] >> 1).bit_count() for i in range(1, N + 1)]
    
    def color_sort_from_mask(mask):
        """Greedy coloring of subgraph induced by mask. Returns (ordered_nodes, node_colors)."""
        if mask == 0:
            return [], {}
        
        # Extract nodes from mask
        nodes_in_mask = []
        m = mask
        while m:
            lsb = m & -m
            node = lsb.bit_length() - 1
            nodes_in_mask.append(node)
            m ^= lsb
        
        # Compute local degrees using bitmask operations (fast)
        local_degrees = {}
        for u in nodes_in_mask:
            local_degrees[u] = (adj[u] & mask).bit_count()
        
        # Sort by local degree descending
        nodes_sorted = sorted(nodes_in_mask, key=lambda x: local_degrees[x], reverse=True)
        
        # Greedy coloring using bitmask for color classes
        color_class_masks = []  # bitmask of nodes in each color class
        node_color = {}
        
        for v in nodes_sorted:
            placed = False
            for c in range(len(color_class_masks)):
                # Check if v has no neighbors in color class c
                if (adj[v] & color_class_masks[c]) == 0:
                    color_class_masks[c] |= (1 << v)
                    node_color[v] = c + 1
                    placed = True
                    break
            if not placed:
                color_class_masks.append(1 << v)
                node_color[v] = len(color_class_masks)
        
        # Sort by color
        ordered_nodes = sorted(nodes_sorted, key=lambda x: node_color[x])
        
        return ordered_nodes, node_color

    def expand(clique, candidates_mask):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            return
            
        # Color the subgraph
        ordered_nodes, node_colors = color_sort_from_mask(candidates_mask)
        
        # Iterate from highest color to lowest
        for i in range(len(ordered_nodes) - 1, -1, -1):
            if time.time() > deadline:
                return
                
            v = ordered_nodes[i]
            
            # Pruning: bound is node_colors[v] (max color in prefix)
            if len(clique) + node_colors[v] <= best_size:
                return
                
            clique.append(v)
            
            # New candidates: neighbors of v in the prefix (nodes with index < i)
            # Build prefix mask
            prefix_mask = 0
            for j in range(i):
                prefix_mask |= (1 << ordered_nodes[j])
            
            new_mask = prefix_mask & adj[v]
            
            expand(clique, new_mask)
            
            clique.pop()

    # Start B&B with all nodes
    all_nodes_mask = 0
    for i in range(1, N + 1):
        all_nodes_mask |= (1 << i)
    
    expand([], all_nodes_mask)
    
    return sorted(best_clique)

# EVOLVE-BLOCK-END
