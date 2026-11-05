"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    N = 512
    
    # Precompute weights: w(a) = 1 / (512 - a)
    # Higher a means higher weight
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Build conflict graph: two numbers conflict if gcd > 1
    # For each node, store which other nodes it conflicts with
    conflicts = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a + 1, N):
            if gcd(a, b) > 1:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Get candidate list sorted by weight descending (i.e., a descending)
    candidates = list(range(N - 1, 0, -1))
    n = len(candidates)
    
    # Precompute index mapping
    idx_of = [0] * N
    for i, a in enumerate(candidates):
        idx_of[a] = i
    
    # Build conflict matrix in terms of candidate indices
    # conflict_mask[i] is a bitmask of candidates that conflict with candidate i
    # Since n can be up to 511, we use Python ints as bitmasks
    conflict_mask = [0] * n
    for i, a in enumerate(candidates):
        for b in conflicts[a]:
            j = idx_of[b]
            conflict_mask[i] |= (1 << j)
    
    # Greedy coloring for upper bound:
    # Given a set of available nodes (as a bitmask), compute an upper bound
    # using greedy coloring on the induced subgraph
    def greedy_coloring_upper_bound(avail_mask):
        """
        Compute an upper bound on the maximum weight independent set
        using greedy coloring. Returns (upper_bound, color_classes).
        """
        if avail_mask == 0:
            return 0.0, []
        
        # Extract nodes from mask
        nodes = []
        m = avail_mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            nodes.append(i)
            m ^= lsb
        
        # Sort by weight descending (already in candidates order, which is weight descending)
        # But we need to ensure nodes are in the right order
        # Actually, candidates are already sorted by weight desc, so nodes extracted
        # from the mask will be in that order if we process them correctly.
        # Let's just sort explicitly for clarity
        nodes.sort(key=lambda i: -weights[candidates[i]])
        
        # Greedy coloring: assign colors greedily
        color_of = {}
        colors = []  # list of lists, each inner list is a color class (set of node indices)
        
        for node in nodes:
            # Find which colors are used by neighbors of node
            neighbor_colors = set()
            for neighbor in nodes:
                if neighbor == node:
                    continue
                if (conflict_mask[node] >> neighbor) & 1:
                    if neighbor in color_of:
                        neighbor_colors.add(color_of[neighbor])
            
            # Assign the smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            
            color_of[node] = c
            while len(colors) <= c:
                colors.append([])
            colors[c].append(node)
        
        # Upper bound: sum of max weight in each color class
        upper_bound = 0.0
        for color_class in colors:
            if color_class:
                max_w = max(weights[candidates[node]] for node in color_class)
                upper_bound += max_w
        
        return upper_bound, colors
    
    # Branch and bound
    best_score = 0.0
    best_solution = []
    
    # Initial solution via greedy
    chosen = []
    chosen_set = set()
    for a in range(N - 1, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)
    
    initial_score = sum(weights[a] for a in chosen)
    best_score = initial_score
    best_solution = list(chosen)
    
    # B&B state
    # We'll use a stack-based approach
    # State: (avail_mask, current_score, selected_nodes)
    # avail_mask: bitmask of nodes still available
    # current_score: score of selected nodes so far
    # selected_nodes: list of selected node indices
    
    # Initialize available mask with all nodes
    full_mask = (1 << n) - 1
    
    # Stack for DFS
    stack = [(full_mask, 0.0, [])]
    
    nodes_visited = 0
    
    while stack:
        if time.time() > deadline:
            break
        
        avail_mask, curr_score, selected = stack.pop()
        nodes_visited += 1
        
        if avail_mask == 0:
            if curr_score > best_score + 1e-12:
                best_score = curr_score
                best_solution = [candidates[i] for i in selected]
            continue
        
        # Upper bound
        ub, _ = greedy_coloring_upper_bound(avail_mask)
        if curr_score + ub <= best_score + 1e-12:
            continue
        
        # Find a node to branch on
        # Choose the node with highest weight among available
        # Extract nodes from mask
        nodes_in_avail = []
        m = avail_mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            nodes_in_avail.append(i)
            m ^= lsb
        
        if not nodes_in_avail:
            continue
        
        # Sort by weight descending
        nodes_in_avail.sort(key=lambda i: -weights[candidates[i]])
        
        # Branch on the first (highest weight) node
        branch_node = nodes_in_avail[0]
        branch_weight = weights[candidates[branch_node]]
        
        # Option 1: Include branch_node
        # New available mask: remove branch_node and all its conflicts
        new_avail_include = avail_mask & ~(1 << branch_node)
        # Remove conflicts
        cm = conflict_mask[branch_node]
        new_avail_include &= ~cm
        
        new_score = curr_score + branch_weight
        new_selected = selected + [branch_node]
        
        # Option 2: Exclude branch_node
        new_avail_exclude = avail_mask & ~(1 << branch_node)
        
        # Push both options onto stack
        # Push exclude first so include is processed first (LIFO)
        stack.append((new_avail_exclude, curr_score, selected))
        stack.append((new_avail_include, new_score, new_selected))
    
    return best_solution
# EVOLVE-BLOCK-END
