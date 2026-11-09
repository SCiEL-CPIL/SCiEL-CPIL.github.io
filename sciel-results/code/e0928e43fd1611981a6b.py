import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Filter self-conflicts
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    
    # Build adjacency masks using 1-based bit positions
    adj_mask = [0] * (N + 1)
    
    for v in valid_nodes:
        for u in valid_nodes:
            if u > v and (v + u) in squares:
                adj_mask[v] |= (1 << u)
                adj_mask[u] |= (1 << v)
                
    initial_mask = 0
    for v in valid_nodes:
        initial_mask |= (1 << v)
        
    # Mod 3 baseline: v % 3 == 1
    best_set = set(v for v in range(1, N+1) if v % 3 == 1)
    best_size = len(best_set)
    
    # Branch and Bound for Maximum Independent Set
    # Using Tomita-style coloring bound for pruning
    
    def greedy_color_bound(mask):
        """Compute upper bound on clique size (independent set in complement) using greedy coloring."""
        if mask == 0:
            return 0
        
        colors = [0] * (N + 1)
        color_count = 0
        remaining = mask
        
        while remaining:
            color_count += 1
            color_mask = 0
            
            # Greedy: pick nodes in decreasing degree order within remaining
            # For simplicity, just iterate through set bits
            temp_remaining = remaining
            while temp_remaining:
                # Get lowest set bit
                v = (temp_remaining & -temp_remaining).bit_length() - 1
                temp_remaining &= temp_remaining - 1
                
                # Check if v can be added to current color class
                # (i.e., no neighbor already in this color class)
                if color_mask & adj_mask[v] == 0:
                    color_mask |= (1 << v)
                    colors[v] = color_count
            
            remaining &= ~color_mask
        
        return color_count
    
    def branch_and_bound(mask, current_set, depth):
        """Recursively find maximum independent set using branch and bound."""
        nonlocal best_set, best_size, deadline, start_time
        
        if time.time() > deadline - 5.0:
            return
        
        if mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Upper bound: current size + maximum possible additional nodes
        # Use greedy coloring bound on the subgraph induced by mask
        upper_bound = len(current_set) + greedy_color_bound(mask)
        
        if upper_bound <= best_size:
            return
        
        # Get a node to branch on
        # Strategy: pick node with maximum degree in mask for better pruning
        max_deg = -1
        branch_node = -1
        temp_mask = mask
        while temp_mask:
            v = (temp_mask & -temp_mask).bit_length() - 1
            temp_mask &= temp_mask - 1
            deg = (adj_mask[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                branch_node = v
        
        if branch_node == -1:
            return
        
        # Branch 1: Include branch_node
        new_set = current_set + [branch_node]
        # Remove branch_node and its neighbors from mask
        new_mask = mask & ~(adj_mask[branch_node] | (1 << branch_node))
        branch_and_bound(new_mask, new_set, depth + 1)
        
        # Branch 2: Exclude branch_node
        new_mask = mask & ~(1 << branch_node)
        branch_and_bound(new_mask, current_set, depth + 1)
    
    # Run branch and bound
    branch_and_bound(initial_mask, [], 0)
    
    # Verify the result
    result = list(best_set)
    for i, a in enumerate(result):
        for b in result[i:]:
            if (a + b) in squares:
                # Fallback to mod3
                return [v for v in range(1, N+1) if v % 3 == 1]
    
    return result
