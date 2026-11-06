"""Rewritten program using Branch and Bound for Maximum Weight Clique."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Setup
    N = 256
    nodes = list(range(2, N + 1))
    n = len(nodes)
    
    # Map node value to index
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    
    # Weights
    weights = [1.0 / v for v in nodes]
    
    # Adjacency Bitsets
    # adj[i] is a bitmask of nodes j such that lcm(nodes[i], nodes[j]) > 256
    # We exclude self-loops (i != j)
    adj = [0] * n
    
    # Precompute LCM compatibility
    for i in range(n):
        vi = nodes[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = nodes[j]
            if lcm(vi, vj) > 256:
                mask |= (1 << j)
        adj[i] = mask
        
    best_weight = 0.0
    best_clique = []
    
    # Global variables for B&B
    # We will use a recursive function.
    # To avoid recursion depth issues (max depth 255, which is fine), 
    # and for speed, we'll use local variables where possible.
    
    def greedy_coloring(P_mask, order):
        """
        Assigns colors to nodes in P_mask.
        Returns a list of (weight_sum_bound, node_indices_in_order).
        Actually, for the bound, we just need the sum of max weights per color.
        We also need to return the order of nodes sorted by color (for branching).
        
        Standard Tomita:
        1. Color nodes greedily.
        2. Sort nodes by color.
        3. The color of the last node is the bound for cardinality.
        
        For Weighted:
        1. Color nodes greedily.
        2. For each color, find the max weight node.
        3. Sum these max weights.
        4. Return the bound and the nodes sorted by color (higher color first? or lower?).
        In Tomita, nodes are processed in reverse order of color assignment to prune earlier.
        """
        # Get list of active nodes
        active_nodes = []
        for i in range(n):
            if (P_mask >> i) & 1:
                active_nodes.append(i)
        
        # Sort by weight descending to get tighter bounds? 
        # Or just process in index order? 
        # For coloring, processing higher weight nodes first might assign them lower color indices?
        # Actually, we want to maximize the "max weight per color" sum.
        # A simple greedy coloring:
        # Iterate through nodes. Assign the smallest color that doesn't conflict.
        
        # To make the bound tight, we should try to group heavy nodes in separate colors?
        # No, nodes in the *same* color must be independent (non-adjacent).
        # So heavy nodes that are *adjacent* must have different colors.
        # Heavy nodes that are *non-adjacent* can have the same color.
        
        # Let's sort active_nodes by weight descending.
        active_nodes.sort(key=lambda x: -weights[x])
        
        color_of = [-1] * n
        # colors is a list of masks for each color class
        colors = []
        
        for u in active_nodes:
            # Find smallest color c such that u is not adjacent to any node in color c
            # i.e., adj[u] & colors[c] == 0
            assigned = False
            for c in range(len(colors)):
                if (adj[u] & colors[c]) == 0:
                    colors[c] |= (1 << u)
                    color_of[u] = c
                    assigned = True
                    break
            if not assigned:
                colors.append(1 << u)
                color_of[u] = len(colors) - 1
                
        # Calculate Bound: Sum of max weight in each color
        bound = 0.0
        max_weights = [0.0] * len(colors)
        for c in range(len(colors)):
            mask = colors[c]
            max_w = 0.0
            # Iterate bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                if weights[idx] > max_w:
                    max_w = weights[idx]
                m ^= lsb
            max_weights[c] = max_w
            bound += max_w
            
        # Return bound and the list of nodes sorted by color (descending color index?)
        # In Tomita, we process nodes with higher colors first because they have fewer options?
        # Actually, the bound is valid for the whole set.
        # We return the nodes in the order we want to branch.
        # Usually, we branch on nodes in reverse order of color assignment (highest color first).
        
        # Create list of (color, node)
        node_color_pairs = [(color_of[u], u) for u in active_nodes]
        # Sort by color descending
        node_color_pairs.sort(key=lambda x: -x[0])
        
        return bound, [u for _, u in node_color_pairs]

    def expand(C, P_mask, current_w):
        nonlocal best_weight, best_clique
        
        if time.time() > deadline:
            return
        
        if P_mask == 0:
            if current_w > best_weight:
                best_weight = current_w
                best_clique = C[:]
            return
            
        # Get bound and ordered nodes
        bound, ordered_nodes = greedy_coloring(P_mask, None)
        
        if current_w + bound <= best_weight:
            return
            
        # Branch
        # Iterate through ordered_nodes
        # Note: We need to update P_mask as we go? 
        # In standard B&B, we pick a node u, recurse with P' = P & N(u), 
        # then remove u from P and continue with the rest.
        
        for u in ordered_nodes:
            if time.time() > deadline:
                return
            
            # If u is no longer in P_mask (it might have been removed by previous branches? 
            # No, in this loop structure, we are iterating a static list. 
            # We must ensure u is still a candidate. 
            # Since we are removing u from P_mask *after* branching, and the list was generated 
            # from the initial P_mask, u is valid at the start of the iteration.
            # However, if we prune branches, do we remove nodes? 
            # The standard algorithm removes u from P for the *next* iterations in the loop.
            
            if not (P_mask & (1 << u)):
                continue
                
            # Recurse
            new_C = C + [u]
            new_P = P_mask & adj[u]
            expand(new_C, new_P, current_w + weights[u])
            
            # Remove u from P for subsequent branches
            P_mask ^= (1 << u)
            
            # Recalculate bound for the remaining P_mask?
            # The bound `bound` was calculated for the full P_mask.
            # After removing u, the bound decreases.
            # We can just check if current_w + (bound - max_weight_in_color_of_u?) <= best.
            # It's safer and faster to just let the recursive call or the next loop iteration handle it.
            # However, a quick check:
            # If current_w + sum(weights of remaining nodes) <= best, we can break.
            # But we don't have that sum easily.
            # The coloring bound is an upper bound. 
            # If we re-color, it's expensive.
            # Let's just rely on the recursive call's bound check.
            
            # Optimization: If current_w + weights[u] + bound_of_remaining <= best, break?
            # We can approximate bound_of_remaining by the original bound minus the contribution of u's color?
            # This is complex. Let's stick to the simple version first.
            # Given N=255, and the graph is dense, the search tree should be small.

    # Initial call
    initial_P = (1 << n) - 1
    # Note: (1<<n)-1 includes bit 0 to n-1.
    expand([], initial_P, 0.0)
    
    # Convert indices back to values
    result = [nodes[i] for i in best_clique]
    
    # Ensure non-empty (problem guarantees solution exists, e.g., {256} or similar, 
    # actually {256} is valid? lcm(256, x) > 256 for x!=256? 
    # lcm(256, 1) = 256 (not > 256). But 1 is not in [2, 256].
    # lcm(256, 2) = 256 (not > 256).
    # lcm(256, 3) = 768 (> 256).
    # So {256, 3} is valid? lcm(256,3)=768.
    # Is {256} valid? Yes, no distinct pairs.
    # So result will never be empty.
    
    return result

# EVOLVE-BLOCK-END
