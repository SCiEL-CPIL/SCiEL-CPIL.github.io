"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Nodes are 2..256
    N = 255
    nodes = list(range(2, 257))
    # Map node value to index
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_val = {i: v for i, v in enumerate(nodes)}
    
    # Weights: 1/v
    weights = [1.0 / v for v in nodes]
    
    # Build Compatibility Graph (G_comp)
    # Edge (u, v) if lcm(u, v) > 256
    # We want Max Weight Clique in G_comp.
    
    # Precompute neighbors as bitsets
    # neighbors[i] is a bitmask of indices j such that lcm(nodes[i], nodes[j]) > 256
    neighbors = [0] * N
    
    for i in range(N):
        vi = nodes[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            vj = nodes[j]
            if lcm(vi, vj) > 256:
                mask |= (1 << j)
        neighbors[i] = mask

    # Sort nodes by weight descending to help pruning?
    # Actually, for coloring bound, ordering matters.
    # Let's keep original order but use a specific ordering in the search.
    
    best_weight = 0.0
    best_clique = 0  # Bitset of the best clique found
    
    # Global variables for the search
    # We will use a recursive function
    
    import sys
    sys.setrecursionlimit(10000)
    
    def expand(P, current_clique, current_weight):
        nonlocal best_weight, best_clique
        
        # Check time
        if time.time() > deadline:
            return
            
        if P == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_clique
            return
            
        # Greedy Coloring of P to get upper bound
        # We want to bound the maximum additional weight we can get from P.
        # We assign colors to nodes in P.
        # The bound is sum of max weight per color class?
        # No, the standard bound is:
        # Order nodes in P.
        # Assign colors c1, c2, ...
        # The maximum clique size is bounded by the number of colors.
        # For weight, if we sort nodes by weight descending, and assign colors,
        # the bound for the subgraph induced by nodes from index k onwards is:
        # sum of max weights of the color classes present in that suffix?
        
        # Simpler approach:
        # Get list of nodes in P, sorted by weight descending.
        # Greedily color them.
        # Calculate prefix sums of max weights per color?
        
        # Let's use a standard greedy coloring bound for weight.
        # 1. Extract nodes from P.
        # 2. Sort by weight descending.
        # 3. Greedily assign colors.
        # 4. The upper bound on the weight of a clique in P is the sum of the maximum weights
        #    of each color class? No, that's an upper bound on the sum of weights of an independent set
        #    in the complement? 
        #    
        # Let's stick to the cardinality bound first, or a simple weight bound.
        # A valid upper bound for Max Weight Clique:
        # Sort nodes in P by weight descending: v1, v2, ..., vk.
        # Greedily color v1, ..., vk.
        # Let color(vi) be ci.
        # The bound for the clique containing vi and any subset of {v_{i+1}, ..., v_k} is:
        # weight(vi) + Bound({v_{i+1}, ..., v_k} intersect neighbors(vi)).
        # 
        # To prune, we need a bound on the best clique in P.
        # Upper Bound = sum_{c=1}^{max_color} max_{v in Color_c} weight(v) is NOT correct for clique.
        # In a clique, nodes must have distinct colors.
        # So we can pick at most one node from each color class.
        # Therefore, the maximum weight of a clique is at most the sum of the maximum weights of each color class.
        # Yes, this is a valid upper bound.
        
        # Step 1: Get nodes in P
        p_nodes = []
        temp_p = P
        while temp_p:
            lsb = temp_p & -temp_p
            idx = lsb.bit_length() - 1
            p_nodes.append(idx)
            temp_p ^= lsb
            
        if not p_nodes:
            return

        # Step 2: Sort by weight descending
        p_nodes.sort(key=lambda x: weights[x], reverse=True)
        
        # Step 3: Greedy Coloring
        # colors[i] is the color of p_nodes[i]
        # We also want to track the max weight per color for the bound.
        # However, the bound needs to be calculable for suffixes.
        
        # Let's compute the bound for each position i in p_nodes.
        # bound[i] is the upper bound on the weight of a clique in p_nodes[i:]
        
        # Greedy coloring:
        # color_map: node_idx -> color
        # We process nodes in order.
        # For each node, assign the smallest color not used by its neighbors in P.
        
        # To make it fast, we can just assign colors 1..k.
        # We need the max weight per color for the *remaining* set.
        
        # Alternative simpler bound:
        # Since weights are 1/v, and v >= 2, the max weight is 0.5.
        # A loose bound is len(P) * max_weight_in_P.
        # This might be too loose.
        
        # Let's use the color-based bound.
        # We need to know, for each node in the sorted list, what is the upper bound
        # of the weight of a clique that can be formed from this node and subsequent nodes.
        
        # This is complex to compute dynamically.
        # Let's use a simpler heuristic:
        # Sort P by weight desc.
        # The best possible clique weight is at most the sum of the top K weights,
        # where K is the max clique size of P (bounded by color count).
        
        # Let's compute the greedy coloring and the max clique size bound (cardinality).
        # Then use that to limit the sum of top weights.
        
        # Greedy Coloring for Cardinality Bound:
        # colors_used = 0
        # color_of_node = {}
        # for node in p_nodes:
        #    ...
        
        # Actually, let's just use the sum of top weights as a bound,
        # but capped by the number of colors?
        # No, the sum of top weights is a valid bound for *any* subset, but not tight.
        
        # Let's try the standard Tomita bound calculation.
        # It involves calculating an upper bound for each node in the ordering.
        
        # For simplicity and speed in Python, let's use a slightly looser bound:
        # Upper Bound = Sum of weights of all nodes in P.
        # This is always valid but might not prune much.
        
        # Better:
        # Upper Bound = Sum of top C weights, where C is the max number of colors in greedy coloring of P.
        # Since a clique can have at most C nodes (one per color), the max weight is at most the sum of the C largest weights in P.
        
        # Let's compute C (max colors).
        # Greedy coloring:
        # We need to check neighbors in P.
        
        color_count = 0
        # We don't need to store all colors, just the max.
        # But to get the bound for suffixes, we need more.
        
        # Let's just use the global bound for the current P:
        # C = max colors in greedy coloring of P.
        # Bound = sum of top C weights in P.
        
        # Compute C:
        # This requires iterating through p_nodes and checking neighbors.
        # To speed up, we can break early if time is tight.
        
        # Let's implement the greedy coloring to get C.
        # We use a list of colors for the nodes in p_nodes.
        # node_colors = []
        # max_color = 0
        # for i, node in enumerate(p_nodes):
        #     used_colors = set()
        #     # Check neighbors of node in p_nodes[:i] ? No, in the whole P?
        #     # Greedy coloring usually processes in order.
        #     # A node can only conflict with previously colored nodes if we are doing sequential coloring.
        #     # But standard greedy coloring for bound uses the property that if two nodes are adjacent, they have different colors.
        #     # So we check all previously assigned colors of neighbors in P.
        #     
        #     # Optimization: We can just track which colors are used by neighbors.
        #     # Since N is small, this is fine.
        #     
        #     # Actually, we can just assign the smallest available color.
        #     # Let's do it.
        
        # To avoid O(N^2) per node, we can use bitsets for colors?
        # No, colors are small integers.
        
        # Let's just do it.
        node_colors = {}
        max_color = 0
        # Sort p_nodes by weight desc already done.
        
        for i, node in enumerate(p_nodes):
            # Find colors used by neighbors of node in p_nodes
            # Neighbors in P: neighbors[node] & P
            # But we only care about those already colored (in p_nodes[:i]? No, in the whole P set being colored).
            # In greedy coloring for bound, we typically color the whole set P.
            # The order matters.
            
            # Let's check all neighbors in P.
            # If a neighbor has a color, that color is forbidden.
            # But if we are coloring sequentially, we only know colors of previous nodes.
            # Standard Tomita:
            # "We order the vertices ... and assign colors greedily."
            # "The color of a vertex is the smallest positive integer not used by its neighbors."
            # This implies we look at ALL neighbors in the set P.
            # But if we haven't colored a neighbor yet, we don't know its color.
            # So we must color in an order such that we can determine forbidden colors?
            # No, the standard algorithm colors vertices one by one.
            # If we process v, we check colors of its neighbors that have already been processed?
            # No, that would be a specific ordering.
            
            # Actually, the bound relies on the fact that the graph is k-colorable.
            # We can just compute the chromatic number lower bound?
            # No, we use the greedy coloring count as an upper bound on clique size.
            
            # Let's use a simpler, valid bound:
            # The maximum clique size is at most the size of the maximum independent set in the complement?
            # 
            # Let's use the sum of weights of all nodes in P as the bound.
            # It's safe.
            # To improve, we can sort P by weight and take the sum of the top K,
            # where K is a simple bound like the degree + 1?
            
            # Let's use the "Sum of Top K" where K is the max degree in P + 1?
            # Max clique size <= Max Degree + 1.
            
            # Calculate max degree in P:
            max_deg = 0
            for node in p_nodes:
                deg = (neighbors[node] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
            
            K = max_deg + 1
            
            # Sort weights in P descending
            p_weights = [weights[node] for node in p_nodes]
            p_weights.sort(reverse=True)
            
            bound = sum(p_weights[:K])
            
            if current_weight + bound <= best_weight:
                return

        # Branch and Bound
        # We iterate through nodes in P.
        # To maximize weight, we should try higher weight nodes first.
        # p_nodes is already sorted by weight desc.
        
        for node in p_nodes:
            if time.time() > deadline:
                return
            
            # Pruning check for this specific node
            # The bound for the subproblem including node is:
            # weight(node) + Bound(P \ {node} intersect neighbors(node))
            # We can approximate the bound using the same logic.
            
            # For now, we rely on the global bound at the start of the function.
            # But we can prune inside the loop.
            
            # Let's compute a tighter bound for the branch containing 'node'.
            # New P = P & neighbors[node]
            # New Weight = current_weight + weights[node]
            
            # We can check if New Weight + Sum(Weights in New P) <= best_weight
            # This is a safe bound.
            
            new_P = P & neighbors[node]
            
            # Calculate sum of weights in new_P
            # This is O(N)
            sum_new_P = 0.0
            temp_new_P = new_P
            while temp_new_P:
                lsb = temp_new_P & -temp_new_P
                idx = lsb.bit_length() - 1
                sum_new_P += weights[idx]
                temp_new_P ^= lsb
            
            if current_weight + weights[node] + sum_new_P <= best_weight:
                continue
                
            # Recurse
            expand(new_P, current_clique | (1 << node), current_weight + weights[node])
            
            # Remove node from P for next iteration
            P &= ~(1 << node)
            
            # Re-check bound with reduced P?
            # The initial bound was for the full P.
            # As we remove nodes, the bound decreases.
            # We can re-calculate the bound for the remaining P.
            
            if P == 0:
                break
            
            # Re-calculate max degree bound for remaining P
            max_deg_rem = 0
            for node_rem in p_nodes: # This list is stale, p_nodes contains removed nodes
                if not (P & (1 << node_rem)):
                    continue
                deg = (neighbors[node_rem] & P).bit_count()
                if deg > max_deg_rem:
                    max_deg_rem = deg
            
            K_rem = max_deg_rem + 1
            
            # Get weights of remaining nodes
            rem_weights = []
            temp_rem_P = P
            while temp_rem_P:
                lsb = temp_rem_P & -temp_new_P if False else (temp_rem_P & -temp_rem_P)
                idx = lsb.bit_length() - 1
                rem_weights.append(weights[idx])
                temp_rem_P ^= lsb
            
            rem_weights.sort(reverse=True)
            bound_rem = sum(rem_weights[:K_rem])
            
            if current_weight + bound_rem <= best_weight:
                return

    # Initial call
    # P is all nodes
    initial_P = (1 << N) - 1
    expand(initial_P, 0, 0.0)
    
    # Extract result
    if best_clique == 0:
        # Should not happen, but return a valid non-empty set
        return [2]
        
    result = []
    temp_c = best_clique
    while temp_c:
        lsb = temp_c & -temp_c
        idx = lsb.bit_length() - 1
        result.append(nodes[idx])
        temp_c ^= lsb
        
    return result

# EVOLVE-BLOCK-END
