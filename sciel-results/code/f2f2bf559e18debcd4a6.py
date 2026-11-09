    # ... inside select_next_node ...
    if current_node == depot:
        # First step: Nearest Neighbor
        feasible_nodes.sort(key=lambda u: distance_matrix[depot, u])
        return int(feasible_nodes[0])
    else:
        # Subsequent steps: Savings Heuristic
        scored_nodes = []
        for u in feasible_nodes:
            score = distance_matrix[current_node, u] - distance_matrix[u, depot]
            # Tie-breaker: nearest neighbor
            tie_breaker = distance_matrix[current_node, u]
            scored_nodes.append((score, tie_breaker, u))
            
        scored_nodes.sort(key=lambda x: (x[0], x[1]))
        return int(scored_nodes[0][2])
