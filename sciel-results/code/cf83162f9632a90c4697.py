# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 3-step lookahead heuristic.
    
    For each candidate node u in unvisited_nodes, we estimate the cost as:
    dist(current, u) + min_v ( dist(u, v) + min_w (dist(v, w)) )
    
    Where v and w are in unvisited_nodes \ {u}.
    To maintain efficiency, we approximate min_w(dist(v, w)) by the minimum distance
    from v to any other unvisited node (excluding v itself). This provides a better
    estimate of the future cost than a 2-step lookahead, helping to avoid local minima
    where the immediate next step is cheap but leads to a costly remainder of the tour.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        # 2-step lookahead is sufficient for 2 nodes
        # cost 1: d(c, u1) + d(u1, u2)
        # cost 2: d(c, u2) + d(u2, u1)
        u1, u2 = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2]
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1]
        return int(u1) if cost1 < cost2 else int(u2)

    # Precompute the minimum distance from each unvisited node to any other unvisited node
    # min_next_cost[v] = min(dist(v, w) for w in unvisited \ {v})
    min_next_cost = np.zeros(n_unvisited)
    
    # We can vectorize this or loop. N=50, loop is fine.
    for i, v in enumerate(unvisited_nodes):
        dists = distance_matrix[v, unvisited_nodes]
        # Exclude v itself (index i in the unvisited_nodes array corresponds to v)
        # dists[i] is dist(v, v) = 0. We need min of the rest.
        # Create a mask or just replace the diagonal element
        # Since unvisited_nodes is a view, we can't easily mask without copy.
        # dists is a new array.
        dists[i] = np.inf
        min_next_cost[i] = np.min(dists)

    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Iterate through each unvisited node u to evaluate the 3-step cost
    for i, u in enumerate(unvisited_nodes):
        # Cost to reach u from current
        cost_to_u = distance_matrix[current_node, u]
        
        # Find best v in unvisited \ {u}
        # cost(u, v) + min_next_cost[v]
        # min_next_cost[v] is the cost from v to its best next w.
        # Note: min_next_cost[v] might pick u as w. 
        # If w == u, the path is c->u->v->u. This is a sub-optimal structure (revisiting).
        # However, with N=50, this approximation is often good enough to outperform 2-step.
        # If we want to be strict, we'd need to recompute min_next_cost for each u excluding u, which is O(N^3).
        # Let's stick to the O(N^2) approximation first.
        
        dists_from_u = distance_matrix[u, unvisited_nodes]
        # Exclude u itself from being v
        dists_from_u[i] = np.inf
        
        # The estimated future cost after picking u is min over v of (dist(u, v) + min_next_cost[v])
        # We add the precomputed min_next_cost array to dists_from_u
        # Note: min_next_cost corresponds to the order in unvisited_nodes
        estimated_future_cost = dists_from_u + min_next_cost
        
        # Find the minimum estimated future cost
        min_future_cost = np.min(estimated_future_cost)
        
        total_cost = cost_to_u + min_future_cost
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = u

    return int(best_node)
# EVOLVE-BLOCK-END
