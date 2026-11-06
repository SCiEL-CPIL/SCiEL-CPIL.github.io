import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return int(depot)

    unvisited_arr = np.array(unvisited_nodes, dtype=np.int64)
    demands_arr = np.array(demands, dtype=np.float64)
    dm = np.array(distance_matrix, dtype=np.float64)

    node_demands = demands_arr[unvisited_arr]
    feasible_mask = node_demands <= rest_capacity

    if not np.any(feasible_mask):
        return int(depot)

    feasible_nodes = unvisited_arr[feasible_mask]
    feasible_demands = demands_arr[feasible_nodes]
    n_feas = len(feasible_nodes)

    # Precompute distances from current node and depot to all feasible nodes
    d_cur = dm[current_node, feasible_nodes]
    d_depot = dm[depot, feasible_nodes]

    # For each candidate j, compute the best "savings" from continuing
    # savings_j = max over k != j, demand[k] <= rest_cap - demand[j] of (d(j,depot) + d(k,depot) - d(j,k))
    # This is the Clarke-Wright savings for the pair (j, k)
    # We want to maximize savings, which means minimizing the effective cost

    best_score = float('inf')
    best_node = int(depot)

    # Precompute d(j, k) matrix for feasible nodes (n_feas x n_feas)
    d_jk_matrix = dm[np.ix_(feasible_nodes, feasible_nodes)]

    # Precompute d(k, depot) for all feasible nodes
    d_kd = d_depot  # same as d_depot since depot is fixed

    for idx in range(n_feas):
        j = feasible_nodes[idx]
        rem_cap = rest_capacity - feasible_demands[idx]

        # Find k candidates: k != idx, demand[k] <= rem_cap
        if rem_cap <= 0:
            # Must return to depot after j
            total_cost = d_cur[idx] + d_depot[idx]
        else:
            k_mask = np.ones(n_feas, dtype=bool)
            k_mask[idx] = False
            k_mask &= (feasible_demands <= rem_cap)

            if np.any(k_mask):
                k_indices = np.where(k_mask)[0]
                # Savings for pairing j with each k: d(j,depot) + d(k,depot) - d(j,k)
                savings = d_depot[idx] + d_kd[k_indices] - d_jk_matrix[idx, k_indices]
                best_savings = np.max(savings)
                # Effective cost: d(current, j) - savings (we "save" by not closing route)
                # But we also need to account for the cost of going to k
                # Actually: if we go to j then k then depot, cost = d(cur,j) + d(j,k) + d(k,depot)
                # If we go to j then depot, cost = d(cur,j) + d(j,depot)
                # Savings = d(j,depot) - d(j,k) - d(k,depot) ... wait, that's negative of what I wrote
                # Let me reconsider:
                # Route via j then k then depot: d(cur,j) + d(j,k) + d(k,depot)
                # Route via j then depot: d(cur,j) + d(j,depot)
                # Difference (savings of going to k): d(j,depot) - d(j,k) - d(k,depot)
                # If this is positive, it's better to go to k
                # So effective cost = d(cur,j) + min(d(j,depot), min_k(d(j,k) + d(k,depot)))
                # = d(cur,j) + d(j,depot) - max(0, max_k(d(j,depot) - d(j,k) - d(k,depot)))
                
                # Recompute properly
                d_jk_vals = d_jk_matrix[idx, k_indices]
                d_kd_vals = d_kd[k_indices]
                # Cost if we go j->k->depot
                cost_jk_dep = d_jk_vals + d_kd_vals
                # Cost if we go j->depot
                cost_j_dep = d_depot[idx]
                # Best continuation cost after j
                best_cont = min(cost_j_dep, np.min(cost_jk_dep))
                total_cost = d_cur[idx] + best_cont
            else:
                total_cost = d_cur[idx] + d_depot[idx]

        # Add a capacity pressure penalty: prefer nodes that leave more remaining capacity
        # This encourages filling routes to capacity
        cap_pressure = feasible_demands[idx] / rest_capacity if rest_capacity > 0 else 1.0
        # Small penalty for low remaining capacity (encourages using full capacity)
        score = total_cost + 0.05 * (1.0 - cap_pressure)

        if score < best_score:
            best_score = score
            best_node = int(j)

    return best_node
