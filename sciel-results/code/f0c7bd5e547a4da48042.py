import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1 

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # 1. Get distances from current to all unvisited
    direct_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Get sub-matrix of distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    np.fill_diagonal(sub_matrix, np.inf)
    
    # 3. Compute 1st and 2nd nearest neighbors for each node in unvisited
    # min1[j] is the smallest distance from j to any other node
    # argmin1[j] is the index of that node
    min1 = np.min(sub_matrix, axis=1)
    argmin1 = np.argmin(sub_matrix, axis=1)
    
    # To find min2, we mask out the first neighbor and take the min again
    # We create a copy to mask
    sub_matrix_masked = sub_matrix.copy()
    # Mask out the first neighbor for each row j
    # indices: row is np.arange(N), col is argmin1
    sub_matrix_masked[np.arange(n_unvisited), argmin1] = np.inf
    min2 = np.min(sub_matrix_masked, axis=1)
    
    # 4. Compute 2-step lookahead costs
    # For each candidate i, we want min_j (d(i, j) + S_ij)
    # where S_ij is min2[j] if argmin1[j] == i, else min1[j]
    
    # Build S matrix: S[i, j] is the lookahead cost from j given we came from i
    # argmin1 is shape (N,). We need to compare argmin1[j] with i for all i, j.
    # argmin1[j] == i  =>  argmin1[None, :] == np.arange(n_unvisited)[:, None]
    is_first_neighbor = (argmin1[None, :] == np.arange(n_unvisited)[:, None])
    
    # S[i, j] = min2[j] if is_first_neighbor[i, j] else min1[j]
    # min2 and min1 are shape (N,). We need to broadcast to (N, N)
    # min2[j] depends on column j.
    S = np.where(is_first_neighbor, min2[None, :], min1[None, :])
    
    # Cost matrix for 2-step: C[i, j] = d(i, j) + S[i, j]
    C = sub_matrix + S
    
    # We cannot go from i to i, so set diagonal to inf
    np.fill_diagonal(C, np.inf)
    
    # For each i, the best 2-step cost is min over j
    cost_2step = np.min(C, axis=1)
    
    # Total cost for choosing i as next node
    total_costs = direct_costs + cost_2step
    
    # Select the best candidate
    best_local_idx = np.argmin(total_costs)
    
    return int(unvisited_nodes[best_local_idx])
