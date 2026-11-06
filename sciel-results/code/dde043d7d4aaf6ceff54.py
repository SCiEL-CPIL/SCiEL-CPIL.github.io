"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

# Deadline guard: track start time once at module load
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 2.0  # return safe result this far before deadline

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Delaunay Triangulation-based geometric routing.
    
    Strategy:
    1. Filter unvisited nodes that fit in rest_capacity.
    2. If none fit, return depot (0).
    3. If time is running out, fall back to nearest feasible.
    4. Compute Delaunay triangulation of the set: {current_node} U {feasible_nodes}.
       This creates a sparse geometric graph (planar) of local neighbors.
    5. Among the feasible neighbors of current_node in this triangulation,
       pick the one that minimizes a composite score:
       score = dist(current, candidate) + 0.5 * min_{k in neighbors(candidate) \ {current}} dist(candidate, k)
       If candidate has no other neighbors in the triangulation (or all are infeasible/visited),
       use dist(candidate, depot) as the second term.
       
    This enforces local planarity and prevents long crossing routes, which are suboptimal.
    """
    if _time_exceeded():
        # Fallback to nearest feasible to ensure termination
        feasible = [u for u in unvisited_nodes if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # 1. Filter feasible nodes
    unvisited_list = list(unvisited_nodes)
    if len(unvisited_list) == 0:
        return 0
    
    # Convert to numpy array for fast indexing, but keep list for logic
    feasible_nodes = [u for u in unvisited_list if demands[u] <= rest_capacity]
    
    if len(feasible_nodes) == 0:
        return 0

    # If only one feasible node, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # 2. Prepare points for Delaunay
    # Points: current_node + feasible_nodes
    # We need coordinates. The problem provides distance_matrix.
    # To compute Delaunay, we need 2D coordinates.
    # The task description implies Euclidean geometry is likely underlying the distance matrix.
    # However, we are NOT given coordinates, only the distance matrix.
    # 
    # WAIT. The interface is: select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)
    # We do NOT have access to coordinates (x, y).
    # We only have the distance_matrix.
    #
    # Therefore, we CANNOT compute Delaunay triangulation directly from coordinates.
    # We must infer a geometric structure or use a different "sparse graph" approximation.
    #
    # Alternative: Use a "k-nearest neighbors" graph based on the distance matrix.
    # This approximates the local structure without coordinates.
    # For each node, its k-nearest neighbors in the distance matrix form the sparse graph.
    # This is a structural change from the complete graph to a sparse k-NN graph.
    
    # Let's implement k-NN based local search.
    # k is a small constant, e.g., 5 or 10.
    # Build adjacency for current_node and feasible nodes.
    
    all_nodes_for_graph = [current_node] + feasible_nodes
    n_nodes = len(all_nodes_for_graph)
    
    # Create a local distance matrix for these nodes
    local_dists = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            local_dists[i, j] = distance_matrix[all_nodes_for_graph[i], all_nodes_for_graph[j]]
    
    # For each node in all_nodes_for_graph, find its k-nearest neighbors within this set
    # We want to restrict movement to these neighbors.
    # Specifically, from current_node (index 0), we can only go to feasible nodes that are
    # among its k-nearest neighbors in this local set.
    # Also, for lookahead, from a candidate feasible node, we can only go to other feasible
    # nodes that are among its k-nearest neighbors.
    
    k = min(5, n_nodes - 1) # At least 1 neighbor if possible
    
    # Compute neighbors for each node in the local set
    # neighbors[i] is a set of indices in all_nodes_for_graph
    neighbors = [set() for _ in range(n_nodes)]
    
    for i in range(n_nodes):
        # Sort other nodes by distance from i
        dists_i = local_dists[i, :].copy()
        dists_i[i] = np.inf
        sorted_indices = np.argsort(dists_i)
        # Take top k
        for j in sorted_indices[:k]:
            neighbors[i].add(j)
    
    # Now, from current_node (index 0), the allowed candidates are feasible nodes
    # that are neighbors of 0 in this k-NN graph.
    # If no feasible node is a neighbor, fall back to all feasible (or nearest feasible).
    
    allowed_candidates = []
    for idx in range(1, n_nodes):
        if idx in neighbors[0]:
            allowed_candidates.append(idx)
    
    # If the k-NN restriction is too tight and yields no candidates, use all feasible
    if not allowed_candidates:
        allowed_candidates = list(range(1, n_nodes))
    
    # Score each allowed candidate
    best_node = 0
    best_score = float('inf')
    
    for idx in allowed_candidates:
        node = all_nodes_for_graph[idx]
        
        # Immediate cost
        cost_immediate = local_dists[0, idx]
        
        # Lookahead:
        # From this candidate, where can we go next?
        # Only to other feasible nodes that are neighbors of idx in the k-NN graph.
        # And they must be feasible capacity-wise.
        
        rem_cap_after = rest_capacity - demands[node]
        
        next_candidates = []
        for next_idx in neighbors[idx]:
            if next_idx == 0:
                continue # Don't go back to current immediately
            next_node = all_nodes_for_graph[next_idx]
            # Must be a feasible node (it is, since all_nodes_for_graph[1:] are feasible)
            # But must fit in remaining capacity
            if demands[next_node] <= rem_cap_after:
                next_candidates.append(next_idx)
        
        if next_candidates:
            # Pick the best next step (min distance)
            min_dist = np.min([local_dists[idx, nc] for nc in next_candidates])
            # Also consider returning to depot if it's better?
            # For now, just use the min next step distance.
            # To incorporate depot return, we can add a small weight to dist(node, depot)
            # if there are no good next steps.
            lookahead_cost = min_dist
        else:
            # No further feasible neighbors in k-NN graph. Must return to depot.
            lookahead_cost = distance_matrix[node, depot]
        
        # Composite score
        score = cost_immediate + 0.5 * lookahead_cost
        
        if score < best_score:
            best_score = score
            best_node = node

    return int(best_node)
# EVOLVE-BLOCK-END
