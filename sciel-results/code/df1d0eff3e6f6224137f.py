"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for MST maintenance
_mst_state = None
_instance_id_counter = 0
_last_distance_matrix_id = None

def _initialize_mst_state(current_node, distance_matrix):
    """Initialize MST state for a new instance."""
    global _mst_state
    n = distance_matrix.shape[0]
    # mst_cost[i] is the minimum edge cost to connect node i to the current MST
    _mst_state = {
        'mst_cost': np.full(n, np.inf),
        'in_mst': np.zeros(n, dtype=bool),
        'current_mst_total_cost': 0.0,
        'visited_count': 1
    }
    _mst_state['mst_cost'][current_node] = 0.0
    _mst_state['in_mst'][current_node] = True

def _update_mst_state(node, distance_matrix):
    """Update MST state after adding a node to the tour."""
    global _mst_state
    if _mst_state is None:
        return
    
    n = distance_matrix.shape[0]
    mst_cost = _mst_state['mst_cost']
    in_mst = _mst_state['in_mst']
    
    if not in_mst[node]:
        # Add node to MST
        cost_to_add = mst_cost[node]
        _mst_state['current_mst_total_cost'] += cost_to_add
        in_mst[node] = True
        _mst_state['visited_count'] += 1
        
        # Update MST costs for other nodes based on the new node
        # dist_to_new_node = distance_matrix[node, :]
        # For all j not in MST, new_cost_j = min(old_cost_j, dist_to_new_node[j])
        # We only need to update if dist_to_new_node[j] < old_cost_j
        
        # Vectorized update
        distances_from_new = distance_matrix[node, :]
        # Only update nodes not yet in MST
        mask = ~in_mst
        # Use np.minimum to update costs, but only where mask is True
        # We can do this efficiently by taking the minimum of current costs and new distances
        # However, we must not update nodes already in MST (though their cost is 0 or irrelevant)
        # Actually, mst_cost for nodes in MST should ideally be 0 or kept as is, but they are excluded from selection.
        
        # Efficient update:
        # new_costs = np.minimum(mst_cost, distances_from_new)
        # mst_cost[mask] = new_costs[mask]
        
        # To avoid creating a new array every time, we can do:
        np.minimum(mst_cost, distances_from_new, out=mst_cost)
        # Reset costs for nodes in MST to 0? No, it doesn't matter for selection as they are masked out.
        # But for correctness of "current_mst_total_cost", we already added the cost.

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _mst_state, _instance_id_counter, _last_distance_matrix_id
    
    # Detect new instance based on distance matrix identity or size change
    # Using id() is risky if GC reuses addresses, but for short-lived instances in a loop, 
    # checking shape and a hash of the first element is safer.
    # However, the prompt implies 8 distinct instances.
    # Let's use a simpler heuristic: if the current MST state is None or the number of visited nodes
    # doesn't match the expected sequence, re-initialize.
    # A more robust way: track the specific distance_matrix object.
    
    dm_id = id(distance_matrix)
    
    if _mst_state is None or _last_distance_matrix_id != dm_id:
        # New instance detected
        _last_distance_matrix_id = dm_id
        _initialize_mst_state(current_node, distance_matrix)
    
    # Deadline guard: if we are running out of time, fallback to simple NN
    # The function is called 50 times per instance, 8 instances = 400 calls.
    # 150s / 400 = 0.375s per call. This is very generous.
    # But let's add a safety check just in case.
    # We don't have a global start time here easily without passing it.
    # Given the budget is 150s for the whole script, and this function is lightweight,
    # we assume we are within budget.
    
    # If no unvisited nodes, return -1 or something? The contract implies unvisited is non-empty until the end.
    if len(unvisited_nodes) == 0:
        return -1

    # Get current MST costs for unvisited nodes
    # mst_cost[unvisited_nodes] gives the cost to connect each unvisited node to the current MST
    
    mst_costs = _mst_state['mst_cost'][unvisited_nodes]
    
    # We need to decide the scoring function.
    # The directive suggests: "minimizing the distance to the current MST ... plus a penalty proportional to the increase in the MST's total weight"
    # The "increase in MST's total weight" when adding node v is exactly mst_cost[v].
    # So if we just minimize mst_cost[v], we are doing Prim's algorithm on the graph, which builds an MST.
    # However, TSP is a Hamiltonian cycle, not a tree. Prim's algorithm (MST) tends to produce "star-like" or branched structures which are bad for a single path tour.
    # We need to encourage a path-like structure.
    
    # A common heuristic for TSP construction that uses MST ideas is the "MST-based 2-approximation" (double tree), but that's not constructive step-by-step.
    
    # Let's look at the "distance to current node" vs "distance to MST".
    # If we pick the node with min mst_cost, we are growing the MST.
    # To make it a path, we should prefer nodes that are close to the *current end* of the path (current_node) as well.
    
    # Let's try a weighted combination:
    # Score = alpha * dist_to_current + beta * mst_cost
    # If alpha=1, beta=0 -> Nearest Neighbor.
    # If alpha=0, beta=1 -> Prim's (MST growth).
    # Prim's is known to be bad for TSP path construction because it creates branches.
    # However, the directive specifically asks for this structural change.
    # Perhaps we should penalize nodes that are "deep" in the MST?
    
    # Alternative interpretation:
    # The "penalty proportional to the increase in MST's total weight" is just mst_cost.
    # Maybe the idea is to use mst_cost as a *filter* or *secondary* key?
    
    # Let's try a hybrid:
    # 1. Calculate distance from current_node to all unvisited.
    # 2. Calculate mst_cost for all unvisited.
    # 3. Score = dist_to_current + lambda * mst_cost
    
    # Let's experiment with lambda.
    # If lambda is large, it behaves like Prim's.
    # If lambda is small, it behaves like NN.
    # Prim's is 2-approx for TSP (via traversal), but the path itself is not the MST.
    # Constructive heuristics like "MST insertion" exist, but they are not simple "pick next node".
    
    # Let's try a different structural approach:
    # The "MST extension" might imply that we look at the edges in the current MST.
    # When we add a new node, we effectively add one edge to the MST (if we view the tour path as a subgraph).
    # Actually, a tour is a cycle. A path is a tree.
    # If we build the tour as a path, the set of visited nodes forms a tree (the path itself).
    # The cost of the path is the sum of edges.
    # The MST of the visited nodes has cost <= path cost.
    # Minimizing the increase in MST cost is a proxy for minimizing the edge cost, but it ignores the connectivity constraint of a single path.
    
    # Let's try: Score = dist_to_current + 0.5 * mst_cost
    # This encourages nodes that are close to the current position AND close to the existing cluster.
    
    dist_to_current = distance_matrix[current_node, unvisited_nodes]
    
    # Combine scores
    # We want to balance local proximity (dist_to_current) with global structure (mst_cost).
    # A high mst_cost indicates the node is far from the current main component, which might create a long jump later if not handled carefully,
    # OR it might be the only way to close the loop? No, we are building a path.
    
    # Let's try a simple weighted sum.
    # Based on literature, "Nearest Neighbor with a look-ahead" or "MST-based" heuristics often outperform pure NN.
    # Let's use a weight of 0.2 for MST cost.
    
    scores = dist_to_current + 0.2 * mst_costs
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    # Select the node
    next_node = unvisited_nodes[min_idx]
    
    # Update MST state
    _update_mst_state(next_node, distance_matrix)
    
    return int(next_node)
# EVOLVE-BLOCK-END
