"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for distance matrix to avoid repeated conversions
_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 10 seconds
    return (time.time() - _start_time) > 140

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Lookahead insertion heuristic.
    For each feasible candidate, estimate the marginal cost of adding it to the route.
    We approximate the future route cost by considering:
    1. Distance from current to candidate (immediate cost)
    2. A penalty for how "expensive" it will be to connect the candidate to the rest
       of the route or depot, weighted by the remaining demand.
    
    This differs from pure nearest-neighbor by penalizing nodes that are isolated
    or far from the centroid of remaining feasible nodes, encouraging compact routes.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: if we're running out of time, fall back to nearest feasible
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n = len(feasible_arr)
    
    # Compute distances from current node to all feasible nodes
    dists_to_current = dm[current_node, feasible_arr]
    
    # Compute distances from each feasible node to depot
    dists_to_depot = dm[feasible_arr, depot]
    
    # Compute the centroid of remaining feasible nodes (in index space, weighted by demand? 
    # No, just geometric centroid of coordinates is not available, so use average distance to others)
    # Instead, let's use a "connectivity" score: how close is this node to the rest of the feasible set?
    
    # For each candidate i, compute the average distance to other feasible nodes j != i
    # This penalizes isolated nodes.
    # Also consider the distance to depot, as the route must eventually close.
    
    if n > 1:
        sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
        # Set diagonal to 0 to ignore self
        np.fill_diagonal(sub_dm, 0)
        
        # Average distance to other feasible nodes
        # Sum of distances to others / (n-1)
        avg_dist_to_others = np.sum(sub_dm, axis=1) / (n - 1)
    else:
        avg_dist_to_others = np.array([0.0])
    
    # Composite score:
    # We want to minimize: dist(current -> cand) + lambda * (penalty for isolation)
    # Isolation penalty: avg_dist_to_others + dist_to_depot
    # If a node is far from others AND far from depot, it's expensive to include.
    # However, we must also respect capacity.
    
    # Let's try a different angle: 
    # The total route length is sum of edges.
    # If we pick candidate c, we add edge (current, c).
    # The future edges will connect c to others or depot.
    # A good proxy for "future cost" is the minimum distance from c to any other feasible node 
    # OR to depot, but weighted by how many nodes remain.
    
    # Let's use: score = dist_to_current + beta * (min_dist_to_other_or_depot)
    # But to differentiate from the current program, let's use a "demand-weighted" distance.
    # Nodes with higher demand should be placed earlier if they are close, 
    # but if they are far, they might be better placed later if capacity allows?
    # Actually, capacity is fixed per route.
    
    # Let's try: score = dist_to_current + gamma * (dist_to_depot * demand_norm)
    # This encourages visiting high-demand nodes that are close to the depot early?
    # No, that might spread them out.
    
    # Alternative: Use a "savings" heuristic.
    # Savings of merging two nodes i and j vs going to depot separately:
    # savings(i,j) = dist(depot, i) + dist(depot, j) - dist(i, j)
    # We want to pick the node that maximizes savings with the current node?
    # Or pick the node that has the highest potential savings with any other node?
    
    # Let's implement a simplified savings-based lookahead.
    # For each candidate i, compute the maximum savings it could provide with any other feasible node j.
    # savings(i, j) = dist(depot, i) + dist(depot, j) - dist(i, j)
    # We prioritize nodes that are "easy to merge" with others (high savings).
    
    if n > 1:
        # Compute savings matrix among feasible nodes
        # S[i,j] = d(depot, i) + d(depot, j) - d(i, j)
        d_depot = dists_to_depot
        # S[i,j] = d_depot[i] + d_depot[j] - sub_dm[i,j]
        savings_matrix = d_depot[:, None] + d_depot[None, :] - sub_dm
        # For each i, find the max savings with any j != i
        # Set diagonal to -inf to ignore self
        np.fill_diagonal(savings_matrix, -np.inf)
        max_savings = np.max(savings_matrix, axis=1)
    else:
        max_savings = np.array([0.0])
    
    # Score: minimize distance to current, but penalize low savings (i.e., prefer high savings)
    # score = dist_to_current - delta * max_savings
    # delta controls how much we value merging potential over immediate proximity.
    
    # Normalize savings to be comparable to distances
    max_s = np.max(np.abs(max_savings)) if np.max(np.abs(max_savings)) > 0 else 1.0
    norm_savings = max_savings / max_s
    
    delta = 0.5  # Weight for savings
    scores = dists_to_current - delta * norm_savings
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
