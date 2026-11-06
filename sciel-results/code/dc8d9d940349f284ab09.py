"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import squareform, pdist

_START_TIME = None
_DEADLINE = 115.0

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def _get_coordinates(dist_matrix, node_indices):
    """
    Recovers 2D coordinates for a subset of nodes using Classical MDS.
    dist_matrix: full distance matrix
    node_indices: list of node indices to extract
    """
    n = len(node_indices)
    if n < 3:
        # Fallback for small sets
        if n == 1:
            return np.array([[0.0, 0.0]])
        if n == 2:
            d = dist_matrix[node_indices[0], node_indices[1]]
            return np.array([[0.0, 0.0], [d, 0.0]])
        return np.zeros((n, 2))
    
    # Extract sub-distance matrix
    sub_dist = dist_matrix[np.ix_(node_indices, node_indices)]
    
    # MDS
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ (sub_dist ** 2) @ J)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2 dimensions
    # Ensure real values
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    
    return coords

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter to only feasible nodes
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Time guard
    if _time_exceeded():
        dist_to_feasible = distance_matrix[current_node, feasible_nodes]
        best_idx = np.argmin(dist_to_feasible)
        return int(feasible_nodes[best_idx])
    
    # Structural Change: Delaunay-based neighbor selection
    # Nodes of interest: current, depot, and feasible nodes
    # We include depot in the triangulation to anchor the geometry
    # and current to find its neighbors.
    
    # To keep MDS fast and stable, we use a subset if feasible_nodes is large?
    # N is max 50. Full set is fine.
    
    node_indices = np.concatenate(([current_node, depot], feasible_nodes))
    
    try:
        coords = _get_coordinates(distance_matrix, node_indices)
        
        # Compute Delaunay triangulation
        # Note: Delaunay requires points in R^2. MDS gives R^2.
        tri = Delaunay(coords)
        
        # Find the index of current_node in node_indices (it's 0)
        # Find the index of depot in node_indices (it's 1)
        # Indices of feasible_nodes in node_indices: 2 to 2+len(feasible_nodes)-1
        
        # Get neighbors of current_node (index 0) in the triangulation
        # tri.simplices contains triangles (i, j, k)
        # We need to find all triangles containing index 0.
        
        # Efficient way: iterate simplices or use a pre-computed adjacency if available?
        # scipy.spatial.Delaunay has .simplices.
        # For N=50, iterating is fine.
        
        neighbor_indices_in_subset = set()
        for simplex in tri.simplices:
            if 0 in simplex:
                for node_idx in simplex:
                    if node_idx != 0:
                        neighbor_indices_in_subset.add(node_idx)
        
        # Filter neighbors to those corresponding to feasible_nodes
        # feasible_nodes correspond to indices 2..N-1 in node_indices
        # depot is 1, current is 0.
        # We want to go to a feasible node, so we ignore depot (1) and current (0).
        
        candidate_subset_indices = [idx for idx in neighbor_indices_in_subset if idx >= 2]
        
        if not candidate_subset_indices:
            # Fallback if no feasible neighbors in Delaunay graph
            # This can happen if the feasible nodes are far apart or geometry is degenerate
            dist_to_feasible = distance_matrix[current_node, feasible_nodes]
            best_idx = np.argmin(dist_to_feasible)
            return int(feasible_nodes[best_idx])
        
        # Map subset indices back to original node IDs
        # subset index k corresponds to node_indices[k]
        # feasible_nodes are at positions 2.. in node_indices
        # So candidate_subset_indices are indices into node_indices
        # We need the indices into feasible_nodes
        # pos_in_node_indices = candidate_subset_indices
        # pos_in_feasible_nodes = pos_in_node_indices - 2
        
        feasible_local_indices = [idx - 2 for idx in candidate_subset_indices]
        
        # Get the actual node IDs
        candidate_nodes = feasible_nodes[feasible_local_indices]
        
        # Score the candidates
        # Score = dist(current, j) + lambda * dist(j, depot)
        # lambda encourages closing the route (going back to depot) when capacity is low
        # or when few nodes are left.
        
        dist_c_j = distance_matrix[current_node, candidate_nodes]
        dist_j_d = distance_matrix[depot, candidate_nodes]
        
        # Adaptive lambda
        # If rest_capacity is low, we should prefer nodes closer to depot?
        # Or if few nodes left, prefer closer to depot?
        # Let's use a fixed lambda first, or one based on remaining nodes.
        # A common heuristic is to minimize distance to next node + penalty * distance to depot.
        
        # If we have many nodes left, lambda should be small (explore).
        # If we have few nodes left, lambda should be large (close).
        num_unvisited = len(unvisited_nodes)
        # Normalize num_unvisited to 0..1
        # Max unvisited is 50.
        lambda_val = 0.5 * (1.0 - (num_unvisited / 50.0))
        
        # Also consider capacity
        # If capacity is very tight, maybe go to depot?
        # But we must visit all.
        
        scores = dist_c_j + lambda_val * dist_j_d
        
        best_local_idx = np.argmin(scores)
        return int(candidate_nodes[best_local_idx])
        
    except Exception:
        # Fallback to nearest neighbor if MDS or Delaunay fails
        dist_to_feasible = distance_matrix[current_node, feasible_nodes]
        best_idx = np.argmin(dist_to_feasible)
        return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
