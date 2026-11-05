"""
Strategy: Sweep-First-Cluster via MDS Projection.
Instead of using pairwise distance savings (Clarke-Wright), we model the problem as a spatial clustering task.
1. We project customer coordinates onto a 2D plane using Multidimensional Scaling (MDS) derived from the distance matrix.
2. We maintain a "heading" vector from the depot to the current node.
3. We select the next node that minimizes the angular deviation from this heading, favoring compact, sweep-like sectors.
This structural change aims to reduce route crossings and create geographically coherent routes, which often outperform greedy savings heuristics on Euclidean-like instances.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

# Global cache for MDS coordinates to avoid recomputation
_mds_coords_cache = {}

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _get_mds_coords(distance_matrix, num_nodes):
    """
    Compute 2D MDS coordinates from the distance matrix.
    Caches results based on the matrix's hash (approximate via first few elements + shape).
    """
    # Simple cache key: shape + hash of first row
    key = (num_nodes, tuple(distance_matrix[0, :5]), tuple(distance_matrix[5, :5]) if num_nodes > 5 else 0)
    
    if key in _mds_coords_cache:
        return _mds_coords_cache[key]
    
    # If time is critical, we can't afford MDS. Return dummy or use a simpler fallback.
    # But MDS is fast for N=50.
    
    # Standard MDS (Classical)
    # D is distance matrix
    D = distance_matrix
    # 1. Compute double centering
    # J = I - (1/n) * 11^T
    n = num_nodes
    J = np.eye(n) - np.ones((n, n)) / n
    # B = -0.5 * J * D^2 * J
    D_sq = D ** 2
    B = -0.5 * (J @ D_sq @ J)
    
    # 2. Eigen decomposition
    # We only need the top 2 eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
    except np.linalg.LinAlgError:
        # Fallback: use random projection or just index-based if MDS fails
        # For robustness, if MDS fails, return identity-like coords
        coords = np.eye(n)
        _mds_coords_cache[key] = coords
        return coords
        
    # Eigenvalues are in ascending order, so take the last 2
    # Note: B is symmetric, so eigenvectors are orthogonal
    # We want the two largest positive eigenvalues
    # eigh returns ascending order.
    idx1 = -1
    idx2 = -2
    
    # Ensure we take the largest
    # Sometimes numerical issues make the largest slightly negative if data is degenerate, but usually fine.
    v1 = eigenvectors[:, idx1]
    v2 = eigenvectors[:, idx2]
    
    # Normalize to unit scale to avoid magnitude issues, though angle is scale-invariant
    coords = np.column_stack((v1, v2))
    
    _mds_coords_cache[key] = coords
    return coords

def _select_sweep_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node based on angular deviation (Sweep Heuristic).
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Find feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # If time is critical, fall back to fast nearest neighbor
    if _time_remaining() < 0.1:
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    num_nodes = len(demands)
    
    # Get MDS coordinates
    coords = _get_mds_coords(distance_matrix, num_nodes)
    
    # Depot is node 0
    depot_coord = coords[depot]
    current_coord = coords[current_node]
    
    # Vector from Depot to Current Node (Heading)
    # If current node is depot, heading is undefined. 
    # In that case, we can pick the node that is "closest" to some arbitrary start, 
    # or simply use distance to depot as a tie-breaker for the first step.
    # Actually, for the first node, any direction is fine. We can just pick nearest.
    if current_node == depot:
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])
        
    heading_vec = current_coord - depot_coord
    
    # If heading vector is zero (depot and current same? shouldn't happen unless current is depot)
    # Handle zero vector
    if np.linalg.norm(heading_vec) < 1e-9:
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    # Calculate vectors from Depot to each feasible candidate
    # We want to minimize the angle between heading_vec and (candidate_coord - depot_coord)
    # Cosine similarity is a good proxy for angle (maximize cos -> minimize angle)
    
    candidate_coords = coords[feasible]
    # Vector from depot to candidate
    cand_vecs = candidate_coords - depot_coord
    
    # Compute norms
    heading_norm = np.linalg.norm(heading_vec)
    cand_norms = np.linalg.norm(cand_vecs, axis=1)
    
    # Avoid division by zero
    cand_norms_safe = np.where(cand_norms < 1e-9, 1e-9, cand_norms)
    
    # Cosine similarity
    # dot(heading, cand) / (norm(heading) * norm(cand))
    dots = np.dot(candidate_coords, heading_vec) - np.dot(depot_coord, heading_vec)
    # Wait, dot(a-b, c) = dot(a,c) - dot(b,c)
    # heading = current - depot
    # cand_vec = cand - depot
    # dot(heading, cand_vec) = dot(current - depot, cand - depot)
    # = dot(current, cand) - dot(current, depot) - dot(depot, cand) + dot(depot, depot)
    
    # Let's just compute it directly for clarity and safety, N is small
    dots = np.sum(cand_vecs * heading_vec, axis=1)
    
    cos_sim = dots / (heading_norm * cand_norms_safe)
    
    # We want to MAXIMIZE cos_sim (minimize angle)
    # However, pure angle minimization might pick a very far node if it's in the same direction.
    # A sweep heuristic typically works on angular sectors. 
    # To ensure we don't jump too far, we can blend angle with distance.
    # Score = w1 * (1 - cos_sim) + w2 * distance
    # Or simply: among nodes with small angle, pick nearest.
    
    # Let's try a weighted score: 
    # Primary: Angle (via 1 - cos_sim)
    # Secondary: Distance to current node
    # This creates a "cone of vision" search.
    
    d_current_feasible = distance_matrix[current_node, feasible]
    
    # Normalize scores to [0, 1] roughly for blending
    # 1 - cos_sim is in [0, 2] (if anti-parallel)
    # d is in [0, max_dist]
    
    # Simple approach: 
    # 1. Filter candidates to those within a certain angle (e.g., 90 degrees, cos > 0)
    # 2. Among those, pick nearest.
    # If none, pick nearest overall.
    
    # Let's use a softer score:
    # Score = d_current_feasible * (1 + (1 - cos_sim))
    # This penalizes large angles by scaling up the distance.
    
    # Let's try: Score = d_current_feasible / (cos_sim + 1.0)
    # If cos_sim = 1 (same direction), Score = d / 2
    # If cos_sim = 0 (perpendicular), Score = d / 1
    # If cos_sim = -1 (opposite), Score = d / 0 -> Inf (bad)
    # This heavily favors forward motion.
    
    # To be safe against division by near-zero, we add a small epsilon.
    # Also, we should probably consider that "forward" isn't always best if the route needs to turn.
    # But sweep heuristics rely on this.
    
    # Let's use a simpler, robust metric:
    # Score = d_current_feasible * (2.0 - cos_sim)
    # Min angle (cos=1) -> factor 1.0
    # Perpendicular (cos=0) -> factor 2.0
    # Max angle (cos=-1) -> factor 3.0
    
    scores = d_current_feasible * (2.0 - cos_sim)
    
    # Find min score
    min_idx = np.argmin(scores)
    best_node = feasible[min_idx]
    
    # Optional: 1-step lookahead to verify
    # If the best node leads to a dead end or very poor next move, consider second best?
    # For now, keep it simple to ensure speed and structural purity.
    
    return int(best_node)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.1:
        # Fallback to fast nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    if len(unvisited_nodes) == 0:
        return depot

    return _select_sweep_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
