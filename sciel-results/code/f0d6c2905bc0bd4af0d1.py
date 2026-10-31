"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

# Global start time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _compute_convex_hull_indices(points):
    """
    Computes the convex hull of a set of 2D points.
    Returns the indices of the points on the hull, ordered cyclically.
    Returns None if the hull is degenerate (less than 3 points or collinear).
    """
    if len(points) < 3:
        return None
    
    try:
        # scipy.spatial.ConvexHull works on N x 2 arrays
        # qhull_options='QJ' handles degenerate cases (collinear points) by joggging them
        hull = ConvexHull(points, qhull_options='QJ')
        return hull.vertices
    except Exception:
        return None

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Convex Hull Peeling strategy.
    
    Strategy:
    1. Check time budget. If tight, fall back to nearest neighbor.
    2. Identify the convex hull of the unvisited nodes.
    3. If a hull exists, prioritize nodes on the hull.
       Among hull nodes, select the one closest to the current node.
       This effectively "peels" the hull, keeping the tour on the periphery.
    4. If no hull (all points collinear or few points), select nearest neighbor.
    
    This approach avoids the "crossing" problem of simple greedy TSP heuristics
    where the tour enters the interior of the point cloud, leading to long
    return paths. By staying on the convex hull, the tour length is kept
    closer to the perimeter of the point set, which is a lower bound for
    Euclidean TSP.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # We need coordinates to compute the convex hull.
    # The distance_matrix is Euclidean. We can recover coordinates? 
    # No, we only have the distance matrix. 
    # Wait, the contract says "distance_matrix: np.ndarray". 
    # Usually in these tasks, the distance matrix is provided. 
    # Can we compute the convex hull from a distance matrix?
    # Not directly and efficiently without coordinates.
    
    # Alternative Interpretation:
    # The prompt mentions "Euclidean instances". 
    # If we only have the distance matrix, we cannot easily get the convex hull 
    # unless we assume the input allows us to infer coordinates or we use a different 
    # structural heuristic based on the distance matrix properties.
    
    # However, looking at the inspiration: "global geometric convex-hull peeling".
    # If the evaluator only passes distance_matrix, we are stuck with metric-based heuristics.
    # BUT, often in these EoH tasks, if the problem is Euclidean, the points might be 
    # recoverable or the distance matrix is the only interface.
    
    # Let's re-read the interface:
    # select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)
    # It does NOT pass the coordinates.
    
    # Therefore, we cannot compute the geometric convex hull directly from coordinates.
    # We must emulate "hull-like" behavior using the distance matrix.
    
    # How to emulate hull peeling with just distances?
    # A node is on the "hull" if it is an extreme point.
    # In a metric space, extreme points are those that are far from the centroid?
    # Or, we can use a "farthest point" heuristic combined with nearest?
    
    # Let's pivot the strategy to a **Metric Convex Hull Emulation** or **Angle-based** heuristic 
    # if we can infer angles? No, no coordinates.
    
    # Alternative Structural Strategy: **2-Opt Local Search within the Greedy Step**?
    # No, that's too slow per step.
    
    # Let's look at the "MST reduction" idea from the directive.
    # "Represent the remaining unvisited nodes as a dynamic set where the next city is 
    # selected by maximizing the reduction in the total weight of the Minimum Spanning Tree (MST)..."
    
    # This is computationally expensive (O(N^2 log N) per step for MST) but N=50.
    # 50 steps * O(50^2) = 125,000 ops. Very fast.
    # Let's implement the MST-based heuristic.
    
    # Heuristic:
    # For each candidate node `c` in `unvisited_nodes`:
    # 1. Imagine we add `c` to the current path (which is fixed so far, but we only care about the connection cost).
    # 2. Actually, the directive suggests: "maximizing the reduction in the total weight of the MST of the remaining graph".
    # This is a bit abstract. A simpler, powerful structural heuristic for TSP is **Nearest Neighbor with a "Repulsion" term** 
    # or **Farthest Neighbor from the current path centroid**.
    
    # Let's try a **Centroid-based** heuristic.
    # The optimal TSP tour tends to avoid the center of the mass.
    # Nodes close to the centroid of the unvisited set are "interior" and should be visited later.
    # Nodes far from the centroid are "peripheral" and should be visited earlier.
    
    # Step 1: Compute the "centroid" of unvisited nodes?
    # We don't have coordinates. We have a distance matrix.
    # We can compute the "metric centroid" (Fermat-Weber point) approximation?
    # Or simpler: The node in `unvisited_nodes` that minimizes the sum of distances to all other unvisited nodes is the "center".
    # Nodes that MAXIMIZE this sum are the "extremes" (hull-like).
    
    # Strategy:
    # 1. For each candidate `c`, calculate its "peripherality" score:
    #    peripherality(c) = sum(distance_matrix[c, u] for u in unvisited_nodes)
    #    Higher peripherality = more likely on the hull.
    # 2. We want to pick a node that is:
    #    a) Close to `current_node` (to keep edge short)
    #    b) Highly peripheral (to stay on the hull)
    # 3. Combined Score: 
    #    score(c) = distance_matrix[current_node, c] - beta * peripherality(c)
    #    We want to MINIMIZE score.
    #    So we subtract a reward for being peripheral.
    
    # This is a structural change: from local distance to global metric geometry.
    
    # Optimization:
    # unvisited_nodes is an array.
    # dists_from_current = distance_matrix[current_node, unvisited_nodes]  # Shape (N,)
    # dists_to_all = distance_matrix[unvisited_nodes, :]                  # Shape (N, M) where M is total cities
    # We only care about distances to OTHER unvisited nodes.
    # sum_dists = np.sum(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    # This computes the sum of distances from each unvisited node to all other unvisited nodes.
    
    # Let's refine the score:
    # We want to balance immediate cost (dist to current) with structural benefit (peripherality).
    # Normalize peripherality to be comparable to distance?
    # Or just use a weighted sum.
    
    # Let's test this logic.
    
    # Calculate distances from current to all candidates
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate peripherality: sum of distances from each candidate to all other unvisited nodes
    # Note: distance_matrix[unvisited_nodes] is a 2D array (N_unvisited x N_total)
    # We sum over the columns corresponding to unvisited_nodes
    # This includes the self-distance (0) which is fine.
    dists_between_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    peripherality = np.sum(dists_between_unvisited, axis=1)
    
    # Normalize peripherality to [0, 1] or similar scale to balance with distances?
    # Distances are Euclidean, typically O(1) to O(10) for 50 points in unit square.
    # Sum of distances is O(N * D).
    # Let's normalize peripherality by the max peripherality to get a relative score.
    max_periph = np.max(peripherality)
    if max_periph > 0:
        norm_periph = peripherality / max_periph
    else:
        norm_periph = np.zeros_like(peripherality)
    
    # Heuristic:
    # We want to pick nodes that are close to current BUT also peripheral.
    # If we just pick the most peripheral, we might jump across the map.
    # If we just pick the closest, we get stuck in the center.
    # 
    # Let's use a multiplicative or additive penalty.
    # Score = dist_from_current * (1 + alpha * (1 - norm_periph))
    # If norm_periph is high (1.0), penalty is 1.0 -> Score = dist.
    # If norm_periph is low (0.0), penalty is 1+alpha -> Score = dist * (1+alpha).
    # This penalizes interior nodes.
    
    alpha = 1.0  # Tunable weight for interior penalty
    
    scores = dists_from_current * (1.0 + alpha * (1.0 - norm_periph))
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
