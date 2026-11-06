"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit

# Cache for convex hull perimeter computations
_hull_cache = {}
_hull_cache_size = 0
_MAX_HULL_CACHE = 2000


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def _convex_hull_perimeter(points):
    """Compute the perimeter of the convex hull of a set of 2D points."""
    if len(points) <= 1:
        return 0.0
    if len(points) == 2:
        return 2.0 * np.linalg.norm(points[0] - points[1])
    
    # Sort by x, then y
    points = np.array(points)
    sorted_idx = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_idx]
    
    # Build lower hull
    lower = []
    for p in sorted_points:
        while len(lower) >= 2:
            cross = (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) - \
                    (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in sorted_points[::-1]:
        while len(upper) >= 2:
            cross = (upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) - \
                    (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0])
            if cross <= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    
    # Combine hull (remove last point of each half to avoid duplication)
    hull = np.array(lower[:-1] + upper[:-1])
    
    if len(hull) <= 1:
        return 0.0
    
    # Compute perimeter
    diffs = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(lengths))


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return int(current_node)

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # If time is running out, fall back to nearest neighbor for speed
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Boundary expansion heuristic:
    # Score each candidate by how much it reduces the "enclosed area" metric.
    # We approximate this by: minimize dist(current, candidate) + beta * dist(candidate, destination)
    # but with a twist: we penalize candidates that are far from the "center of mass"
    # of the remaining unvisited nodes, encouraging compact tour construction.
    
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute center of mass of unvisited nodes using distance matrix
    # The center of mass in Euclidean space can be approximated via distances
    # Actually, we can't directly get coordinates from distance matrix alone.
    # Instead, use a different geometric signal: the "spread" of candidates.
    
    # Alternative structural approach: regret-based selection
    # For each candidate, compute the gap between the best and second-best scores
    # This favors candidates that are clearly better than alternatives
    
    # Combine local and global signals:
    # 1. Distance from current (local extension cost)
    # 2. Distance to destination (future connection cost)  
    # 3. Regret signal: how much worse is the 2nd best option?
    
    scores = dist_from_current + 0.5 * dist_to_destination
    
    # Add a compactness penalty: penalize nodes that are far from the median-distance node
    # This encourages keeping the tour "tight"
    median_score_idx = np.argmin(np.abs(scores - np.median(scores)))
    if n_unvisited > 2:
        # Distance from the "central" candidate as a proxy for compactness
        central_node = unvisited_nodes[median_score_idx]
        dist_from_central = distance_matrix[unvisited_nodes, central_node]
        scores = scores + 0.1 * dist_from_central
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
