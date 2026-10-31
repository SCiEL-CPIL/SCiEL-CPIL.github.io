"""Angle-based greedy: penalize deviations from the direction of the destination."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    if unvisited.size == 0:
        return int(current_node)
    
    n = int(distance_matrix.shape[0])
    d = distance_matrix
    cur = int(current_node)
    dest = int(destination_node)

    if time.time() > _DEADLINE:
        return int(unvisited[int(np.argmin(d[cur, unvisited]))])

    # If only 1 or 2 nodes left, nearest neighbor is usually optimal or close enough
    if unvisited.size <= 2:
        return int(unvisited[int(np.argmin(d[cur, unvisited]))])

    # Get coordinates? No, we only have distance matrix.
    # We cannot compute angles directly from distance matrix without coordinates.
    # WAIT. The task says "distance_matrix". It does NOT provide coordinates.
    # My previous assumption about vectors requires coordinates.
    # Can we approximate angles from distance matrix?
    # Law of Cosines: c^2 = a^2 + b^2 - 2ab cos(C)
    # Here, triangle is (cur, dest, candidate).
    # a = dist(cur, dest)
    # b = dist(cur, candidate)
    # c = dist(dest, candidate)
    # cos(angle_at_cur) = (a^2 + b^2 - c^2) / (2 * a * b)
    
    # This is valid! We can compute the angle at 'current_node' between the vector to 'destination' and the vector to 'candidate'.
    # This encourages the path to go "towards" the destination.

    dist_cur_dest = d[cur, dest]
    if dist_cur_dest < 1e-9:
        # Degenerate case, fall back to nearest
        return int(unvisited[int(np.argmin(d[cur, unvisited]))])

    # Precompute distances from current to all unvisited
    dist_cur_unv = d[cur, unvisited]
    
    # Precompute distances from destination to all unvisited
    dist_dest_unv = d[dest, unvisited]
    
    # Calculate cosine of angle at current node for each candidate
    # cos(theta) = (dist_cur_dest^2 + dist_cur_unv^2 - dist_dest_unv^2) / (2 * dist_cur_dest * dist_cur_unv)
    
    # Handle division by zero if dist_cur_unv is 0 (candidate is current node, but unvisited excludes current? 
    # Usually unvisited excludes current. If not, dist is 0, angle undefined. 
    # Assume standard TSP: current is not in unvisited.)
    
    # Vectorized calculation
    # Avoid division by zero
    denom = 2.0 * dist_cur_dest * dist_cur_unv
    # If any denom is 0, set cos to 1 (best case) or handle separately
    # Since unvisited nodes are distinct from current, dist_cur_unv > 0
    # But just in case:
    valid = denom > 1e-9
    cos_theta = np.zeros_like(dist_cur_unv)
    cos_theta[valid] = (dist_cur_dest**2 + dist_cur_unv[valid]**2 - dist_dest_unv[valid]**2) / denom[valid]
    # Clamp cos_theta to [-1, 1] to handle numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Penalty: we want to minimize deviation from 0 angle.
    # Penalty = (1 - cos_theta)
    # Weight: How much to care about direction?
    # Let's try a weight of 0.5
    weight = 0.5
    penalty = weight * (1.0 - cos_theta)
    
    # Total score
    scores = dist_cur_unv + penalty
    
    best_idx = np.argmin(scores)
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
