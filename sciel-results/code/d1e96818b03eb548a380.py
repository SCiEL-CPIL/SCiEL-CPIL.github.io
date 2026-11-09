"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard to ensure we finish well before the 150s limit
    if _get_time_remaining() < 1.0:
        # Fallback to simple NN if time is critical
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
        
    # STRATEGY: Geometric Angle-Deficit / Directional Consistency
    # Instead of minimizing immediate distance or savings, we aim to construct a path
    # that approximates the convex hull or a smooth curve. 
    # We estimate the "preferred" direction from the current node towards the destination,
    # and penalize candidates that deviate sharply from this direction.
    # This encourages a "sweep" of the area, which often yields shorter tours than
    # erratic nearest-neighbor jumps, especially in uniform distributions.
    
    # 1. Get coordinates? No, we only have distance_matrix.
    # We can estimate angles using the Law of Cosines if we know 3 points, but we don't have the previous node.
    # Wait, the interface is `select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)`.
    # We do NOT have access to the previous node (the one before current_node).
    # This limits our ability to use "turning angle" directly unless we assume the "direction"
    # is defined by the vector from current to destination.
    
    # If we don't have the previous node, we can't calculate the angle of arrival.
    # However, we can use a "Lookahead Geometric" heuristic:
    # We want to pick a node v such that the path current -> v -> ... -> destination
    # is geometrically efficient.
    
    # Let's look at the triangle (current, v, destination).
    # Let a = d(current, v), b = d(v, destination), c = d(current, destination).
    # By Law of Cosines: c^2 = a^2 + b^2 - 2ab * cos(theta)
    # where theta is the angle at v? No, theta is the angle opposite to c, i.e., at v?
    # Actually, c is the side opposite to angle C (at v? No, at the node opposite c).
    # Sides:
    # a = d(current, v)
    # b = d(v, destination)
    # c = d(current, destination)
    # The angle at v is opposite to side c.
    # cos(angle_at_v) = (a^2 + b^2 - c^2) / (2ab)
    
    # If the angle at v is close to 180 degrees (pi), then v lies on the straight line between current and destination.
    # This is ideal for a direct path.
    # If the angle is small, v is "behind" the path or creates a sharp turn.
    # We want to maximize the angle at v? 
    # If angle is 180, cos is -1. a^2+b^2-c^2 is -2ab.
    # If angle is 0, cos is 1. a^2+b^2-c^2 is 2ab.
    # So we want to MINIMIZE (a^2 + b^2 - c^2) / (2ab) ? 
    # Minimizing the cosine means maximizing the angle.
    # Max angle at v means v is closest to the straight line segment (current, destination).
    
    # Let's test this "Collinearity" heuristic.
    # Score(v) = (a^2 + b^2 - c^2) / (2ab)
    # We want to pick the v with the MINIMUM score (closest to -1, i.e., angle 180).
    # This effectively picks the node that is "on the line" between current and destination.
    # This is a strong heuristic for TSP: it forces the path to move generally towards the destination
    # without taking big detours to the sides, effectively building a "spine" of the tour.
    
    # However, if we always pick the node on the line, we might ignore large areas.
    # But for uniform 50-city instances, the optimal tour is roughly convex.
    # Moving along the "spine" towards the destination is a good first approximation.
    
    # Let's refine: 
    # We want to minimize the deviation from the straight line.
    # Deviation can be measured by the area of the triangle or the angle.
    # Area = 0.5 * c * h, where h is the height of v from the line (current, destination).
    # We want to MINIMIZE the area? 
    # If we minimize area, we pick nodes closest to the line.
    # This creates a very narrow path.
    
    # Alternative: **Angle at Current Node**
    # We don't have the previous node. But we can assume the "desired direction" is towards the destination.
    # Let vector u = destination - current (conceptually).
    # We want v such that vector (v - current) is aligned with u.
    # Cosine similarity between (v-current) and (dest-current).
    # cos(theta) = (d(current,dest)^2 + d(current,v)^2 - d(v,dest)^2) / (2 * d(current,dest) * d(current,v))
    # We want to MAXIMIZE this cosine (theta close to 0).
    # This picks the node that is most "forward" in the direction of the destination.
    
    # Let's compare:
    # 1. Minimize angle at v (collinearity at v) -> v is on the line.
    # 2. Maximize alignment at current (forward progress) -> v is in the direction of dest.
    
    # Heuristic 2 is essentially "Nearest Neighbor in the direction of the destination".
    # This is often better than pure NN because it avoids backtracking.
    
    # Let's implement Heuristic 2: Maximize Forward Progress.
    # Score(v) = cos(theta_v) where theta_v is angle between (current->dest) and (current->v).
    # We want to maximize Score(v).
    # If denominator is 0 (current==dest or current==v), handle separately.
    
    d_cd = distance_matrix[current_node, destination_node]
    if d_cd < 1e-10:
        # If current is destination, just pick nearest
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    d_cv = distance_matrix[current_node, unvisited_nodes]
    d_vd = distance_matrix[destination_node, unvisited_nodes]
    
    # Avoid division by zero
    denom = 2.0 * d_cd * d_cv
    # If d_cv is 0, the node is current, which shouldn't happen if unvisited excludes current.
    # But just in case, mask zeros.
    valid_mask = denom > 1e-10
    
    scores = np.zeros_like(d_cv)
    if np.any(valid_mask):
        # Law of Cosines for angle at Current:
        # d_vd^2 = d_cd^2 + d_cv^2 - 2 * d_cd * d_cv * cos(theta)
        # cos(theta) = (d_cd^2 + d_cv^2 - d_vd^2) / (2 * d_cd * d_cv)
        num = d_cd**2 + d_cv**2 - d_vd**2
        scores[valid_mask] = num[valid_mask] / denom[valid_mask]
        
        # For invalid cases (shouldn't happen), assign a very low score
        scores[~valid_mask] = -2.0
        
        # We want to MAXIMIZE cos(theta).
        best_idx = np.argmax(scores)
    else:
        # Fallback to NN
        best_idx = np.argmin(d_cv)
        
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
