import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ACO orienteering.
    
    The heuristic eta[i, j] represents the attractiveness of moving from node i to node j.
    We use a prize-to-cost ratio where the cost is the sum of the distance from i to j
    and the distance from j back to the depot (node 0). This encourages visiting high-prize
    nodes that are efficient to reach and return from, respecting the implicit global
    length constraint better than a simple prize/distance ratio.
    """
    n = len(prizes)
    # Ensure prizes is a 1D array
    prizes = np.asarray(prizes).flatten()
    # Ensure distance_matrix is a 2D array
    d = np.asarray(distance_matrix)
    
    # d_j0 is the distance from each node j back to the depot (node 0)
    # Shape: (n,)
    d_j0 = d[:, 0]
    
    # We want eta[i, j] proportional to prize[j] / (d[i, j] + d[j, 0])
    # To vectorize:
    # d[i, j] is d
    # d[j, 0] is d_j0, which needs to be broadcast along the rows (axis 0)
    # So we add d_j0[None, :] to d
    
    # Add epsilon to denominator to avoid division by zero
    # The prompt says non-positive become 1e-9.
    # If prize[j] is 0, the numerator is 0. 0/positive = 0.
    # We should ensure the final matrix has no non-positive values if we want to strictly follow "not positive become 1e-9".
    # However, usually in ACO, 0 heuristic is fine if pheromone is positive, but the prompt says "Heuristic entries that are not positive become 1e-9".
    # This implies we should clamp the result.
    
    eps = 1e-9
    denom = d + d_j0[None, :] + eps
    numer = prizes[None, :] + eps # Adding eps to numerator ensures positive if prize > -eps, but prize >= 0.
    # Actually, if prize is 0, numer is eps. denom is positive. Result is small positive.
    # If we just did prizes / denom, prize 0 would be 0. Then we'd clamp to 1e-9.
    # Let's calculate raw ratio then clamp.
    
    raw_eta = prizes[None, :] / denom
    
    # Apply the constraint: non-positive becomes 1e-9
    # np.where handles this
    eta = np.where(raw_eta > 0, raw_eta, eps)
    
    return eta
