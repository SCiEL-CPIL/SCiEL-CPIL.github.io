import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic matrix for ant colony orienteering.
    
    Structural change: Uses a sharp ratio of prize to detour cost,
    raised to a higher power to emphasize high-efficiency moves.
    
    eta[i,j] = (prize[j] / (d[i,j] + d[j,0]))^3
    
    This formulation directly rewards nodes that offer high prize
    relative to the "commitment" required to visit them and return.
    The cubic power sharpens the preference for the best moves,
    helping the ACO converge faster to high-quality paths by
    reducing exploration of suboptimal edges.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    eps = 1e-9
    
    # Calculate the "commitment" cost for each edge (i, j)
    # This is the distance to go to j and then back to 0.
    # d[i, j] is shape (N, N)
    # d[j, 0] is shape (N,), broadcast along rows
    detour = distance_matrix + distance_matrix[:, 0][None, :]
    
    # Calculate the prize-to-cost ratio
    # prize[j] is shape (N,), broadcast along columns
    prize_term = prizes + eps
    cost_term = detour + eps
    
    ratio = prize_term[None, :] / cost_term
    
    # Raise to power 3 to sharpen the heuristic
    heuristic = ratio ** 3
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple fallback heuristic
        fallback = np.ones_like(distance_matrix)
        return fallback
    
    # Ensure validity: non-positive or non-finite values become 1e-9
    # The ratio is always positive since we added eps.
    # However, if detour is huge, ratio is small.
    # If prize is 0, ratio is eps/cost which is tiny.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
