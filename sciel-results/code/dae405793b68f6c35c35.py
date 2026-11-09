# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    N = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    # Precompute edge masks for each demand direction
    # cw_mask[d]: boolean array of length N, True for edges on clockwise path
    # ccw_mask[d]: boolean array of length N, True for edges on counter-clockwise path
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        # Clockwise: edges s, s+1, ..., t-1 (mod N)
        # If s < t: edges s to t-1
        # If s > t: edges s to N-1 and 0 to t-1
        # If s == t: no edges? But demands have s != t typically
        cw = np.zeros(N, dtype=bool)
        if s < t:
            cw[s:t] = True
        elif s > t:
            cw[s:] = True
            cw[:t] = True
        # else s == t: no edges (but unlikely in this problem)
        
        # Counter-clockwise: all other edges
        ccw = ~cw
        # But wait, if s == t, clockwise uses no edges, so counter-clockwise uses all?
        # Let's re-read: "Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10); counter-clockwise uses all other ring edges."
        # So if s == t, clockwise uses 0 edges, counter-clockwise uses all 10.
        # However, looking at the demands, none have s == t.
        
        cw_masks.append(cw)
        ccw_masks.append(ccw)
    
    cw_masks = np.array(cw_masks)  # shape (D, N)
    ccw_masks = np.array(ccw_masks)  # shape (D, N)
    
    def compute_loads(directions):
        """
        directions: array of 0/1 of length D
        0 = clockwise, 1 = counter-clockwise
        Returns: array of edge loads of length N
        """
        # For each demand d:
        # if directions[d] == 0: add weights[d] to edges in cw_masks[d]
        # if directions[d] == 1: add weights[d] to edges in ccw_masks[d]
        
        # Vectorized:
        # loads = sum over d of weights[d] * mask[d]
        # mask[d] = cw_masks[d] if directions[d]==0 else ccw_masks[d]
        
        # Create a 2D mask array: (D, N)
        # For each d, select the appropriate row
        # selected_masks = np.where(directions[:, None] == 0, cw_masks, ccw_masks)
        # loads = (selected_masks * weights[:, None]).sum(axis=0)
        
        selected = np.where(directions[:, None] == 0, cw_masks, ccw_masks)
        return (selected * weights[:, None]).sum(axis=0)
    
    def max_load(directions):
        return np.max(compute_loads(directions))
    
    # Local search
    start_time = time.time()
    deadline = start_time + 600 - 30  # 30s safety margin
    
    # Start with all clockwise
    best_dirs = np.zeros(D, dtype=int)
    best_score = max_load(best_dirs)
    
    # Simple local search: try flipping each bit, keep if better
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for d in range(D):
            if time.time() >= deadline:
                break
            # Try flipping demand d
            new_dirs = best_dirs.copy()
            new_dirs[d] = 1 - new_dirs[d]
            new_score = max_load(new_dirs)
            if new_score < best_score:
                best_dirs = new_dirs
                best_score = new_score
                improved = True
    
    # Return the directions as floats (0 or 1)
    # The task says: "An entry >= 0.5 routes the demand counter-clockwise, any other value clockwise."
    # So 0.0 = clockwise, 1.0 = counter-clockwise
    return best_dirs.astype(np.float64)


# EVOLVE-BLOCK-END
