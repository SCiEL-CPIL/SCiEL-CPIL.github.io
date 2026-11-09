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
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    deadline = time.time() + 620  # safety margin before 630s kill

    # Problem data
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)  # 15

    # Precompute edge masks for each demand in both directions
    # Edge e connects node e to (e+1) mod 10
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod 10)
    # Counter-clockwise: all other edges

    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw = np.zeros(n_nodes, dtype=np.int64)
        # Clockwise edges: s, s+1, ..., t-1 mod n
        if s <= t:
            for e in range(s, t):
                cw[e] = 1
        else:
            for e in range(s, n_nodes):
                cw[e] = 1
            for e in range(0, t):
                cw[e] = 1
        ccw = 1 - cw
        cw_masks.append(cw)
        ccw_masks.append(ccw)

    cw_masks = np.array(cw_masks)  # shape (15, 10)
    ccw_masks = np.array(ccw_masks)  # shape (15, 10)

    weights = np.array([d[2] for d in demands], dtype=np.int64)  # shape (15,)

    # Base load: all demands routed clockwise
    base_load = weights @ cw_masks  # shape (10,)

    # Delta for each demand: if we switch from CW to CCW, load change is
    # weight * (ccw_mask - cw_mask)
    deltas = (ccw_masks - cw_masks) * weights[:, np.newaxis]  # shape (15, 10)

    best_max_load = float('inf')
    best_routing = np.zeros(D)  # all clockwise as initial
    best_mask = 0

    # Gray code iteration for efficient updates
    # Gray code sequence: G(k) = k ^ (k >> 1)
    # We iterate k from 0 to 2^D - 1
    # Load for mask 0 is base_load
    
    current_load = base_load.copy()
    
    # Check mask 0
    max_load = np.max(current_load)
    if max_load < best_max_load:
        best_max_load = max_load
        best_mask = 0
        best_routing = np.zeros(D)

    prev_gray = 0
    for k in range(1, 1 << D):
        if time.time() > deadline:
            break

        gray = k ^ (k >> 1)
        
        # Find the bit that changed
        diff = gray ^ prev_gray
        bit_idx = diff.bit_length() - 1
        
        # Update load
        # If the bit is set in the new gray code, we added the demand (switched to CCW)
        # If not, we removed it (switched to CW)
        if gray & (1 << bit_idx):
            current_load += deltas[bit_idx]
        else:
            current_load -= deltas[bit_idx]
            
        prev_gray = gray
        
        max_load = np.max(current_load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_mask = gray
            # Extract routing vector from gray code mask
            # Note: The mask here is the Gray code value, which represents the state.
            # We need to map this back to the routing vector.
            # The state is simply which bits are set in `gray`.
            routing = np.zeros(D)
            m = gray
            for j in range(D):
                if m & 1:
                    routing[j] = 1.0
                m >>= 1
            best_routing = routing

    return best_routing


# EVOLVE-BLOCK-END
