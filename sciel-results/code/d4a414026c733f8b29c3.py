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
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    start_time = time.time()
    DEADLINE = 620.0  # safety margin before 630s kill

    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW edge masks for each demand
    # CW from s to t: edges s, s+1, ..., t-1 (mod 10)
    cw_masks = []
    ccw_masks = []
    weights = []
    for (s, t, w) in demands:
        cw = 0
        if s <= t:
            for e in range(s, t):
                cw |= (1 << e)
        else:
            for e in range(s, N):
                cw |= (1 << e)
            for e in range(0, t):
                cw |= (1 << e)
        cw_masks.append(cw)
        # CCW uses all other edges
        all_edges = (1 << N) - 1
        ccw_masks.append(all_edges ^ cw)
        weights.append(w)

    # Convert to numpy arrays for speed
    cw_masks_arr = np.array(cw_masks, dtype=np.int32)
    ccw_masks_arr = np.array(ccw_masks, dtype=np.int32)
    weights_arr = np.array(weights, dtype=np.float64)

    def compute_max_load(routing_bits):
        """Compute max edge load given routing bits (0=CW, 1=CCW)."""
        # For each edge, sum weights of demands routed through it
        # Use the mask approach: for each demand, add weight to edges in its mask
        loads = np.zeros(N, dtype=np.float64)
        for i in range(D):
            mask = ccw_masks_arr[i] if routing_bits[i] else cw_masks_arr[i]
            # Extract edge loads from bitmask
            # This is a bit slow; let's use a different approach
            for e in range(N):
                if mask & (1 << e):
                    loads[e] += weights_arr[i]
        return np.max(loads)

    # Faster approach: precompute for each edge which demands pass through it in CW and CCW
    # edge_cw_mask[d] = 1 if demand d's CW route uses edge d... no, let me think differently.
    
    # Better: for each edge e, precompute which demands use it in CW and in CCW
    # Then load[e] = sum of weights of demands where (routing_bit == 0 and cw uses e) or (routing_bit == 1 and ccw uses e)
    
    # Even better for exhaustive search: iterate over all 2^15 combos
    # But 32768 * 15 * 10 = ~5M ops, should be fine
    
    # Let's use a vectorized approach for the exhaustive search
    # Precompute: for each demand i and edge e, does CW route use edge e? (0/1)
    # Does CCW route use edge e? (0/1)
    
    cw_edge_use = np.zeros((D, N), dtype=np.float64)
    ccw_edge_use = np.zeros((D, N), dtype=np.float64)
    for i in range(D):
        for e in range(N):
            if cw_masks_arr[i] & (1 << e):
                cw_edge_use[i, e] = 1.0
            if ccw_masks_arr[i] & (1 << e):
                ccw_edge_use[i, e] = 1.0

    # For a given routing vector r (0/1), load[e] = sum_i w_i * (r_i * ccw_edge_use[i,e] + (1-r_i) * cw_edge_use[i,e])
    # = sum_i w_i * cw_edge_use[i,e] + sum_i w_i * r_i * (ccw_edge_use[i,e] - cw_edge_use[i,e])
    
    # Let base_load[e] = sum_i w_i * cw_edge_use[i,e]
    # Let delta_use[i,e] = ccw_edge_use[i,e] - cw_edge_use[i,e]  (is -1, 0, or 1)
    # load[e] = base_load[e] + sum_i r_i * w_i * delta_use[i,e]
    
    base_load = np.zeros(N, dtype=np.float64)
    delta_use = np.zeros((D, N), dtype=np.float64)
    for i in range(D):
        for e in range(N):
            base_load[e] += weights_arr[i] * cw_edge_use[i, e]
            delta_use[i, e] = ccw_edge_use[i, e] - cw_edge_use[i, e]

    # Now for routing r, load = base_load + r @ (weights[:,None] * delta_use)
    # Let W_delta[i,e] = w_i * delta_use[i,e]
    W_delta = weights_arr[:, None] * delta_use  # shape (D, N)

    # Exhaustive search over all 2^15 combinations
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    total_combos = 1 << D
    for combo in range(total_combos):
        # Extract routing bits
        r = np.array([(combo >> i) & 1 for i in range(D)], dtype=np.float64)
        loads = base_load + r @ W_delta
        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = r.copy()
        
        # Deadline check every 4096 combos
        if combo % 4096 == 0 and time.time() > DEADLINE:
            break

    # Local search refinement: flip one demand at a time
    improved = True
    while improved:
        improved = False
        for i in range(D):
            # Try flipping demand i
            r_new = best_routing.copy()
            r_new[i] = 1.0 - r_new[i]
            loads = base_load + r_new @ W_delta
            max_load = np.max(loads)
            if max_load < best_max_load - 1e-10:
                best_max_load = max_load
                best_routing = r_new
                improved = True
        if time.time() > DEADLINE:
            break

    # Return routing where 0=CW, 1=CCW; task says >=0.5 means CCW, <0.5 means CW
    # So return best_routing directly (0 or 1)
    return best_routing

# EVOLVE-BLOCK-END
