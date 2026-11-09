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
        value >= 0.5 -> route the demand counter-clockwise
        value < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_mask[d, e] = 1 if edge e is used when routing demand d clockwise
    #   ccw_mask[d, e] = 1 if edge e is used when routing demand d counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.float64)
    ccw_mask = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1.0
            e = (e + 1) % n
        # Counter-clockwise uses all other edges
        ccw_mask[d, :] = 1.0 - cw_mask[d, :]

    # Total weight on all demands
    # Base load: all demands routed clockwise
    base_load = (weights[:, np.newaxis] * cw_mask).sum(axis=0)
    
    # Difference vector: change in load if demand d is routed CCW instead of CW
    # diff[d, e] = w_d * (ccw_mask[d, e] - cw_mask[d, e])
    diff = weights[:, np.newaxis] * (ccw_mask - cw_mask)

    best_max_load = float('inf')
    best_mask = 0

    # Exhaustive search using Gray code for O(n) update per step
    # We iterate through all 2^D masks.
    # Gray code g(i) = i ^ (i >> 1).
    # The bit that changes between g(i) and g(i+1) is the index of the least significant 0 in i (which becomes 1 in g(i+1) relative to g(i) logic? 
    # Actually, the bit that flips between g(i) and g(i+1) is the position of the least significant 1 in (i+1).
    # Let's just track the current mask and the previous gray code.
    
    current_mask = 0
    current_load = base_load.copy()
    
    # Initialize best with mask 0
    max_load_0 = current_load.max()
    best_max_load = max_load_0
    best_mask = 0

    # Iterate through all other masks
    for i in range(1, 1 << D):
        # Determine which bit flips from g(i-1) to g(i)
        # g(i) = i ^ (i >> 1)
        # g(i-1) = (i-1) ^ ((i-1) >> 1)
        # The differing bit is the index of the least significant 1 in i.
        # Let's verify:
        # i=1 (001): g(1)=001. g(0)=000. Bit 0 flips. LSB 1 in 1 is at pos 0. Correct.
        # i=2 (010): g(2)=011. g(1)=001. Bit 1 flips. LSB 1 in 2 is at pos 1. Correct.
        # i=3 (011): g(3)=010. g(2)=011. Bit 0 flips. LSB 1 in 3 is at pos 0. Correct.
        
        # Find index of least significant set bit in i
        # (i & -i) gives the lowest set bit.
        # We need the index.
        # Using bit_length - 1 on the lowest set bit.
        lowest_bit = i & -i
        bit_idx = lowest_bit.bit_length() - 1
        
        # Determine if we are adding or subtracting diff[bit_idx]
        # Check if bit bit_idx is set in current_mask (which is g(i-1))
        if (current_mask >> bit_idx) & 1:
            # Bit was 1, now becomes 0. We are removing the CCW contribution (or adding back CW? No.)
            # Our formula: Load = Base + sum(r_d * diff[d])
            # If r_d goes from 1 to 0, we subtract diff[d].
            current_load -= diff[bit_idx]
            current_mask ^= (1 << bit_idx)
        else:
            # Bit was 0, now becomes 1. We are adding the CCW contribution.
            current_load += diff[bit_idx]
            current_mask ^= (1 << bit_idx)
        
        # Check max load
        max_load = current_load.max()
        if max_load < best_max_load:
            best_max_load = max_load
            best_mask = current_mask

    # Construct the result array
    # best_mask bit d is 1 -> CCW (1.0), 0 -> CW (0.0)
    routing = np.zeros(D, dtype=np.float64)
    for d in range(D):
        if (best_mask >> d) & 1:
            routing[d] = 1.0
        else:
            routing[d] = 0.0

    return routing

# EVOLVE-BLOCK-END
