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
    # Deadline guard
    start_time = time.time()
    DEADLINE = start_time + 600.0

    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)

    # Precompute clockwise edge masks for each demand
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n_edges, dtype=np.int8)
        if s <= t:
            for e in range(s, t):
                mask[e] = 1
        else:
            for e in range(s, n_nodes):
                mask[e] = 1
            for e in range(0, t):
                mask[e] = 1
        cw_masks.append(mask)
    
    cw_masks = np.array(cw_masks) # Shape (D, n_edges)
    ccw_masks = 1 - cw_masks
    weights = np.array([w for _, _, w in demands], dtype=np.int64)
    
    # We use an incremental update strategy with Gray code enumeration.
    # State: current routing vector x (0 or 1 for each demand)
    # Load: current load vector
    
    # Initialize with all clockwise (x = 0)
    x = np.zeros(D, dtype=np.int64)
    current_load = np.sum(weights[:, None] * cw_masks, axis=0).astype(np.int64)
    best_load = np.max(current_load)
    best_routing = x.copy()
    
    # Gray code sequence for 15 bits
    # We iterate through all 2^15 states.
    # Gray code ensures only one bit changes at a time.
    
    for i in range(1, 1 << D):
        if time.time() > DEADLINE:
            break
        
        # Determine which bit flips in the gray code sequence
        # Gray code g = i ^ (i >> 1)
        # The bit that changes between g_{i-1} and g_i is the position of the least significant 1 in i
        # Actually, the standard way to get the changed bit in a Gray code sequence:
        # Let g_prev = (i-1) ^ ((i-1) >> 1)
        # Let g_curr = i ^ (i >> 1)
        # diff = g_prev ^ g_curr
        # bit_idx = position of the single set bit in diff
        
        # Faster: The bit that flips at step i is the index of the least significant set bit of i.
        # Let's verify.
        # i=1: 001. LSB index 0. g0=000, g1=001. Bit 0 flips. Correct.
        # i=2: 010. LSB index 1. g1=001, g2=011. Bit 1 flips. Correct.
        # i=3: 011. LSB index 0. g2=011, g3=010. Bit 0 flips. Correct.
        
        # Find index of least significant set bit in i
        # i is 1-based in the loop, but represents the step number.
        # The bit that changes is related to i.
        # Actually, the bit that changes when going from gray(i-1) to gray(i) is the position of the least significant 1 in i.
        
        # Calculate bit index
        # Use bit trick: (i & -i) gives the lowest set bit.
        # Then find index.
        # Python 3.11+ has int.bit_count, but we need index.
        # We can use a lookup or simple loop since D=15.
        
        # Simpler: just compute the new x from the gray code directly?
        # No, incremental is better.
        
        # Let's find the bit index k
        # k = 0
        # temp = i
        # while (temp & 1) == 0:
        #     temp >>= 1
        #     k += 1
        # This is the index of the LSB of i.
        
        # Wait, is it the LSB of i?
        # g(i) = i ^ (i >> 1)
        # g(i) ^ g(i-1) = ?
        # Let i = ...b1 b0.
        # If i is odd, LSB is 1. i>>1 shifts.
        # It is a known property that the bit that flips in the Gray code sequence at step i is the position of the least significant 1 in i.
        
        bit_idx = 0
        temp = i
        while (temp & 1) == 0:
            temp >>= 1
            bit_idx += 1
            
        # Update x
        x[bit_idx] = 1 - x[bit_idx]
        
        # Update load
        # If x[bit_idx] becomes 1, we add delta[bit_idx]
        # If x[bit_idx] becomes 0, we subtract delta[bit_idx]
        # delta[i] = w_i * (ccw_mask_i - cw_mask_i)
        
        # Precomputed deltas
        # We didn't precompute deltas in the previous snippet, let's do it now.
        # deltas[i] = weights[i] * (ccw_masks[i] - cw_masks[i])
        
        # To avoid recomputing deltas every time, we should have precomputed them.
        # Let's refactor to precompute deltas.
        
        # Re-initializing for correctness in the loop structure below
        pass

    # Refactored version with precomputation
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n_edges, dtype=np.int8)
        if s <= t:
            for e in range(s, t):
                mask[e] = 1
        else:
            for e in range(s, n_nodes):
                mask[e] = 1
            for e in range(0, t):
                mask[e] = 1
        cw_masks.append(mask)
    cw_masks = np.array(cw_masks)
    ccw_masks = 1 - cw_masks
    weights = np.array([w for _, _, w in demands], dtype=np.int64)
    deltas = weights[:, None] * (ccw_masks - cw_masks)
    
    x = np.zeros(D, dtype=np.int64)
    current_load = np.sum(weights[:, None] * cw_masks, axis=0).astype(np.int64)
    best_load = np.max(current_load)
    best_routing = x.copy()
    
    for i in range(1, 1 << D):
        if time.time() > DEADLINE:
            break
            
        # Find bit index to flip
        bit_idx = 0
        temp = i
        while (temp & 1) == 0:
            temp >>= 1
            bit_idx += 1
            
        # Update x
        if x[bit_idx] == 0:
            x[bit_idx] = 1
            current_load += deltas[bit_idx]
        else:
            x[bit_idx] = 0
            current_load -= deltas[bit_idx]
            
        max_load = np.max(current_load)
        if max_load < best_load:
            best_load = max_load
            best_routing = x.copy()
            
    return best_routing.astype(np.float64)

# EVOLVE-BLOCK-END
