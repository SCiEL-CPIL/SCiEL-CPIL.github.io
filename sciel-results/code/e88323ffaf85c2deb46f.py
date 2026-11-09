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
    cw_mask = np.zeros((D, n), dtype=np.int32)
    ccw_mask = np.zeros((D, n), dtype=np.int32)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
        # Counter-clockwise uses all other edges
        for e in range(n):
            if cw_mask[d, e] == 0:
                ccw_mask[d, e] = 1

    # Total weight on all demands
    total_weight = weights.sum()

    # For each demand, if routed CW it adds weight to cw edges; if CCW to ccw edges.
    # We can think of it as: for each demand, there's a "delta" vector that represents
    # the difference in load contribution between CCW and CW.
    # delta[d, e] = ccw_mask[d, e] * w - cw_mask[d, e] * w
    # If all CW: load = sum_d w_d * cw_mask[d, :]
    # If demand d is flipped to CCW: load changes by delta[d, :]
    
    delta = (ccw_mask - cw_mask) * weights[:, np.newaxis]

    # Base load when all demands are CW
    base_load = (weights[:, np.newaxis] * cw_mask).sum(axis=0)

    start_time = time.time()
    deadline = 620  # safety margin before 630s kill

    def compute_max_load_from_flips(flips):
        """
        flips: boolean array of shape (D,), True means demand is CCW
        Returns the max edge load.
        """
        load = base_load + delta[flips].sum(axis=0)
        return load.max()

    def compute_all_loads_from_flips(flips):
        """
        flips: boolean array of shape (D,)
        Returns the full load vector.
        """
        return base_load + delta[flips].sum(axis=0)

    # Brute force is 2^15 = 32768 possibilities. Let's try to do it smartly.
    # Actually 2^15 is small enough to brute force directly!
    # Let's enumerate all 2^15 assignments.
    
    best_routing = np.zeros(D, dtype=np.int8)
    best_max_load = float('inf')

    # Since 2^15 = 32768, we can brute force all combinations
    # Use bit representation
    for mask in range(1 << D):
        flips = np.zeros(D, dtype=bool)
        for d in range(D):
            if mask & (1 << d):
                flips[d] = True
        
        max_load = compute_max_load_from_flips(flips)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = flips.astype(np.int8)
        
        # Check time
        if (mask + 1) % 4096 == 0:
            if time.time() - start_time > deadline:
                break

    # The brute force should find the optimal solution since 2^15 is small
    # Convert: 0 = clockwise, 1 = counter-clockwise
    # Return as float array where >= 0.5 means CCW
    result = best_routing.astype(float)
    
    return result


# EVOLVE-BLOCK-END
