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
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Deadline guard: 630s limit, use 600s for safety.
    # Since 2^15 is small, this is unlikely to be hit, but required by directive.
    start_time = time.time()
    deadline = start_time + 600.0

    # Precompute masks.
    # Edge e is between node e and (e+1)%n.
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n).
    # Counter-clockwise: all other edges.
    cw_mask = np.zeros((D, n), dtype=np.float64)
    ccw_mask = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D)
    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1.0
            e = (e + 1) % n
        ccw_mask[d, :] = 1.0 - cw_mask[d, :]

    # Weighted contribution of each demand routed CW or CCW.
    cw_w = cw_mask * weights[:, None]
    ccw_w = ccw_mask * weights[:, None]

    # base_load: load if every demand routed CW.
    base_load = cw_w.sum(axis=0)  # shape (n,)

    # diff[d, e] = ccw_w[d, e] - cw_w[d, e].
    # load(r) = base_load + r @ diff, where r[d] in {0,1} is the direction.
    diff = ccw_w - cw_w  # shape (D, n)

    # ---- Fast exhaustive search via Gray-code single-bit updates ----
    total = 1 << D  # 32768

    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)

    # Start at routing = 0 (all CW).
    load = base_load.copy()  # shape (n,)
    r = np.zeros(D, dtype=np.float64)
    cur_max = load.max()
    if cur_max < best_max_load:
        best_max_load = cur_max
        best_routing = r.copy()

    for i in range(1, total):
        # Check deadline
        if time.time() > deadline:
            break
            
        bit = (i & (-i)).bit_length() - 1  # index of flipped bit
        if r[bit] == 0.0:
            r[bit] = 1.0
            load += diff[bit]
        else:
            r[bit] = 0.0
            load -= diff[bit]

        cur_max = load.max()
        if cur_max < best_max_load:
            best_max_load = cur_max
            best_routing = r.copy()

    return best_routing.astype(float)


# EVOLVE-BLOCK-END
