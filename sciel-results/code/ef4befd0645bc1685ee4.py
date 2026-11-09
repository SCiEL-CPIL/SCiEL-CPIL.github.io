# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

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

    # We model the routing direction x_d in [0, 1].
    # x_d = 0 means CW, x_d = 1 means CCW.
    # Load on edge e: L_e(x) = sum_d w_d * ( (1-x_d)*cw_mask[d,e] + x_d*ccw_mask[d,e] )
    #                 = sum_d w_d * ( cw_mask[d,e] + x_d*(ccw_mask[d,e] - cw_mask[d,e]) )
    
    # Let A[d, e] = cw_mask[d, e]
    # Let B[d, e] = ccw_mask[d, e] - cw_mask[d, e]
    # L_e(x) = sum_d w_d * A[d, e] + sum_d w_d * B[d, e] * x_d
    # Let base_load[e] = sum_d w_d * A[d, e]
    # Let Delta[e] = sum_d w_d * B[d, e] * x_d
    
    base_load = np.zeros(n, dtype=np.float64)
    Delta_matrix = np.zeros((n, D), dtype=np.float64) # Delta_matrix[e, d] = w_d * B[d, e]

    for d in range(D):
        for e in range(n):
            base_load[e] += weights[d] * cw_mask[d, e]
            Delta_matrix[e, d] = weights[d] * (ccw_mask[d, e] - cw_mask[d, e])

    def soft_max_load(x, alpha=10.0):
        """
        Compute the smooth approximation of the max load.
        x: array of shape (D,)
        """
        # L = base_load + Delta_matrix @ x
        loads = base_load + Delta_matrix @ x
        # Log-sum-exp trick for numerical stability
        # max(loads) + log(sum(exp(loads - max(loads))))
        m = np.max(loads)
        if m > 500: # prevent overflow
            return m
        return m + np.log(np.sum(np.exp(alpha * (loads - m)))) / alpha

    def soft_max_load_grad(x, alpha=10.0):
        """
        Gradient of the smooth approximation.
        """
        loads = base_load + Delta_matrix @ x
        m = np.max(loads)
        if m > 500:
            # If overflow, just return gradient of the max element? 
            # Actually if m is large, exp(alpha*(loads-m)) is exp(0)=1 for max, 0 for others.
            # So it's just the max element. Gradient is Delta_matrix[argmax, :]
            idx = np.argmax(loads)
            return Delta_matrix[idx, :] * alpha / alpha # Wait, derivative of (1/alpha)*log(sum(exp(alpha*L)))
            # d/dx (1/alpha) log(sum exp(alpha L)) = (1/alpha) * (sum exp(alpha L) * alpha * dL/dx) / sum exp(alpha L)
            # = sum (exp(alpha L) / sum exp(alpha L)) * dL/dx
            # = p_e * Delta_matrix[e, :]
            # If m is very large, p is a one-hot vector at argmax.
            return Delta_matrix[idx, :]
        
        exp_loads = np.exp(alpha * (loads - m))
        probs = exp_loads / np.sum(exp_loads)
        # Gradient is probs @ Delta_matrix
        return probs @ Delta_matrix

    start_time = time.time()
    deadline = 600 # Safety margin

    # Exhaustive search over all 2^15 = 32768 possible routings
    # This guarantees finding the global optimum
    best_x = None
    best_max_load = float('inf')
    
    # Precompute weight matrix for faster computation
    # weight_matrix[d, e] = weight of demand d on edge e if routed CW
    weight_cw = np.zeros((D, n), dtype=np.float64)
    weight_ccw = np.zeros((D, n), dtype=np.float64)
    
    for d, (s, t, w) in enumerate(demands):
        if s == t:
            continue
        e = s
        while e != t:
            weight_cw[d, e] = w
            e = (e + 1) % n
        for e in range(n):
            if weight_cw[d, e] == 0:
                weight_ccw[d, e] = w
    
    # Precompute constant matrices outside the loop
    diff = weight_ccw - weight_cw  # (D, n)
    base = np.sum(weight_cw, axis=0)  # (n,)
    
    # Vectorized exhaustive search
    # For each routing assignment, compute loads and track best
    for bits in itertools.product([0, 1], repeat=D):
        if time.time() - start_time > deadline:
            break
        
        # Compute loads: for each edge, sum of weights of demands using that edge
        # If bit[d] == 0: use CW, if bit[d] == 1: use CCW
        # Load on edge e = sum_d (bit[d] * weight_ccw[d,e] + (1-bit[d]) * weight_cw[d,e])
        #                 = sum_d weight_cw[d,e] + sum_d bit[d] * (weight_ccw[d,e] - weight_cw[d,e])
        
        bits_arr = np.array(bits, dtype=np.float64)
        loads = base + bits_arr @ diff  # (n,)
        
        max_load = np.max(loads)
        if max_load < best_max_load - 1e-9:
            best_max_load = max_load
            best_x = bits_arr
    
    # Fallback if no valid solution found (shouldn't happen)
    if best_x is None:
        best_x = np.zeros(D, dtype=np.float64)
    
    return best_x

# EVOLVE-BLOCK-END
