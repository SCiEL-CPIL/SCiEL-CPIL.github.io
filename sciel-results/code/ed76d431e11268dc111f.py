def _precompute_policy(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    max_depth = 3.0 / k
    # Grid of depths
    G = 50 # Grid resolution
    depths = np.linspace(0, max_depth, G)
    
    # Precompute fill probabilities for grid depths
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # Vectorize over depths
    exp_term = np.exp(-k * depths)
    prob_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # V[n, q_index] where q_index maps to q in [-Q, Q]
    # q = -Q + index
    V = np.zeros((steps + 1, 2*Q + 1))
    # Policy tables
    best_bid = np.zeros((steps, 2*Q + 1))
    best_ask = np.zeros((steps, 2*Q + 1))
    
    # Iterate backwards in time
    for n in range(steps - 1, -1, -1):
        for q_idx in range(2*Q + 1):
            q = q_idx - Q
            
            # Get future values
            v_curr = V[n+1, q_idx]
            
            # Handle boundaries
            if q == Q:
                # No bid fill allowed? "no bid fill at q=Q"
                # Actually, if q=Q, we cannot increase inventory via ask?
                # "no bid fill at q=Q" -> bid fill reduces q. So bid fill is allowed?
                # "no ask fill at q=-Q" -> ask fill increases q.
                # Wait, standard convention:
                # Bid hit -> Buy asset -> Inventory increases?
                # Task says: "A fill earns its depth and moves inventory one unit"
                # Usually: Bid quote is price below mid. If hit, we buy. Inventory +1.
                # Ask quote is price above mid. If hit, we sell. Inventory -1.
                # Let's re-read carefully: "no bid fill at q = Q, no ask fill at q = -Q".
                # If q=Q (max long), we shouldn't buy more? Or sell more?
                # If q is inventory of asset.
                # Buy (Bid hit) -> q increases.
                # Sell (Ask hit) -> q decreases.
                # If q=Q, we are max long. We should not buy more. So "no bid fill" makes sense if Bid=Buy.
                # If q=-Q, we are max short. We should not sell more. So "no ask fill" makes sense if Ask=Sell.
                
                # Let's assume:
                # Bid depth d_b -> Prob p_b. If fill, q_new = q + 1. Reward d_b.
                # Ask depth d_a -> Prob p_a. If fill, q_new = q - 1. Reward d_a.
                
                # Constraint:
                # If q == Q, p_b must be 0 (or d_b=inf? No, just no fill).
                # If q == -Q, p_a must be 0.
                
                pass # Logic in loop below
            
            # We need to optimize d_b, d_a.
            # V_next(q+1), V_next(q-1), V_next(q)
            # If q == Q, q+1 is invalid. Transition to q+1 is impossible.
            # We can set V_next(q+1) to 0 or handle in prob mask.
            
            v_q_plus_1 = V[n+1, q_idx + 1] if q < Q else 0.0
            v_q_minus_1 = V[n+1, q_idx - 1] if q > -Q else 0.0
            
            # Create grids for d_b, d_a
            # d_b shape (G, 1), d_a shape (1, G)
            d_b_grid = depths[:, np.newaxis]
            d_a_grid = depths[np.newaxis, :]
            
            p_b_grid = prob_grid[:, np.newaxis] # (G, 1)
            p_a_grid = prob_grid[np.newaxis, :] # (1, G)
            
            # Broadcast to (G, G)
            p_b_full = np.broadcast_to(p_b_grid, (G, G))
            p_a_full = np.broadcast_to(p_a_grid, (G, G))
            
            # Mask invalid transitions
            if q == Q:
                p_b_full = np.zeros_like(p_b_full)
            if q == -Q:
                p_a_full = np.zeros_like(p_a_full)
            
            # Immediate Reward
            # Reward = d_b * p_b + d_a * p_a
            reward = d_b_grid * p_b_full + d_a_grid * p_a_full
            
            # Cost
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Future Value
            # E[V] = p_b p_a V(q) + p_b(1-p_a)V(q-1) + (1-p_b)p_a V(q+1) + (1-p_b)(1-p_a)V(q)
            # Note: My mapping:
            # Bid fill (Buy) -> q+1. Prob p_b.
            # Ask fill (Sell) -> q-1. Prob p_a.
            
            # Terms:
            # Both fill: p_b * p_a * v_curr  (Wait, if both fill, q goes to q? 
            # "independently". If both fill, net change 0? Or sequential?
            # Usually in market making, fills are independent events in the interval.
            # If both happen, inventory change is +1 - 1 = 0.
            # So state remains q.
            
            # Bid only: p_b * (1-p_a) * v_q_plus_1
            # Ask only: (1-p_b) * p_a * v_q_minus_1
            # Neither: (1-p_b) * (1-p_a) * v_curr
            
            exp_val = (p_b_full * p_a_full * v_curr + 
                       p_b_full * (1 - p_a_full) * v_q_plus_1 + 
                       (1 - p_b_full) * p_a_full * v_q_minus_1 + 
                       (1 - p_b_full) * (1 - p_a_full) * v_curr)
            
            total_val = reward - cost + exp_val
            
            # Find max
            max_idx = np.argmax(total_val)
            i, j = np.unravel_index(max_idx, (G, G))
            
            V[n, q_idx] = total_val[max_idx]
            best_bid[n, q_idx] = depths[i]
            best_ask[n, q_idx] = depths[j]
            
    return best_bid, best_ask
