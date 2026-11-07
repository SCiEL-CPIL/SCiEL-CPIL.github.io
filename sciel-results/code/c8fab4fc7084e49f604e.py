<<<<<<< SEARCH
def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depth policy via backward induction on a discrete state grid.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depth for each (time, inventory)
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depth for each (time, inventory)
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_grid = np.arange(-Q, Q + 1)  # -Q, -Q+1, ..., Q
    
    # Depth grid: we search over possible depths
    # The depth must be in (0, 3/k) to be valid
    max_depth = 3.0 / k
    # Use a fine grid for depth search
    n_depth = 200
    depth_grid = np.linspace(0.01, max_depth - 0.01, n_depth)
    
    # Precompute fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k * d) * dt)
    p_fill_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Value function: V[n, i] where i indexes inventory from -Q to Q
    # We'll store V for current and previous step
    V_next = np.zeros(n_inv)  # Terminal: no charge
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_inv)
        
        for i in range(n_inv):
            q = inv_grid[i]
            
            # Inventory cost for this step
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            best_bid_val = -np.inf
            best_bid_depth = 0.0
            best_ask_val = -np.inf
            best_ask_depth = 0.0
            
            # If q == Q, no bid fill possible
            if q < Q:
                for di in range(n_depth):
                    d_bid = depth_grid[di]
                    p_bid = p_fill_grid[di]
                    
                    # If bid fills: inventory becomes q+1, earn d_bid
                    new_q_bid = q + 1
                    if new_q_bid <= Q:
                        idx_next = new_q_bid + Q  # index in inv_grid
                        if idx_next >= 0 and idx_next < n_inv:
                            val_bid = d_bid + (1 - p_bid) * V_next[i] + p_bid * V_next[idx_next]
                        else:
                            val_bid = d_bid + (1 - p_bid) * V_next[i]
                    else:
                        # Should not happen since we check q < Q
                        val_bid = (1 - p_bid) * V_next[i]
                    
                    if val_bid > best_bid_val:
                        best_bid_val = val_bid
                        best_bid_depth = d_bid
            
            # If q == -Q, no ask fill possible
            if q > -Q:
                for di in range(n_depth):
                    d_ask = depth_grid[di]
                    p_ask = p_fill_grid[di]
                    
                    # If ask fills: inventory becomes q-1, earn d_ask
                    new_q_ask = q - 1
                    if new_q_ask >= -Q:
                        idx_next = new_q_ask + Q
                        if idx_next >= 0 and idx_next < n_inv:
                            val_ask = d_ask + (1 - p_ask) * V_next[i] + p_ask * V_next[idx_next]
                        else:
                            val_ask = d_ask + (1 - p_ask) * V_next[i]
                    else:
                        val_ask = (1 - p_ask) * V_next[i]
                    
                    if val_ask > best_ask_val:
                        best_ask_val = val_ask
                        best_ask_depth = d_ask
            
            # The total value is the sum of independent bid and ask decisions
            # Since bid and ask fills are independent:
            # E[reward] = E[bid_reward] + E[ask_reward]
            # But we need to be careful: if both fill, inventory changes by 2?
            # Actually, the problem says "bid hit with probability ..., ask likewise, independently"
            # So both can fill in the same step.
            
            # Let's redo this properly with joint consideration.
            # Actually, for the value function:
            # V = -inv_cost + E[depth_earned + V_next[new_inventory]]
            # where depth_earned = d_bid * I(bid_fills) + d_ask * I(ask_fills)
            # and new_inventory = q + I(bid_fills) - I(ask_fills)
            
            # So we need to consider all 4 cases:
            # (no bid, no ask): prob (1-p_b)(1-p_a), reward 0, new_q = q
            # (bid only): prob p_b(1-p_a), reward d_b, new_q = q+1
            # (ask only): prob (1-p_b)p_a, reward d_a, new_q = q-1
            # (both): prob p_b*p_a, reward d_b+d_a, new_q = q
            
            # But wait, we choose d_b and d_a simultaneously.
            # This is a 2D optimization. Let's do it properly.
            
            best_total_val = -np.inf
            best_bid_depth = 0.0
            best_ask_depth = 0.0
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if can_bid and can_ask:
                # 2D search
                for di_b in range(n_depth):
                    d_b = depth_grid[di_b]
                    p_b = p_fill_grid[di_b]
                    for di_a in range(n_depth):
                        d_a = depth_grid[di_a]
                        p_a = p_fill_grid[di_a]
                        
                        # Case 1: no fills
                        val1 = V_next[i]
                        # Case 2: bid only
                        idx_b = (q + 1) + Q
                        val2 = d_b + V_next[idx_b] if 0 <= idx_b < n_inv else d_b
                        # Case 3: ask only
                        idx_a = (q - 1) + Q
                        val3 = d_a + V_next[idx_a] if 0 <= idx_a < n_inv else d_a
                        # Case 4: both
                        val4 = d_b + d_a + V_next[i]
                        
                        total = ((1 - p_b) * (1 - p_a) * val1 +
                                 p_b * (1 - p_a) * val2 +
                                 (1 - p_b) * p_a * val3 +
                                 p_b * p_a * val4)
                        
                        if total > best_total_val:
                            best_total_val = total
                            best_bid_depth = d_b
                            best_ask_depth = d_a
            elif can_bid:
                for di_b in range(n_depth):
                    d_b = depth_grid[di_b]
                    p_b = p_fill_grid[di_b]
                    idx_b = (q + 1) + Q
                    val_b = d_b + V_next[idx_b] if 0 <= idx_b < n_inv else d_b
                    total = (1 - p_b) * V_next[i] + p_b * val_b
                    if total > best_total_val:
                        best_total_val = total
                        best_bid_depth = d_b
                        best_ask_depth = 0.0
            elif can_ask:
                for di_a in range(n_depth):
                    d_a = depth_grid[di_a]
                    p_a = p_fill_grid[di_a]
                    idx_a = (q - 1) + Q
                    val_a = d_a + V_next[idx_a] if 0 <= idx_a < n_inv else d_a
                    total = (1 - p_a) * V_next[i] + p_a * val_a
                    if total > best_total_val:
                        best_total_val = total
                        best_bid_depth = 0.0
                        best_ask_depth = d_a
            else:
                best_total_val = V_next[i]
                best_bid_depth = 0.0
                best_ask_depth = 0.0
            
            V_curr[i] = best_total_val - inv_cost
            bid_table[n, i] = best_bid_depth
            ask_table[n, i] = best_ask_depth
        
        V_next = V_curr
    
    return bid_table, ask_table
=======
def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depth policy via vectorized backward induction.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_grid = np.arange(-Q, Q + 1)
    
    max_depth = 3.0 / k
    n_depth = 100  # Reduced for speed, still fine resolution
    depth_grid = np.linspace(0.01, max_depth - 0.01, n_depth)
    
    # Precompute fill probabilities and depths
    p_fill_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    V_next = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Precompute indices for next states
    # q is in [-Q, Q]. Index i = q + Q.
    # q+1 index: i+1, valid if i < n_inv-1
    # q-1 index: i-1, valid if i > 0
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_inv)
        
        # Vectorize over inventory states
        # We need to find max over d_b, d_a of the expected value.
        # Let's loop over depth grid indices for bid and ask.
        
        best_val = np.full(n_inv, -np.inf)
        best_idx_b = np.zeros(n_inv, dtype=int)
        best_idx_a = np.zeros(n_inv, dtype=int)
        
        # Masks for valid actions
        # can_bid[i] is True if inv_grid[i] < Q  => i < n_inv - 1
        # can_ask[i] is True if inv_grid[i] > -Q => i > 0
        can_bid = np.arange(n_inv) < n_inv - 1
        can_ask = np.arange(n_inv) > 0
        
        for di_b in range(n_depth):
            d_b = depth_grid[di_b]
            p_b = p_fill_grid[di_b]
            
            # If we choose this bid depth, what is the contribution?
            # We need to combine with ask depths.
            # To avoid nested loops over di_b and di_a inside the vectorized op,
            # we can iterate di_b and di_a and update the max arrays.
            
            for di_a in range(n_depth):
                d_a = depth_grid[di_a]
                p_a = p_fill_grid[di_a]
                
                # Calculate expected value for all i simultaneously
                # V_next[i], V_next[i+1], V_next[i-1]
                V_q = V_next
                V_q_plus = np.roll(V_next, -1) # q+1
                V_q_minus = np.roll(V_next, 1)  # q-1
                
                # Handle boundaries
                # If i = n_inv-1 (q=Q), V_q_plus is invalid (rolled from i=0)
                # If i = 0 (q=-Q), V_q_minus is invalid (rolled from i=n_inv-1)
                
                # Probabilities
                # P(no fill) = (1-p_b)(1-p_a)
                # P(bid only) = p_b(1-p_a)
                # P(ask only) = (1-p_b)p_a
                # P(both) = p_b p_a
                
                # Rewards
                # R(no) = 0
                # R(bid) = d_b
                # R(ask) = d_a
                # R(both) = d_b + d_a
                
                # Values
                # V(no) = V_q
                # V(bid) = V_q_plus (if valid)
                # V(ask) = V_q_minus (if valid)
                # V(both) = V_q
                
                # Construct expected value
                EV = np.zeros(n_inv)
                
                # Case 1: No fills
                EV += (1 - p_b) * (1 - p_a) * V_q
                
                # Case 2: Bid only
                # Valid only if can_bid
                val_bid = d_b + V_q_plus
                # Mask out invalid states (q=Q)
                val_bid = np.where(can_bid, val_bid, -np.inf)
                EV += p_b * (1 - p_a) * val_bid
                
                # Case 3: Ask only
                # Valid only if can_ask
                val_ask = d_a + V_q_minus
                val_ask = np.where(can_ask, val_ask, -np.inf)
                EV += (1 - p_b) * p_a * val_ask
                
                # Case 4: Both
                # Valid only if can_bid AND can_ask
                val_both = d_b + d_a + V_q
                val_both = np.where(can_bid & can_ask, val_both, -np.inf)
                EV += p_b * p_a * val_both
                
                # If a state cannot bid, the term p_b(1-p_a)val_bid should be 0, but we set val_bid to -inf.
                # This works if p_b(1-p_a) is 0? No, p_b is not 0.
                # We need to ensure that for states that cannot bid, we don't add this term.
                # Actually, if can_bid is False, we should not consider this action.
                # But we are iterating over ALL di_b. If can_bid is False for some i, we should ignore this di_b for that i.
                # The mask `np.where(can_bid, val_bid, -np.inf)` makes the contribution -inf * prob.
                # This is problematic if other terms are finite.
                
                # Better approach: Only update best_val for states where the action is valid.
                # Or, simpler: For states that cannot bid, the optimal bid depth is "none" (inf).
                # We can handle boundaries separately or mask the update.
                
                # Let's refine:
                # For each i, we want max over valid (di_b, di_a).
                # If i cannot bid, we should only consider di_b that represent "no quote"?
                # The problem says "np.inf means no quote".
                # In our grid, we don't have inf. We have depths.
                # If q=Q, we must quote inf (no bid).
                # If we don't quote, p_b = 0? No, p_b = 1 - exp(0) = 0.
                # So if we set d_b = inf, p_b = 0.
                # But our grid is finite.
                
                # Alternative:
                # For i where can_bid is False, the best bid is inf.
                # For i where can_ask is False, the best ask is inf.
                
                # Let's just compute EV for all i, but for invalid combinations, set EV to -inf.
                # Then take max.
                
                # However, if can_bid is False, we should NOT use this di_b.
                # So we should mask the entire EV for this di_b, di_a pair for those i's.
                
                valid_pair = can_bid & can_ask
                # If can_bid is false, we can't use this bid depth.
                # If can_ask is false, we can't use this ask depth.
                
                # Wait, if can_bid is false, we must use "no bid".
                # "No bid" corresponds to p_b=0.
                # We can simulate "no bid" by having a specific depth or handling it outside.
                
                # Let's handle boundaries by adding a virtual "no quote" option.
                # But the grid is fixed.
                
                # Simpler:
                # For each i, if can_bid[i] is False, we ignore all di_b except a special "no bid" index.
                # Let's add index n_depth for "no bid" (p=0, d=0).
                # And index n_depth for "no ask".
                
                # This complicates the grid.
                
                # Let's stick to the previous logic but vectorized.
                # The previous logic had separate branches for can_bid/can_ask.
                # We can replicate that by masking.
                
                # For states where can_bid & can_ask:
                #   Consider all di_b, di_a.
                # For states where can_bid & ~can_ask:
                #   Consider all di_b, di_a = "no ask" (p_a=0, d_a=0).
                # For states where ~can_bid & can_ask:
                #   Consider di_b = "no bid" (p_b=0, d_b=0), all di_a.
                # For states where ~can_bid & ~can_ask:
                #   Consider di_b="no bid", di_a="no ask".
                
                # We can do this by computing EV for the "full" case and then correcting for boundaries?
                # No, the max is non-linear.
                
                # Let's just do 3 separate vectorized passes for the 3 boundary types?
                # Type 1: Middle (can bid and ask)
                # Type 2: Right boundary (can bid, no ask) -> q=Q
                # Type 3: Left boundary (no bid, can ask) -> q=-Q
                # Type 4: Corners (no bid, no ask) -> q=-Q and Q? No, Q>0.
                # If Q>0, q=-Q and q=Q are distinct.
                # q=Q: can_bid=False, can_ask=True (if Q>0).
                # q=-Q: can_bid=True (if Q>0), can_ask=False.
                
                # So we have 3 cases:
                # 1. i in [1, n_inv-2]: can_bid, can_ask
                # 2. i == n_inv-1 (q=Q): no bid, can_ask
                # 3. i == 0 (q=-Q): can_bid, no ask
                
                # We can compute the max for the middle part, and handle boundaries separately.
                
                pass # This placeholder will be replaced by the actual logic in the next block
>>>>>>> REPLACE

The above replacement is a placeholder because the logic is complex. I will provide the full corrected `_compute_policy` function in a single edit.

