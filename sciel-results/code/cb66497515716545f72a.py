"""Optimal market making with inventory risk using pre-computed backward-induction policy."""
import numpy as np
import time

# Module-level cache for policy tables
_POLICY_CACHE = {}
_START_TIME = time.time()
_TIME_BUDGET = 320  # seconds, safety margin before 330s kill

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth for each (step, inventory) via backward induction.
    
    The state is (n, q) where n is the step index and q is inventory in [-Q, Q].
    At each state, we choose bid_depth b >= 0 and ask_depth a >= 0 to maximize
    expected objective.
    
    The objective per step (before fills) is:
      - (gamma/2) * sigma^2 * q^2 * dt  (inventory holding cost)
    Plus the expected value of next state after fills.
    
    A bid fill: earns b, inventory increases by 1 (q -> q+1), probability p_b = 1-exp(-A*exp(-k*b)*dt)
    An ask fill: earns a, inventory decreases by 1 (q -> q-1), probability p_a = 1-exp(-A*exp(-k*a)*dt)
    No bid fill at q=Q, no ask fill at q=-Q.
    
    We discretize depth on a grid and do backward induction.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_offset = Q  # q + Q gives array index
    
    # Depth grid: we need to search over bid and ask depths.
    # The optimal depth is typically in [0, ~5/k] range.
    # Use a fine grid for depth search.
    max_depth = 10.0 / k
    n_depth = 400  # number of depth grid points - finer for better resolution
    depth_grid = np.linspace(0, max_depth, n_depth)
    
    # Precompute fill probabilities for each depth
    # p_fill[depth_idx] = 1 - exp(-A * exp(-k * depth_grid[depth_idx]) * dt)
    exp_term = A * np.exp(-k * depth_grid) * dt
    p_fill = 1.0 - np.exp(-exp_term)
    
    # Value function: V[n, q_idx] = optimal expected cumulative reward from step n onwards
    # V[steps, q] = 0 for all q (no terminal charge)
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    
    # Policy tables: optimal depth for bid and ask (store actual float values)
    bid_policy = np.zeros((steps, n_inv), dtype=np.float64)
    ask_policy = np.zeros((steps, n_inv), dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET:
            break
        
        # Inventory holding cost for this step
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
        hold_cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Determine valid actions
            # Bid: if q < Q, we can quote bid (bid fill increases inventory)
            # Ask: if q > -Q, we can quote ask (ask fill decreases inventory)
            
            # We need to choose (bid_depth_idx, ask_depth_idx) to maximize:
            # - hold_cost[q_idx]
            # + p_b * (b_depth + V[n+1, q+1]) + (1-p_b) * V[n+1, q]   [bid side]
            # + p_a * (a_depth + V[n+1, q-1]) + (1-p_a) * V[n+1, q]   [ask side]
            # But bid and ask fills are independent, so we have 4 cases:
            #   no bid, no ask: prob (1-p_b)(1-p_a), reward 0, next state q
            #   bid only: prob p_b(1-p_a), reward b, next state q+1
            #   ask only: prob (1-p_b)p_a, reward a, next state q-1
            #   both: prob p_b*p_a, reward b+a, next state q (bid+1, ask-1 cancel)
            
            # For efficiency, precompute the next-state values
            V_next_q = V[n+1, q_idx]
            
            if q < Q:
                V_next_q_plus = V[n+1, q_idx + 1]
            else:
                V_next_q_plus = -np.inf  # invalid, won't be used
            
            if q > -Q:
                V_next_q_minus = V[n+1, q_idx - 1]
            else:
                V_next_q_minus = -np.inf  # invalid, won't be used
            
            best_val = -np.inf
            best_bid_depth = 0.0
            best_ask_depth = 0.0
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # Search over depth grid
            # For speed, we can vectorize over depth indices
            # But the joint search is 2D. Let's do it with numpy.
            
            if can_bid and can_ask:
                # Both sides active
                # b_depths = depth_grid, a_depths = depth_grid
                # For each (bi, ai):
                #   p_b = p_fill[bi], p_a = p_fill[ai]
                #   b = depth_grid[bi], a = depth_grid[ai]
                #   val = -hold_cost[q_idx]
                #         + (1-p_b)(1-p_a) * V_next_q
                #         + p_b(1-p_a) * (b + V_next_q_plus)
                #         + (1-p_b)p_a * (a + V_next_q_minus)
                #         + p_b*p_a * (b + a + V_next_q)
                
                # Vectorize over bi and ai
                p_b = p_fill  # shape (n_depth,)
                p_a = p_fill  # shape (n_depth,)
                b_vals = depth_grid  # shape (n_depth,)
                a_vals = depth_grid  # shape (n_depth,)
                
                # Create 2D arrays
                P_b = p_b[:, None]  # (n_depth, 1)
                P_a = p_a[None, :]  # (1, n_depth)
                B = b_vals[:, None]  # (n_depth, 1)
                A_ = a_vals[None, :]  # (1, n_depth)
                
                # Compute expected value for each (bi, ai) pair
                # term1: no fills
                term1 = (1 - P_b) * (1 - P_a) * V_next_q
                # term2: bid only
                term2 = P_b * (1 - P_a) * (B + V_next_q_plus)
                # term3: ask only
                term3 = (1 - P_b) * P_a * (A_ + V_next_q_minus)
                # term4: both
                term4 = P_b * P_a * (B + A_ + V_next_q)
                
                total = -hold_cost[q_idx] + term1 + term2 + term3 + term4
                
                flat_idx = np.argmax(total)
                best_bid_idx = flat_idx // n_depth
                best_ask_idx = flat_idx % n_depth
                best_val = total.flat[flat_idx]
                best_bid_depth = depth_grid[best_bid_idx]
                best_ask_depth = depth_grid[best_ask_idx]
                
                # Local refinement: try nearby grid points for finer accuracy
                step_b = depth_grid[1] - depth_grid[0] if n_depth > 1 else 0.01
                # Use 5x5 local grid centered on best
                b_local = best_bid_depth + np.linspace(-2*step_b, 2*step_b, 5)
                a_local = best_ask_depth + np.linspace(-2*step_b, 2*step_b, 5)
                b_local = np.clip(b_local, 0, max_depth)
                a_local = np.clip(a_local, 0, max_depth)
                
                p_b_loc = 1.0 - np.exp(-A * np.exp(-k * b_local) * dt)
                p_a_loc = 1.0 - np.exp(-A * np.exp(-k * a_local) * dt)
                
                Pb_l = p_b_loc[:, None]
                Pa_l = p_a_loc[None, :]
                B_l = b_local[:, None]
                A_l = a_local[None, :]
                
                t1 = (1 - Pb_l) * (1 - Pa_l) * V_next_q
                t2 = Pb_l * (1 - Pa_l) * (B_l + V_next_q_plus)
                t3 = (1 - Pb_l) * Pa_l * (A_l + V_next_q_minus)
                t4 = Pb_l * Pa_l * (B_l + A_l + V_next_q)
                total_ref = -hold_cost[q_idx] + t1 + t2 + t3 + t4
                
                ref_flat = np.argmax(total_ref)
                if total_ref.flat[ref_flat] > best_val:
                    best_val = total_ref.flat[ref_flat]
                    best_bid_depth = b_local[ref_flat // 5]
                    best_ask_depth = a_local[ref_flat % 5]
                
            elif can_bid:
                # Only bid side active
                p_b = p_fill
                b_vals = depth_grid
                
                term1 = (1 - p_b) * V_next_q
                term2 = p_b * (b_vals + V_next_q_plus)
                total = -hold_cost[q_idx] + term1 + term2
                
                best_bid_idx = np.argmax(total)
                best_val = total[best_bid_idx]
                best_bid_depth = depth_grid[best_bid_idx]
                best_ask_depth = 0.0
                
                # Local refinement
                step_b = depth_grid[1] - depth_grid[0] if n_depth > 1 else 0.01
                b_local = best_bid_depth + np.linspace(-2*step_b, 2*step_b, 5)
                b_local = np.clip(b_local, 0, max_depth)
                p_b_loc = 1.0 - np.exp(-A * np.exp(-k * b_local) * dt)
                t1 = (1 - p_b_loc) * V_next_q
                t2 = p_b_loc * (b_local + V_next_q_plus)
                total_ref = -hold_cost[q_idx] + t1 + t2
                ref_idx = np.argmax(total_ref)
                if total_ref[ref_idx] > best_val:
                    best_val = total_ref[ref_idx]
                    best_bid_depth = b_local[ref_idx]
                    
            elif can_ask:
                # Only ask side active
                p_a = p_fill
                a_vals = depth_grid
                
                term1 = (1 - p_a) * V_next_q
                term2 = p_a * (a_vals + V_next_q_minus)
                total = -hold_cost[q_idx] + term1 + term2
                
                best_ask_idx = np.argmax(total)
                best_val = total[best_ask_idx]
                best_ask_depth = depth_grid[best_ask_idx]
                best_bid_depth = 0.0
                
                # Local refinement
                step_a = depth_grid[1] - depth_grid[0] if n_depth > 1 else 0.01
                a_local = best_ask_depth + np.linspace(-2*step_a, 2*step_a, 5)
                a_local = np.clip(a_local, 0, max_depth)
                p_a_loc = 1.0 - np.exp(-A * np.exp(-k * a_local) * dt)
                t1 = (1 - p_a_loc) * V_next_q
                t2 = p_a_loc * (a_local + V_next_q_minus)
                total_ref = -hold_cost[q_idx] + t1 + t2
                ref_idx = np.argmax(total_ref)
                if total_ref[ref_idx] > best_val:
                    best_val = total_ref[ref_idx]
                    best_ask_depth = a_local[ref_idx]
                    
            else:
                # Neither side active
                best_val = -hold_cost[q_idx] + V_next_q
                best_bid_depth = 0.0
                best_ask_depth = 0.0
            
            V[n, q_idx] = best_val
            bid_policy[n, q_idx] = best_bid_depth
            ask_policy[n, q_idx] = best_ask_depth
    
    # Policy tables already store actual float depths
    return bid_policy, ask_policy


def _get_policy_table(params):
    """Get or compute the policy table for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    Each element is the distance below/above mid to quote.
    np.inf means no quote.
    """
    # Determine step index from t
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(max(n, 0), params['steps'] - 1)
    
    Q = params['Q']
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index
    inv_idx = inventory + Q  # inventory in [-Q, Q], so inv_idx in [0, 2Q]
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths (already float values)
    bid_depths = bid_table[n, inv_idx].astype(np.float64)
    ask_depths = ask_table[n, inv_idx].astype(np.float64)
    
    # At q=Q, no bid fill possible -> set bid to inf
    # At q=-Q, no ask fill possible -> set ask to inf
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure values are valid: >= 0, not NaN, not -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), np.inf, ask_depths)
    
    # Ensure shape is (4000,)
    if len(bid_depths) != 4000:
        # Pad or truncate
        if len(bid_depths) < 4000:
            pad = np.full(4000 - len(bid_depths), np.inf)
            bid_depths = np.concatenate([bid_depths, pad])
            ask_depths = np.concatenate([ask_depths, pad])
        else:
            bid_depths = bid_depths[:4000]
            ask_depths = ask_depths[:4000]
    
    return bid_depths, ask_depths
