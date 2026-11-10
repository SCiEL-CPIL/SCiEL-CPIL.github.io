"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s limit

# Global cache for DP tables
_dp_cache = {}

def _compute_dp_table(params):
    """Compute optimal policy table via backward induction on depth grid."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    
    # Depth grid: from min_depth to some max_depth
    min_depth = -3.0 / k
    max_depth = 5.0 / k  # reasonable upper bound
    n_depths = 200
    depths = np.linspace(min_depth, max_depth, n_depths)
    
    # Inventory range
    q_min = -Q
    q_max = Q
    n_q = Q - q_min + 1
    
    # Fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Value function: V[step, q] for step 0..steps (step 0 is terminal)
    # V[0, q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_q))
    
    # Policy tables: best_bid_depth[step, q], best_ask_depth[step, q]
    best_bid = np.zeros((steps, n_q))
    best_ask = np.zeros((steps, n_q))
    
    # Backward induction
    for n in range(1, steps + 1):
        t = (n - 1) * dt
        t_rem = T - t
        
        # Inventory cost for this step
        inv_cost = (gamma / 2.0) * (sigma ** 2) * dt
        q_vals = np.arange(q_min, q_max + 1, dtype=np.float64)
        cost = inv_cost * q_vals ** 2
        
        for i_q in range(n_q):
            q = q_min + i_q
            
            # Try all bid and ask depth combinations
            # For efficiency, we can vectorize over depths
            # But first handle boundary conditions
            
            # No bid fill at q = Q
            if q >= q_max:
                # Must set bid to inf (no quote)
                # Actually, we can still quote but it won't fill
                # The constraint is "no bid fill at q=Q", so we can quote but p_fill effectively 0
                # But the model says "no bid fill at q=Q", meaning the fill doesn't happen
                # So we can quote any depth, but the fill probability is 0
                # To be safe, let's treat it as: if q >= q_max, bid fill prob = 0
                bid_fill_probs = np.zeros(n_depths)
            else:
                bid_fill_probs = fill_probs
            
            # No ask fill at q = -Q
            if q <= q_min:
                ask_fill_probs = np.zeros(n_depths)
            else:
                ask_fill_probs = fill_probs
            
            # For each (bid_depth_i, ask_depth_j) combination:
            # Revenue = bid_depth_i * I_bid_fill + ask_depth_j * I_ask_fill
            # Next inventory: q + I_bid_fill - I_ask_fill
            # Must be in [q_min, q_max]
            
            # Vectorized computation
            # bid_depths[i], ask_depths[j]
            # p_bid = bid_fill_probs[i], p_ask = ask_fill_probs[j]
            
            # To save time, use a coarser search or vectorize
            # Let's vectorize over bid depths for each ask depth
            
            best_val = -np.inf
            best_bi = 0
            best_ai = 0
            
            # Sample a subset of depth pairs for speed
            # Use all bid depths, sample ask depths
            ask_sample = np.linspace(0, n_depths - 1, min(50, n_depths), dtype=int)
            
            for ai in ask_sample:
                ask_d = depths[ai]
                p_ask = ask_fill_probs[ai]
                
                # For each bid depth
                bid_d = depths
                p_bid = bid_fill_probs
                
                # Four cases:
                # 1. No bid fill, no ask fill: prob (1-p_bid)(1-p_ask), reward 0, next_q = q
                # 2. Bid fill, no ask fill: prob p_bid(1-p_ask), reward bid_d, next_q = q+1
                # 3. No bid fill, ask fill: prob (1-p_bid)p_ask, reward ask_d, next_q = q-1
                # 4. Both fill: prob p_bid*p_ask, reward bid_d+ask_d, next_q = q
                
                # Check next_q bounds
                q_plus = q + 1
                q_minus = q - 1
                
                valid_plus = (q_plus >= q_min) & (q_plus <= q_max)
                valid_minus = (q_minus >= q_min) & (q_minus <= q_max)
                
                # If q+1 out of bounds, case 2 is invalid (but model says no fill at boundary)
                # Actually the model handles this: no bid fill at q=Q, so q+1 can't happen from q=Q
                # But we already set p_bid=0 at boundary, so it's fine
                
                # Expected value from next step
                V_next_q = V[n-1, i_q]
                V_next_q_plus = V[n-1, i_q + 1] if valid_plus else -np.inf
                V_next_q_minus = V[n-1, i_q - 1] if valid_minus else -np.inf
                
                # If next state is out of bounds, that transition is impossible
                # But with our boundary handling, p_bid=0 at q_max and p_ask=0 at q_min
                # So those transitions have 0 probability
                
                # Expected reward + continuation
                # Case 1: (1-p_bid)(1-p_ask) * V_next_q
                # Case 2: p_bid(1-p_ask) * (bid_d + V_next_q_plus)
                # Case 3: (1-p_bid)p_ask * (ask_d + V_next_q_minus)
                # Case 4: p_bid*p_ask * (bid_d + ask_d + V_next_q)
                
                # Vectorized over bid depths
                term1 = (1 - p_bid) * (1 - p_ask) * V_next_q
                term2 = p_bid * (1 - p_ask) * (bid_d + V_next_q_plus)
                term3 = (1 - p_bid) * p_ask * (ask_d + V_next_q_minus)
                term4 = p_bid * p_ask * (bid_d + ask_d + V_next_q)
                
                total = term1 + term2 + term3 + term4
                total = total - cost[i_q]
                
                # Handle -inf cases
                if not np.isfinite(V_next_q_plus):
                    term2 = np.where(p_bid > 0, -np.inf, 0)
                    total = term1 + term2 + term3 + term4 - cost[i_q]
                if not np.isfinite(V_next_q_minus):
                    term3 = np.where(p_ask > 0, -np.inf, 0)
                    total = term1 + term2 + term3 + term4 - cost[i_q]
                
                idx = np.argmax(total)
                if total[idx] > best_val:
                    best_val = total[idx]
                    best_bi = idx
                    best_ai = ai
            
            V[n, i_q] = best_val
            best_bid[n-1, i_q] = depths[best_bi]
            best_ask[n-1, i_q] = depths[best_ai]
        
        # Deadline check
        if time.time() - _start_time > _DEADLINE - 5:
            break
    
    # Store in cache
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    _dp_cache[cache_key] = (V, best_bid, best_ask, depths, q_min)
    
    return _dp_cache[cache_key]


def quote(t, inventory, mid, params):
    """
    Optimal market making quote using discrete dynamic programming.
    
    Solves the optimal control problem via backward induction on a depth grid,
    then looks up the optimal bid/ask depths for the current inventory.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        k = params['k']
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.int64)
    
    # Get or compute DP table
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _dp_cache:
        # Try to compute, but with time limit
        try:
            _compute_dp_table(params)
        except Exception:
            pass
    
    if cache_key in _dp_cache:
        V, best_bid, best_ask, depths, q_min = _dp_cache[cache_key]
        
        # Determine current step
        dt = T / steps
        step_idx = int(round(t / dt))
        step_idx = np.clip(step_idx, 0, steps - 1)
        
        # Look up optimal depths
        q_clipped = np.clip(inv, -Q, Q)
        q_idx = q_clipped - q_min
        
        bid_depths = best_bid[step_idx, q_idx]
        ask_depths = best_ask[step_idx, q_idx]
        
        # Enforce constraints
        min_depth = -3.0 / k
        bid_depths = np.maximum(bid_depths, min_depth)
        ask_depths = np.maximum(ask_depths, min_depth)
        
        return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
    else:
        # Fallback: simple linear skew
        t_rem = max(0.0, T - t)
        base_depth = 1.0 / k
        skew_factor = gamma * (sigma ** 2) * t_rem
        inv_f = inv.astype(np.float64)
        bid_depths = base_depth + skew_factor * inv_f
        ask_depths = base_depth - skew_factor * inv_f
        min_depth = -3.0 / k
        bid_depths = np.maximum(bid_depths, min_depth)
        ask_depths = np.maximum(ask_depths, min_depth)
        return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
# EVOLVE-BLOCK-END
