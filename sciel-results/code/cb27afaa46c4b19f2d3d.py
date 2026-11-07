# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # safety margin before 330s kill

_OPT_TABLES = {}


def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    # Depth grid: use a fine grid of possible depths
    # Depths must be > 0 and < 3/k (per validity constraint: values below -3/k are invalid,
    # but depths are distances so they should be positive)
    # Actually re-reading: "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid"
    # This means the returned values (bid_depths, ask_depths) must not be below -3/k.
    # Since these are "distances below and above mid", they should be positive.
    # Let's use a grid from very small to 3/k.
    
    n_depths = 200
    # Use log-spaced grid for better resolution at small depths
    d_min = 1e-6
    d_max = 3.0 / k
    depths = np.geomspace(d_min, d_max, n_depths)
    
    # Value function: V[n, q] for n steps remaining, inventory q
    # q ranges from -Q to Q
    q_min = -Q
    q_max = Q
    n_q = 2 * Q + 1
    
    # V has shape (steps+1, n_q)
    # V[0, q] = 0 (no steps remaining)
    V = np.zeros((steps + 1, n_q))
    
    # Policy tables
    bid_depth = np.full((steps + 1, n_q), np.inf)
    ask_depth = np.full((steps + 1, n_q), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 30:
            # Fallback to simple heuristic if running out of time
            for q in range(n_q):
                bid_depth[n, q] = 1.0 / k
                ask_depth[n, q] = 1.0 / k
            _OPT_TABLES[key] = (bid_depth, ask_depth, depths)
            return _OPT_TABLES[key]
        
        # For each inventory level
        for qi in range(n_q):
            q = qi + q_min  # actual inventory
            
            # Inventory penalty for this step
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # Try all combinations of bid and ask depths
            # But this is O(n_depths^2) per state which is too slow
            # Instead, we can separate: for a given bid depth, find optimal ask depth, etc.
            # Actually, let's think about this differently.
            
            # The transition: with probability p_bid, bid is hit, inventory increases by 1
            # with probability p_ask, ask is hit, inventory decreases by 1
            # These are independent.
            
            # p_bid = 1 - exp(-A * exp(-k * bid_depth) * dt)
            # p_ask = 1 - exp(-A * exp(-k * ask_depth) * dt)
            
            # If bid hit: gain = bid_depth, new inventory = q + 1
            # If ask hit: gain = ask_depth, new inventory = q - 1
            # If both hit: gain = bid_depth + ask_depth, new inventory = q (net zero)
            # If neither: gain = 0, new inventory = q
            
            # Expected value = p_bid*(1-p_ask)*bid_depth + p_ask*(1-p_bid)*ask_depth + 
            #                 p_bid*p_ask*(bid_depth+ask_depth) + 
            #                 (1-p_bid)*(1-p_ask)*0 - penalty + 
            #                 (1-p_ask)*(1-p_bid)*V[n+1, q] + 
            #                 p_bid*(1-p_ask)*V[n+1, q+1] + 
            #                 p_ask*(1-p_bid)*V[n+1, q-1] + 
            #                 p_bid*p_ask*V[n+1, q]
            
            # Simplify:
            # = bid_depth*p_bid + ask_depth*p_ask - penalty + V[n+1, q] +
            #   p_bid*(1-p_ask)*(V[n+1, q+1] - V[n+1, q]) +
            #   p_ask*(1-p_bid)*(V[n+1, q-1] - V[n+1, q])
            
            # Note: if q = Q, bid cannot be hit (no bid fill at q=Q)
            # if q = -Q, ask cannot be hit (no ask fill at q=-Q)
            
            best_val = -np.inf
            best_bid = np.inf
            best_ask = np.inf
            
            if q == Q:
                # No bid fill possible
                p_bid = 0.0
                for di in range(n_depths):
                    ad = depths[di]
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
                    
                    val = ad * p_ask - penalty + V[n+1, q] + \
                          p_ask * (V[n+1, q-1] - V[n+1, q])
                    
                    if val > best_val:
                        best_val = val
                        best_bid = np.inf  # no bid
                        best_ask = ad
            elif q == -Q:
                # No ask fill possible
                p_ask = 0.0
                for di in range(n_depths):
                    bd = depths[di]
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * bd) * dt)
                    
                    val = bd * p_bid - penalty + V[n+1, q] + \
                          p_bid * (V[n+1, q+1] - V[n+1, q])
                    
                    if val > best_val:
                        best_val = val
                        best_bid = bd
                        best_ask = np.inf  # no ask
            else:
                # Both sides can be hit
                # To speed up, we can precompute some terms
                v_next_q = V[n+1, qi]
                v_next_q1 = V[n+1, qi+1]
                v_next_qm1 = V[n+1, qi-1]
                
                for di in range(n_depths):
                    bd = depths[di]
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * bd) * dt)
                    
                    for dj in range(n_depths):
                        ad = depths[dj]
                        p_ask = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
                        
                        val = bd * p_bid + ad * p_ask - penalty + v_next_q + \
                              p_bid * (1.0 - p_ask) * (v_next_q1 - v_next_q) + \
                              p_ask * (1.0 - p_bid) * (v_next_qm1 - v_next_q)
                        
                        if val > best_val:
                            best_val = val
                            best_bid = bd
                            best_ask = ad
            
            V[n, qi] = best_val
            bid_depth[n, qi] = best_bid
            ask_depth[n, qi] = best_ask
    
    _OPT_TABLES[key] = (bid_depth, ask_depth, depths)
    return _OPT_TABLES[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Fallback to simple heuristic
        depth = 1.0 / params["k"]
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
    
    # Get or compute optimal table
    bid_table, ask_table, depths = _get_optimal_table(params)
    
    Q = params["Q"]
    steps = params["steps"]
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    q_idx = inventory + Q
    
    # Clip to valid range
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Get depths for each path
    bid_depths = bid_table[n, q_idx]
    ask_depths = ask_table[n, q_idx]
    
    # Ensure values are valid: not below -3/k, not -inf, not NaN
    min_valid = -3.0 / params["k"]
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_valid), 1.0 / params["k"], bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_valid), 1.0 / params["k"], ask_depths)
    
    # Replace -inf with a large valid value
    bid_depths = np.where(bid_depths == -np.inf, 1.0 / params["k"], bid_depths)
    ask_depths = np.where(ask_depths == -np.inf, 1.0 / params["k"], ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
