import numpy as np
from functools import lru_cache

# Cache for precomputed policies
_policy_cache = {}

def _precompute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for each (t, q).
    Returns two arrays of shape (steps, 2*Q+1) for bid and ask depths.
    """
    dt = T / steps
    # Discretize depths. 
    # Theoretical max depth is limited by -3/k. 
    # We can use a logarithmic or linear grid.
    # Let's try a linear grid from a small positive value to a max value.
    # However, the optimal depth might be very small.
    # Let's use a fine grid.
    
    # To speed up, we can note that for large q, we want to reduce inventory.
    # For q > 0, we want smaller ask (to sell) and larger bid (to avoid buying).
    # For q < 0, we want smaller bid (to buy) and larger ask (to avoid selling).
    
    # Let's define a grid of depths.
    # min_depth is slightly above -3/k.
    # Let's use 100 points for depth search.
    
    # Actually, solving the 2D optimization for each (t,q) is expensive if done naively.
    # We can assume symmetry or separability? No, fills are independent but coupled via inventory change.
    
    # Let's use a smaller grid for speed, say 50 points.
    # Depth range: [0.01, 5.0/k] ?
    # Let's check typical scales. k=1.5 -> 1/k ~ 0.66.
    # Let's use np.linspace(0.01, 5.0, 50) ? No, depends on k.
    # Let's use relative to 1/k.
    
    max_depth = 5.0 / k
    min_depth = 0.01
    depths = np.linspace(min_depth, max_depth, 50)
    
    # V[t, q] where q index is q + Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    # Iterate backwards from t = steps - 1 down to 0
    for t in range(steps - 1, -1, -1):
        for qi in range(-Q, Q + 1):
            q = qi
            # Penalty for current inventory
            penalty = -0.5 * gamma * sigma**2 * q**2 * dt
            
            best_val = -np.inf
            best_b = depths[0]
            best_a = depths[0]
            
            # Search over bid and ask depths
            # Optimization: We can restrict the search range based on q.
            # But for simplicity and robustness, we search the full grid.
            # 50x50 = 2500 evaluations per state.
            # 1000 steps * 61 states * 2500 ops = 152.5 million ops.
            # This might be too slow in Python loops.
            
            # We need to vectorize the inner loop or reduce grid size.
            # Let's try grid size 20. 400 ops per state.
            # 1000 * 61 * 400 = 24.4 million. Still heavy in pure Python.
            
            # Alternative: Use NumPy vectorization for the inner loop.
            
            # p_b depends on b, p_a depends on a.
            # E[reward] = p_b * b * (1-p_a) + p_a * a * (1-p_b) + p_b * p_a * (b+a) + 0
            #           = p_b * b - p_b*p_a*b + p_a * a - p_b*p_a*a + p_b*p_a*b + p_b*p_a*a
            #           = p_b * b + p_a * a
            # Wait, let's re-verify.
            # Outcomes:
            # 1. Bid only: prob (1-p_a)p_b, reward b, next q+1
            # 2. Ask only: prob (1-p_b)p_a, reward a, next q-1
            # 3. Both: prob p_b p_a, reward b+a, next q
            # 4. None: prob (1-p_b)(1-p_a), reward 0, next q
            
            # Expected reward = (1-p_a)p_b b + (1-p_b)p_a a + p_b p_a (b+a)
            # = p_b b - p_b p_a b + p_a a - p_b p_a a + p_b p_a b + p_b p_a a
            # = p_b b + p_a a
            
            # So immediate reward is simply p_b * b + p_a * a.
            # Next state value:
            # V_next = (1-p_a)p_b V(t+1, q+1) + (1-p_b)p_a V(t+1, q-1) + (p_b p_a + (1-p_b)(1-p_a)) V(t+1, q)
            
            # We need to maximize:
            # penalty + p_b b + p_a a + E[V_next]
            
            # Let's vectorize over b and a.
            # b_grid shape (M,), a_grid shape (M,)
            # p_b shape (M,), p_a shape (M,)
            
            # We can compute the value for all b for a fixed a, or use meshgrid.
            # Meshgrid MxM might be large.
            
            # Let's try a smaller grid, M=20.
            depths_small = np.linspace(0.01, 3.0/k, 20)
            
            p_b = 1 - np.exp(-A * np.exp(-k * depths_small) * dt)
            p_a = 1 - np.exp(-A * np.exp(-k * depths_small) * dt)
            
            # Create meshgrids
            B, A_depth = np.meshgrid(depths_small, depths_small, indexing='ij')
            PB, PA = np.meshgrid(p_b, p_a, indexing='ij')
            
            # Immediate rewards
            rew_b = PB * B
            rew_a = PA * A_depth
            
            # Next state probabilities
            # q+1
            prob_q_plus = (1 - PA) * PB
            # q-1
            prob_q_minus = (1 - PB) * PA
            # q
            prob_q = PB * PA + (1 - PB) * (1 - PA)
            
            # Get V(t+1, ...)
            q_idx = q + Q
            # Handle boundaries
            # If q == Q, q+1 is invalid (no bid fill allowed? No, "no bid fill at q=Q" means we don't quote bid? 
            # The task says: "no bid fill at q = Q, no ask fill at q = -Q".
            # This implies if q=Q, we should not quote a bid (or bid depth inf).
            # If q=-Q, we should not quote an ask.
            
            # In our DP, if q=Q, we restrict b to inf (or very large) such that p_b=0?
            # Or simply, if q=Q, we don't consider transitions to Q+1.
            # The model says "no bid fill at q=Q". This is a constraint on the process.
            # So if q=Q, p_b is effectively 0 regardless of b?
            # Or does it mean we must set b=inf?
            # "np.inf means no quote".
            # So if q=Q, we set b=inf, p_b=0.
            # If q=-Q, we set a=inf, p_a=0.
            
            # Let's handle this by masking or setting values.
            # For simplicity in precomputation:
            if q == Q:
                # Force p_b to 0
                PB = np.zeros_like(PB)
                # And we don't care about b value for reward, but p_b=0 makes rew_b=0.
                # prob_q_plus = 0
                prob_q_plus = np.zeros_like(prob_q_plus)
                prob_q_minus = (1 - PB) * PA # = PA
                prob_q = 1 - PA # since PB=0, (1-PB)=1, PB*PA=0, (1-PB)(1-PA)=(1-PA). Sum = 1-PA.
                # Wait, prob_q = PB*PA + (1-PB)(1-PA) = 0 + 1*(1-PA) = 1-PA.
                # prob_q_minus = (1-PB)PA = PA.
                # Sum = 1. Correct.
                
                # We should also set B to something valid, but it doesn't matter if PB=0.
                # However, we want to return a valid depth. We can return 0 or inf.
                # The task says return depths. If we don't quote, inf is appropriate?
                # "np.inf means no quote".
                # So for q=Q, optimal bid is inf.
                # We will handle this in the final return logic.
                # For DP, we just need the value.
                
            if q == -Q:
                # Force p_a to 0
                PA = np.zeros_like(PA)
                prob_q_minus = np.zeros_like(prob_q_minus)
                prob_q_plus = (1 - PA) * PB # = PB
                prob_q = 1 - PB
                
            # V_next values
            # q_idx is index for current q.
            # q+1 index: q_idx + 1
            # q-1 index: q_idx - 1
            
            V_next_q = V[t+1, q_idx]
            
            # Handle boundaries for V_next
            if q < Q:
                V_next_q_plus = V[t+1, q_idx + 1]
            else:
                V_next_q_plus = 0 # Should not be accessed if prob is 0, but safe.
                
            if q > -Q:
                V_next_q_minus = V[t+1, q_idx - 1]
            else:
                V_next_q_minus = 0
            
            V_next = prob_q_plus * V_next_q_plus + prob_q_minus * V_next_q_minus + prob_q * V_next_q
            
            total_val = penalty + rew_b + rew_a + V_next
            
            # Find max
            max_val = np.max(total_val)
            idx = np.argmax(total_val)
            i, j = np.unravel_index(idx, total_val.shape)
            
            best_b = depths_small[i]
            best_a = depths_small[j]
            
            V[t, q_idx] = max_val
            opt_bid[t, q_idx] = best_b
            opt_ask[t, q_idx] = best_a
            
            # Handle boundary quotes
            if q == Q:
                opt_bid[t, q_idx] = np.inf
            if q == -Q:
                opt_ask[t, q_idx] = np.inf
                
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        # Precompute policy
        # This will take some time, but only once per regime.
        # To avoid timeout, we need to make this fast.
        # If 20x20 grid is too slow, we might need to reduce steps or grid.
        # But 1000 steps is fixed.
        # Let's try.
        
        # Optimization: Use smaller grid if possible?
        # Let's stick with 20 for now.
        
        opt_bid, opt_ask = _precompute_policy(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (opt_bid, opt_ask)
    else:
        opt_bid, opt_ask = _policy_cache[key]
        
    # Get current step index
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
        
    # Get inventory indices
    # inventory is array of ints
    q_idx = inventory + Q
    
    # Fetch depths
    d_bid = opt_bid[n, q_idx]
    d_ask = opt_ask[n, q_idx]
    
    # Ensure shapes and types
    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    
    return d_bid, d_ask
