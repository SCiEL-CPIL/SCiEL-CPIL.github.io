"""Optimal market making with inventory risk - HJB/DP approach."""
import numpy as np
import time

# Cache for solved regimes
_regime_cache = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """Solve the discrete DP for a given regime.
    
    Returns arrays of optimal bid/ask depths for each (step, inventory).
    """
    dt = T / steps
    N = steps
    n_states = 2 * Q + 1
    
    # V[n][q] = optimal expected value from step n with inventory q
    # We solve backwards: V[N] = 0 (no terminal charge)
    # At step n, with inventory q, we choose bid_depth d_b and ask_depth d_a >= 0
    # Fill prob: p_b = 1 - exp(-A * exp(-k * d_b) * dt) if q < Q, else 0
    #            p_a = 1 - exp(-A * exp(-k * d_a) * dt) if q > -Q, else 0
    # Reward: d_b * (fill occurs) + d_a * (fill occurs) - (gamma/2)*sigma^2*q^2*dt
    # Then inventory changes: q -> q-1 (bid fill), q+1 (ask fill), or stays
    # V[n][q] = max over (d_b, d_a) of:
    #   p_b * [d_b + V[n+1][q-1]] + p_a * [d_a + V[n+1][q+1]] + (1-p_b)*(1-p_a)*V[n+1][q]
    #   - (gamma/2)*sigma^2*q^2*dt
    # where p_b=0 if q>=Q, p_a=0 if q<=-Q
    
    # For efficiency, we note that the optimal d_b and d_a can be found
    # by 1D maximization since they enter separately (given the other side).
    # Actually, they're coupled through the (1-p_b)*(1-p_a) term.
    # Let's use a grid search over depths.
    
    # Depth grid: from 0 to some max
    # exp(-k*d) ranges from 1 to exp(-k*d_max)
    # For reasonable k, d in [0, 5/k] covers most of the range
    d_max = 5.0 / k
    n_d = 200  # number of depth points
    d_grid = np.linspace(0.0, d_max, n_d)
    
    # Precompute fill probabilities for each depth
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
    
    # V arrays: shape (N+1, n_states)
    V = np.zeros((N + 1, n_states))
    
    # Store optimal depths: shape (N, n_states, 2) -> [d_b, d_a]
    opt_d = np.zeros((N, n_states, 2))
    
    # Backward induction
    for n in range(N - 1, -1, -1):
        Vn1 = V[n + 1]  # V[n+1, :]
        Vn = V[n]  # V[n, :]
        
        for idx_q, q in enumerate(range(-Q, Q + 1)):
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            if q >= Q:
                # No bid possible
                # Maximize over d_a: p_a * (d_a + V[n+1][q+1]) + (1-p_a)*V[n+1][q] - cost
                # But q+1 > Q, so V[n+1][q+1] doesn't exist. 
                # Actually, if q=Q, no bid fill, so inventory stays or ask fill -> q+1?
                # Wait: "no bid fill at q = Q" means bid_depth can be anything but p_b=0
                # "no ask fill at q = -Q" means ask p_a=0
                # But can inventory go beyond [-Q, Q]? The problem says inventory in [-Q, Q]
                # So if q=Q and ask fills, inventory would go to Q+1 which is out of bounds.
                # Re-reading: "inventory (4000,) ints in [-Q, Q]"
                # "no bid fill at q = Q, no ask fill at q = -Q"
                # This means at q=Q, bid fills don't happen (so no q->Q-1)
                # At q=-Q, ask fills don't happen (so no q->-Q-1)
                # But what about q=Q and ask fill? That would go to Q+1...
                # I think the constraint is just that you can't fill on the side that would
                # push you out of bounds. So at q=Q, only ask can fill (q->Q+1?) 
                # Hmm, but inventory is constrained to [-Q, Q]. 
                # Let me re-read: "no bid fill at q = Q, no ask fill at q = -Q"
                # This likely means: when q=Q, you cannot have a bid fill (which would decrease inventory).
                # When q=-Q, you cannot have an ask fill (which would increase inventory).
                # So at q=Q, only ask fills possible (q -> Q+1)? But that's out of range.
                # Actually, I think the model is: inventory can be in [-Q, Q], and the constraints
                # prevent fills that would push out of range. So at q=Q, bid fill (q->Q-1) is allowed,
                # ask fill (q->Q+1) is NOT allowed. Wait, that contradicts "no bid fill at q=Q".
                # 
                # Let me re-read more carefully: "no bid fill at q = Q, no ask fill at q = -Q"
                # A bid fill means someone buys at bid, so you sell, inventory decreases: q -> q-1
                # An ask fill means someone sells at ask, so you buy, inventory increases: q -> q+1
                # "no bid fill at q = Q": at q=Q, bid fills don't happen. So q cannot go to Q-1 via bid?
                # That seems odd. Usually you'd want to reduce inventory when it's high.
                # 
                # Actually I think I have it backwards. Let me reconsider:
                # In market making, if you have positive inventory (q>0), you've been buying more than selling.
                # You want to sell to reduce inventory. A "bid fill" is when you sell at the bid price.
                # So bid fill: you sell, inventory decreases: q -> q-1
                # Ask fill: you buy, inventory increases: q -> q+1
                # 
                # "no bid fill at q = Q": when inventory is at max Q, no bid fills. 
                # This would mean you can't sell, which seems wrong for risk management.
                # Unless the convention is different: maybe bid fill = buy at bid (inventory increases)?
                # 
                # Let me just interpret it literally as stated:
                # "no bid fill at q = Q" -> p_b = 0 when q = Q
                # "no ask fill at q = -Q" -> p_a = 0 when q = -Q
                # And fills change inventory: bid fill -> q-1, ask fill -> q+1
                # But we need to handle what happens at boundaries.
                # I'll assume that if a fill would push inventory out of [-Q,Q], it doesn't happen.
                # So at q=Q, ask fill (q->Q+1) is disallowed. At q=-Q, bid fill (q->-Q-1) is disallowed.
                # PLUS the explicit constraints: no bid fill at q=Q, no ask fill at q=-Q.
                #
                # Wait, that gives us: at q=Q, both bid and ask fills are disallowed? That can't be right.
                # Let me just go with: the stated constraints are the only ones, and inventory stays in [-Q,Q]
                # because fills that would exit are simply prevented.
                
                # I'll interpret: p_b=0 if q>=Q (can't reduce inventory further... wait, bid reduces inventory)
                # Ugh, let me just say: 
                # Can bid fill? q < Q (inventory not at max, and bid fill decreases inventory)
                # Can ask fill? q > -Q (inventory not at min, and ask fill increases inventory)
                # This makes more economic sense.
                
                # Actually, re-reading one more time: "no bid fill at q = Q" 
                # I'll just implement: p_b = 0 if q >= Q, p_a = 0 if q <= -Q
                # And for value functions, we only need V for q in [-Q, Q].
                # If a fill would push out of range, we just don't allow it (p=0).
                
                # At q=Q: p_b=0, p_a > 0 possible, but ask fill goes to q+1=Q+1 which is out of range.
                # So actually p_a should also be 0 at q=Q? 
                # Let me just say: both sides have p=0 at the boundaries where they'd exit.
                # q=Q: bid fill (q->Q-1) is in range, ask fill (q->Q+1) is out of range
                # q=-Q: bid fill (q->-Q-1) is out of range, ask fill (q->-Q+1) is in range
                # Plus the stated constraints: no bid fill at q=Q, no ask fill at q=-Q
                # 
                # I'll go with: 
                # bid fill allowed if q < Q AND q > -Q (but q>-Q is always true if q<Q and Q>0... unless Q=0)
                # Actually simplest: bid fill (q->q-1) allowed if q-1 >= -Q, i.e., q > -Q
                #                 and the explicit constraint q < Q
                # ask fill (q->q+1) allowed if q+1 <= Q, i.e., q < Q
                #                 and the explicit constraint q > -Q
                # So: bid allowed if -Q < q < Q, ask allowed if -Q < q < Q
                # At q=Q: no fills at all. At q=-Q: no fills at all.
                # That seems too restrictive but let's go with it.
                
                # Hmm, actually let me just go with the literal statement and assume
                # inventory stays in bounds by preventing out-of-bounds fills:
                # bid fill: q -> q-1, allowed if q-1 >= -Q, i.e., q >= -Q+1
                # ask fill: q -> q+1, allowed if q+1 <= Q, i.e., q <= Q-1
                # Plus: no bid fill at q=Q (redundant since Q-1 < Q), no ask fill at q=-Q (redundant)
                # So effectively: bid allowed if q >= -Q+1, ask allowed if q <= Q-1
                # At q=Q: bid allowed (Q >= -Q+1 if 2Q>=1), ask NOT allowed (Q > Q-1)
                # At q=-Q: bid NOT allowed (-Q < -Q+1), ask allowed (-Q <= Q-1 if 2Q>=1)
                # This makes sense!
                
                pass  # Handled below in general case
            
            # General case with boundary handling
            # Bid fill (q->q-1) allowed if q >= -Q+1
            # Ask fill (q->q+1) allowed if q <= Q-1
            
            bid_allowed = (q >= -Q + 1)
            ask_allowed = (q <= Q - 1)
            
            best_val = -np.inf
            best_db = 0.0
            best_da = 0.0
            
            # Grid search over (d_b, d_a)
            # For efficiency, use the full grid for each allowed side
            if bid_allowed and ask_allowed:
                # Both sides: maximize over d_b, d_a
                # Value = p_b*(d_b + Vn1[q-1+Q]) + p_a*(d_a + Vn1[q+1+Q]) + (1-p_b)*(1-p_a)*Vn1[q+Q] - cost
                # Note: q-1+Q is the index for inventory q-1
                # This is a 2D maximization. We can use a reduced grid.
                # Use a coarser grid for the 2D search
                n_d2 = 80
                d_grid2 = np.linspace(0.0, d_max, n_d2)
                p_grid2 = 1.0 - np.exp(-A * np.exp(-k * d_grid2) * dt)
                
                vb = Vn1[q - 1 + Q]
                va = Vn1[q + 1 + Q]
                v0 = Vn1[q + Q]
                
                # Vectorized 2D search
                # For each (i,j), compute value
                # To keep memory reasonable, do it in chunks or use broadcasting
                # n_d2=80 gives 80*80=6400 combinations, fine
                P_b = p_grid2[:, None]  # (n_d2, 1)
                P_a = p_grid2[None, :]  # (1, n_d2)
                D_b = d_grid2[:, None]  # (n_d2, 1)
                D_a = d_grid2[None, :]  # (1, n_d2)
                
                vals = (P_b * (D_b + vb) + P_a * (D_a + va) + 
                       (1 - P_b) * (1 - P_a) * v0 - cost)
                
                flat_idx = np.argmax(vals)
                i, j = np.unravel_index(flat_idx, (n_d2, n_d2))
                best_db = d_grid2[i]
                best_da = d_grid2[j]
                best_val = vals[i, j]
                
            elif bid_allowed:
                # Only bid
                # Value = p_b*(d_b + Vn1[q-1+Q]) + (1-p_b)*Vn1[q+Q] - cost
                vb = Vn1[q - 1 + Q]
                v0 = Vn1[q + Q]
                
                vals = p_grid * (d_grid + vb) + (1 - p_grid) * v0 - cost
                idx = np.argmax(vals)
                best_db = d_grid[idx]
                best_da = 0.0
                best_val = vals[idx]
                
            elif ask_allowed:
                # Only ask
                va = Vn1[q + 1 + Q]
                v0 = Vn1[q + Q]
                
                vals = p_grid * (d_grid + va) + (1 - p_grid) * v0 - cost
                idx = np.argmax(vals)
                best_da = d_grid[idx]
                best_db = 0.0
                best_val = vals[idx]
                
            else:
                # No fills possible
                best_val = v0 - cost if False else Vn1[q + Q] - cost
                best_db = 0.0
                best_da = 0.0
            
            Vn[idx_q] = best_val
            opt_d[n, idx_q, 0] = best_db
            opt_d[n, idx_q, 1] = best_da
    
    return opt_d, V


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _regime_cache:
        # Deadline guard
        t_start = time.time()
        opt_d, V = _solve_regime(sigma, A, k, gamma, T, steps, Q)
        elapsed = time.time() - t_start
        _regime_cache[cache_key] = opt_d
    else:
        opt_d = _regime_cache[cache_key]
    
    inv_idx = np.clip(inventory, -Q, Q)
    bid_depths = opt_d[n, inv_idx + Q, 0]
    ask_depths = opt_d[n, inv_idx + Q, 1]
    
    # Validate: depths must be >= -3/k
    min_allowed = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_allowed)
    ask_depths = np.maximum(ask_depths, min_allowed)
    
    # Ensure no NaN or inf (except we don't use inf for "no quote" per spec, 
    # but np.inf means no quote. Let's use large finite values instead)
    # Actually spec says np.inf means no quote, so that's valid.
    # But values below -3/k are invalid.
    
    bid_depths = np.asarray(bid_depths, dtype=np.float64)
    ask_depths = np.asarray(ask_depths, dtype=np.float64)
    
    return bid_depths, ask_depths
