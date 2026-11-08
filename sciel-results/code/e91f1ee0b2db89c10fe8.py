"""Market making with inventory risk - HJB-based optimal quoting with vectorization."""
import numpy as np
import time

_policy_cache = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """Compute optimal bid/ask depths via backward induction with vectorized depth search."""
    dt = T / steps
    n_levels = 2 * Q + 1
    
    V = np.zeros((steps + 1, n_levels))
    opt_bid = np.full((steps, n_levels), 1.0 / k)
    opt_ask = np.full((steps, n_levels), 1.0 / k)
    
    # Depth grid
    d_min = 0.02 / k
    d_max = 15.0 / k
    n_depth = 60
    depth_grid = np.geomspace(d_min, d_max, n_depth)
    
    # Precompute fill probabilities
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # For each step, vectorize over inventory levels
    for n in range(steps - 1, -1, -1):
        V_next = V[n + 1, :]  # shape (n_levels,)
        
        # For each inventory level, we need to find optimal (bid, ask)
        # Vectorize: create 2D grid of (bid_idx, ask_idx)
        # But that's n_depth^2 x n_levels which is large
        
        # Alternative: for each bid depth, find best ask depth, then pick best bid
        # Or use a coarser 2D search
        
        # Let's do: for each bid depth i, compute the best ask depth j,
        # then pick the best i. This is O(n_depth^2 * n_levels) but vectorized over n_levels.
        
        best_val_all = np.full(n_levels, -np.inf)
        best_bid_all = np.full(n_levels, depth_grid[0])
        best_ask_all = np.full(n_levels, depth_grid[0])
        
        inv_levels = np.arange(-Q, Q + 1)
        inv_costs = 0.5 * gamma * sigma**2 * inv_levels**2 * dt  # shape (n_levels,)
        
        for i in range(n_depth):
            bd = depth_grid[i]
            pb = fill_probs[i]
            
            # For each ask depth j, compute the total value for all inventory levels
            # This is vectorized over j and inventory levels
            # pa varies with j
            pa_vec = fill_probs  # shape (n_depth,)
            
            # For each inventory level q:
            # v_bid[q] = V_next[q+1] if q < Q else V_next[q]
            # v_ask[q] = V_next[q-1] if q > -Q else V_next[q]
            
            v_bid = np.where(inv_levels < Q, V_next[np.minimum(inv_levels + 1 + Q, n_levels - 1)], V_next)
            v_ask = np.where(inv_levels > -Q, V_next[np.maximum(inv_levels - 1 + Q, 0)], V_next)
            v_none = V_next
            
            # For each (i, j) pair, compute total value for all q
            # total[j, q] = pb*(1-pa_j)*bd + (1-pb)*pa_j*ad_j + pb*pa_j*(bd+ad_j) 
            #             - inv_cost[q] + pb*(1-pa_j)*v_bid[q] + (1-pb)*pa_j*v_ask[q] 
            #             + pb*pa_j*v_none[q] + (1-pb)*(1-pa_j)*v_none[q]
            
            # Let's compute for all j at once
            # pa_vec shape (n_depth,), ad_vec = depth_grid shape (n_depth,)
            ad_vec = depth_grid
            
            # Terms:
            # reward = pb*(1-pa)*bd + (1-pb)*pa*ad + pb*pa*(bd+ad)
            #        = pb*bd*(1-pa) + (1-pb)*pa*ad + pb*pa*bd + pb*pa*ad
            #        = pb*bd - pb*bd*pa + (1-pb)*pa*ad + pb*pa*bd + pb*pa*ad
            #        = pb*bd + pa*ad*((1-pb) + pb) - pb*bd*pa + pb*pa*bd
            #        = pb*bd + pa*ad - pb*pa*bd + pb*pa*bd
            #        = pb*bd + pa*ad
            
            # Wait, that simplifies nicely! Let me verify:
            # P(bid only) = pb(1-pa), reward = bd
            # P(ask only) = (1-pb)pa, reward = ad  
            # P(both) = pb*pa, reward = bd + ad
            # P(neither) = (1-pb)(1-pa), reward = 0
            # E[reward] = pb(1-pa)bd + (1-pb)pa*ad + pb*pa*(bd+ad)
            #           = pb*bd - pb*pa*bd + (1-pb)pa*ad + pb*pa*bd + pb*pa*ad
            #           = pb*bd + pa*ad - pb*pa*ad + pb*pa*ad
            #           = pb*bd + pa*ad
            
            # Yes! E[reward] = pb*bd + pa*ad. Much simpler.
            
            # Future value:
            # E[V_next] = pb(1-pa)v_bid + (1-pb)pa*v_ask + pb*pa*v_none + (1-pb)(1-pa)*v_none
            #           = pb(1-pa)v_bid + (1-pb)pa*v_ask + [pb*pa + (1-pb)(1-pa)]*v_none
            #           = pb(1-pa)v_bid + (1-pb)pa*v_ask + [1 - pb*pa - (1-pb)(1-pa)]*v_none
            #           = pb(1-pa)v_bid + (1-pb)pa*v_ask + [pb + pa - 2*pb*pa]*v_none
            
            # Hmm, let me just compute directly.
            
            # For vectorization over j (ask depth index):
            # pa shape (n_depth,)
            # reward_j = pb*bd + pa_j*ad_j  -> shape (n_depth,)
            
            # For future value, it depends on q, so shape (n_depth, n_levels)
            # This might be too large: 60 * 61 = 3660, which is fine.
            
            # Let's compute for all j at once
            # pa shape (n_depth,)
            # For each q, the future value as function of j:
            # fv[j, q] = pb*(1-pa[j])*v_bid[q] + (1-pb)*pa[j]*v_ask[q] + pa[j]*pb*v_none[q] + (1-pa[j])*(1-pb)*v_none[q]
            
            # Let me expand:
            # = pb*v_bid[q] - pb*pa[j]*v_bid[q] + (1-pb)*pa[j]*v_ask[q] + pa[j]*pb*v_none[q] + v_none[q] - pb*v_none[q] - pa[j]*v_none[q] + pb*pa[j]*v_none[q]
            # = pb*v_bid[q] + (1-pb)*v_none[q] - pb*v_none[q] + pa[j]*(-pb*v_bid[q] + (1-pb)*v_ask[q] + pb*v_none[q] - v_none[q] + pb*v_none[q])
            # Let me just compute numerically.
            
            # fv[j, q] = pb*(1-pa[j])*v_bid[q] + (1-pb)*pa[j]*v_ask[q] + pa[j]*pb*v_none[q] + (1-pa[j])*(1-pb)*v_none[q]
            
            # For fixed q, this is linear in pa[j]:
            # = pb*v_bid[q] + pa[j]*(-pb*v_bid[q] + (1-pb)*v_ask[q] + pb*v_none[q] - (1-pb)*v_none[q])
            # = pb*v_bid[q] + pa[j]*(-pb*v_bid[q] + (1-pb)*v_ask[q] + pb*v_none[q] - v_none[q] + pb*v_none[q])
            # = pb*v_bid[q] + pa[j]*(-pb*v_bid[q] + (1-pb)*v_ask[q] + (2*pb - 1)*v_none[q])
            
            # Hmm, this is getting complicated. Let me just compute the full matrix.
            
            # reward shape (n_depth,)
            reward = pb * bd + pa_vec * ad_vec  # (n_depth,)
            
            # Future value: shape (n_depth, n_levels)
            # fv[j, q] = pb*(1-pa[j])*v_bid[q] + (1-pb)*pa[j]*v_ask[q] + pa[j]*pb*v_none[q] + (1-pa[j])*(1-pb)*v_none[q]
            
            # Let me compute term by term
            # Term1: pb*(1-pa[j])*v_bid[q] -> (n_depth, n_levels)
            # Term2: (1-pb)*pa[j]*v_ask[q] -> (n_depth, n_levels)
            # Term3: pa[j]*pb*v_none[q] -> (n_depth, n_levels)
            # Term4: (1-pa[j])*(1-pb)*v_none[q] -> (n_depth, n_levels)
            
            # To save memory, let's compute for each j separately but vectorize over q
            # Actually, n_depth=60, n_levels=61, so 60*61=3660 floats per i, that's fine.
            
            fv = np.zeros((n_depth, n_levels))
            for j in range(n_depth):
                pa_j = pa_vec[j]
                fv[j, :] = pb * (1 - pa_j) * v_bid + (1 - pb) * pa_j * v_ask + pa_j * pb * v_none + (1 - pa_j) * (1 - pb) * v_none
            
            # Total value: reward[j] - inv_cost[q] + fv[j, q]
            total = reward[:, np.newaxis] - inv_costs[np.newaxis, :] + fv  # (n_depth, n_levels)
            
            # Find best j for each q
            best_j = np.argmax(total, axis=0)  # shape (n_levels,)
            best_val_for_i = total[best_j, np.arange(n_levels)]  # shape (n_levels,)
            
            # Update global best
            mask = best_val_for_i > best_val_all
            if np.any(mask):
                best_val_all[mask] = best_val_for_i[mask]
                best_bid_all[mask] = bd
                best_ask_all[mask] = depth_grid[best_j[mask]]
        
        V[n, :] = best_val_all
        opt_bid[n, :] = best_bid_all
        opt_ask[n, :] = best_ask_all
    
    return opt_bid, opt_ask, depth_grid


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get or compute policy
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(sigma, A, k, gamma, T, steps, Q)
    
    opt_bid, opt_ask, depth_grid = _policy_cache[key]
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = min(n, steps - 1)
    
    # Get depths for each path based on inventory
    n_paths = len(inventory)
    bid_depths = np.empty(n_paths)
    ask_depths = np.empty(n_paths)
    
    for i in range(n_paths):
        q = int(inventory[i])
        q = max(-Q, min(Q, q))
        qi = q + Q
        
        bid_depths[i] = opt_bid[n, qi]
        ask_depths[i] = opt_ask[n, qi]
    
    return bid_depths, ask_depths
