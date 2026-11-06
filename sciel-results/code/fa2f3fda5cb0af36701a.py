import time
import random
import numpy as np

_N = 128
_LEN = 129
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets
_DEADLINE = 55.0      # safety margin before the 60 s kill

_START = time.time()

def _elapsed():
    return time.time() - _START

def _compute_initial(x):
    """Compute Y and U for initial x."""
    y = np.convolve(x, x, mode="full")
    ks = np.arange(_T0, _T1 + 1)
    is_even = (ks % 2 == 0)
    mids = ks // 2
    diag_corr = np.where(is_even, x[mids], 0)
    u = (y[_T0:_T1 + 1] + diag_corr) // 2
    return y, u

def _update_after_flip(x, y, u, i, adding):
    """
    Update y and u after flipping x[i].
    Returns new x, y, u.
    """
    # We modify x in place or return new? 
    # For SA, it's better to update in place and revert if rejected.
    # But we need to be careful with y and u.
    # Let's compute delta.
    
    # Delta Y
    # shift(x, i) is x[k-i]
    # y_new = y + 2 * shift(x, i) + delta_2i (if adding)
    # y_new = y - 2 * shift(x, i) - delta_2i (if removing)
    
    # We only care about y in [_T0, _T1] for u, 
    # but y is used for future updates? 
    # No, future updates only depend on current x and current y.
    # So we must update the full y vector (length 257).
    
    sign = 1 if adding else -1
    
    # Create shift vector
    # shift[k] = x[k-i] if 0 <= k-i < _LEN else 0
    # This is x shifted by i.
    # We can do this with slicing.
    # y length is 2*_LEN - 1 = 257.
    # i is in [0, 128].
    
    # Indices in y affected: i to i+_LEN-1.
    # x indices: 0 to _LEN-1.
    # y[k] += sign * 2 * x[k-i]
    
    # Let's just compute the full delta to be safe and simple, 
    # or use numpy slicing for speed.
    
    # Slicing approach:
    # The vector s of length 257 where s[k] = x[k-i]
    # s[i : i+_LEN] = x
    
    # We need to add sign * 2 * s to y.
    # And sign * 1 to y[2*i].
    
    # But wait, if we are adding i, x[i] was 0. 
    # The term x[i]*x[i] is new. 
    # The cross terms x[i]*x[j] for j != i are new.
    # The formula y_new = y + 2*x*e_i + e_i*e_i works.
    # x*e_i is x shifted by i.
    # e_i*e_i is delta at 2i.
    
    # So:
    # y[i:i+_LEN] += sign * 2 * x
    # y[2*i] += sign
    
    # However, we must be careful: 
    # If we are adding i, x[i] is currently 0. 
    # The shift x used in the formula is the OLD x (before flip).
    # So x[i] is 0. 
    # y[i:i+_LEN] += 2 * x_old
    # y[2*i] += 1
    # This is correct.
    
    # If we are removing i, x[i] is currently 1.
    # y[i:i+_LEN] -= 2 * x_old
    # y[2*i] -= 1
    # This is correct.
    
    # Now update u.
    # u = (y_window + diag) // 2
    # Only y_window and diag change.
    # y_window is y[_T0:_T1+1].
    # diag is D_k.
    # D_k changes only if k=2*i.
    # If 2*i is in [_T0, _T1], then D_{2*i} changes.
    # D_{2*i} = 1 if x[i]=1 else 0.
    # If adding: D goes 0->1.
    # If removing: D goes 1->0.
    
    # Let's update y first.
    y[i:i+_LEN] += sign * 2 * x
    y[2*i] += sign
    
    # Update x
    x[i] = 1 if adding else 0
    
    # Update u
    # u is derived from y and x.
    # We can recompute u from y and x? 
    # u = (y[_T0:_T1+1] + diag_corr) // 2
    # diag_corr depends on x.
    # Recalculating u is O(W) = 129 ops. Very fast.
    ks = np.arange(_T0, _T1 + 1)
    is_even = (ks % 2 == 0)
    mids = ks // 2
    diag_corr = np.where(is_even, x[mids], 0)
    u = (y[_T0:_T1 + 1] + diag_corr) // 2
    
    return x, y, u

def construct():
    global _START
    _START = time.time()
    
    rng = random.Random(12345)
    
    best_x = None
    best_peak = float("inf")
    best_cost = float("inf")
    
    # Initial solution: random subset of size 64
    # We'll try a few different initial sizes or random seeds
    # and run SA on each, keeping the best result.
    
    for attempt in range(10): # 10 restarts
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Random initial set
        # Size between 60 and 75
        size = rng.randint(60, 75)
        x = np.zeros(_LEN, dtype=np.int64)
        indices = rng.sample(range(_LEN), size)
        x[indices] = 1
        
        y, u = _compute_initial(x)
        
        # Calculate initial cost
        peak = np.max(u)
        missing = np.sum(u == 0)
        cost = peak + 100 * missing
        
        if missing == 0 and peak < best_peak:
            best_peak = peak
            best_x = x.copy()
            best_cost = cost
            
        # Simulated Annealing
        T_start = 10.0
        T_end = 0.01
        num_iters = 50000 # Adjust based on time
        
        # Estimate time per iter
        # ~10-20us
        # 50000 * 20us = 1s
        # We have ~5s per attempt if 10 attempts.
        
        t = T_start
        decay = (T_end / T_start) ** (1.0 / num_iters)
        
        current_x = x.copy()
        current_y = y.copy()
        current_u = u.copy()
        current_cost = cost
        
        for it in range(num_iters):
            if _elapsed() > _DEADLINE - 2:
                break
                
            # Propose move: flip a random bit
            i = rng.randint(0, _LEN - 1)
            adding = current_x[i] == 0
            
            # Save state
            old_x_i = current_x[i]
            old_y = current_y.copy()
            old_u = current_u.copy()
            old_cost = current_cost
            
            # Apply move
            new_x, new_y, new_u = _update_after_flip(current_x, current_y, current_u, i, adding)
            
            # Calculate new cost
            new_peak = np.max(new_u)
            new_missing = np.sum(new_u == 0)
            new_cost = new_peak + 100 * new_missing
            
            # Acceptance
            delta = new_cost - old_cost
            if delta < 0:
                # Accept
                current_x = new_x
                current_y = new_y
                current_u = new_u
                current_cost = new_cost
                
                if new_missing == 0 and new_peak < best_peak:
                    best_peak = new_peak
                    best_x = current_x.copy()
                    best_cost = new_cost
            else:
                # Maybe accept
                if delta == 0:
                    # Accept to explore
                    current_x = new_x
                    current_y = new_y
                    current_u = new_u
                    current_cost = new_cost
                else:
                    prob = np.exp(-delta / t)
                    if rng.random() < prob:
                        current_x = new_x
                        current_y = new_y
                        current_u = new_u
                        current_cost = new_cost
                    else:
                        # Revert
                        # We already saved old state, but _update_after_flip modified in place?
                        # No, _update_after_flip returns new arrays or modifies?
                        # In my implementation above, it modifies x, y, u in place?
                        # Let's check _update_after_flip.
                        # It does y[i:i+_LEN] += ... which modifies y in place.
                        # And x[i] = ... modifies x in place.
                        # And u = ... creates new u.
                        # So x and y are modified in place.
                        # I need to revert them.
                        
                        # Revert x
                        current_x[i] = old_x_i
                        # Revert y
                        # y[i:i+_LEN] -= sign * 2 * old_x (but old_x had x[i]=old_x_i)
                        # It's safer to just copy back old_y
                        current_y[:] = old_y
                        current_u[:] = old_u
                        current_cost = old_cost
                        
            t *= decay
            
    if best_x is None:
        # Fallback
        best_x = np.ones(_LEN, dtype=np.int64)
        
    result = [int(i) for i in np.where(best_x == 1)[0]]
    return result
