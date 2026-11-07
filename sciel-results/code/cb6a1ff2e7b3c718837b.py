"""Optimal market making with inventory risk: Avellaneda-Stoikov closed-form policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def quote(t, inventory, mid, params):
    """
    Closed-form Avellaneda-Stoikov policy.
    Shifts quotes based on inventory risk.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    
    # Time remaining
    dt_rem = T - t
    
    # Inventory risk adjustment (Reservation Price Shift)
    # R = gamma * sigma^2 * q * (T - t)
    # If q > 0 (long), R > 0. We want to sell.
    # Bid should be further away (deeper) to avoid buying more.
    # Ask should be closer (shallower) to encourage selling.
    R = gamma * (sigma ** 2) * inventory * dt_rem
    
    # Optimal base depth (risk-neutral)
    # In the continuous limit, the optimal spread is often 1/k for simple models,
    # but A-S suggests a log form. Let's stick to 1/k as a robust base,
    # as the seed used 1/k and it's the peak of x*exp(-kx).
    delta_star = 1.0 / k
    
    # Calculate depths
    # Bid depth: distance below mid.
    # We want to move bid down (increase depth) if long.
    bid_depths = R + delta_star
    
    # Ask depth: distance above mid.
    # We want to move ask down (decrease depth) if long.
    ask_depths = -R + delta_star
    
    # Validity checks
    # Depths must be > -3/k. 
    # Also, physically, depths should probably be positive.
    # If ask_depths <= 0, it means we are so long we want to sell at a price below mid?
    # Or just that the quote is extremely tight.
    # The contract says "distances below and above mid".
    # If depth is negative, it implies quoting on the other side?
    # "bid_depths ... distances below mid". A negative distance would be above mid.
    # This is likely invalid or interpreted as a cross.
    # We should clip to a small positive value or np.inf.
    # If we are very long, we should NOT quote a bid. So bid_depths -> np.inf.
    # If we are very short, we should NOT quote an ask. So ask_depths -> np.inf.
    
    # Heuristic for no-quote:
    # If the shift R is so large that the "desired" depth becomes negative or very small,
    # it's better to pull the quote (np.inf) to avoid adverse selection or invalid states.
    
    min_depth = 1e-6
    
    # For bid: if R is very negative (short inventory), bid_depths = R + delta_star might be <= 0.
    # If short, we want to buy. We want bid to be tight (small depth).
    # If R is very positive (long), bid_depths is large. That's fine.
    # But if R is very negative, bid_depths < 0.
    # Let's check: if q < 0, R < 0. bid_depths = -|R| + delta_star.
    # If |R| > delta_star, bid_depths < 0.
    # This means we want to quote the bid ABOVE mid? No, bid is always below mid.
    # So we should set bid_depths to min_depth (very tight) or np.inf?
    # If we are short, we WANT to buy. So we want a tight bid.
    # So clip to min_depth.
    
    # For ask: if q > 0, R > 0. ask_depths = -R + delta_star.
    # If R > delta_star, ask_depths < 0.
    # We want to sell. We want a tight ask.
    # So clip to min_depth.
    
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Ensure no NaNs or Infs from calculation (though R is finite)
    # Also ensure shape is correct (4000,)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
