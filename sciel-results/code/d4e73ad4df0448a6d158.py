"""Market making with inventory risk: Analytical Taylor Expansion Policy."""
# EVOLVE-BLOCK-START
import numpy as np

def quote(t, inventory, mid, params):
    """
    Quote bid/ask depths using a closed-form approximation derived from
    a second-order Taylor expansion of the value function.
    
    The optimal depth is approximated as:
    d_bid  = max(0, 1/k + beta * (2*inventory + 1))
    d_ask  = max(0, 1/k + beta * (-2*inventory + 1))
    
    where beta captures the inventory risk curvature.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    
    # Estimate beta (curvature of value function)
    # Based on the balance between risk aversion (gamma * sigma^2) and fill intensity (A * k)
    # A common scaling in such models is beta ~ gamma * sigma^2 / (2 * A * k)
    # We use a robust estimate. If A is small, beta should be larger (wider quotes to avoid risk).
    beta = (gamma * sigma**2) / (2.0 * A * k)
    
    # Convert inventory to array for vectorized operations
    q = np.asarray(inventory, dtype=np.float64)
    
    # Calculate optimal depths
    # Term 1: 1/k is the baseline depth where marginal profit is maximized for a single side
    # Term 2: Inventory adjustment
    # If q > 0 (long), we want to sell (ask) more aggressively (smaller d_ask) and buy (bid) less aggressively (larger d_bid).
    # d_ask = 1/k + beta * (-2q + 1). If q is large positive, -2q is large negative, reducing d_ask.
    # d_bid = 1/k + beta * (2q + 1). If q is large positive, 2q is large positive, increasing d_bid.
    
    bid_depths = 1.0 / k + beta * (2.0 * q + 1.0)
    ask_depths = 1.0 / k + beta * (-2.0 * q + 1.0)
    
    # Enforce non-negativity
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Handle boundary conditions explicitly as per problem statement:
    # "no bid fill at q = Q, no ask fill at q = -Q"
    # This implies we should not quote bid if q == Q, and not quote ask if q == -Q.
    # The problem states "np.inf means no quote".
    # However, the prompt says "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid."
    # It does NOT say np.inf is invalid, it says "np.inf means no quote".
    # But wait, the prompt says "distances below and above mid ... np.inf means no quote".
    # So we should return np.inf for the side that is forbidden.
    
    Q = params['Q']
    
    # If q >= Q, no bid. Set bid to inf.
    # If q <= -Q, no ask. Set ask to inf.
    bid_depths = np.where(q >= Q, np.inf, bid_depths)
    ask_depths = np.where(q <= -Q, np.inf, ask_depths)
    
    # Ensure no NaNs
    bid_depths = np.where(np.isnan(bid_depths), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 1.0/k, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
