# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths), float arrays of shape (N,), distances
    below/above mid to quote this step; np.inf means no quote."""
    sigma = float(params["sigma"])
    A = float(params["A"])
    k = float(params["k"])
    gamma = float(params["gamma"])
    T = float(params["T"])
    Q = int(params["Q"])

    n = len(inventory)
    q = np.asarray(inventory, dtype=np.float64)

    # Time remaining
    tau = np.clip(T - float(t), 0.0, T)

    # Inventory risk term
    # S = gamma * sigma^2 * q * tau
    S = gamma * sigma * sigma * q * tau

    # Analytic optimal depths from HJB/First Order Conditions
    # d_bid = 1/k + S
    # d_ask = 1/k - S
    
    base_depth = 1.0 / k
    
    bid = base_depth + S
    ask = base_depth - S

    # Constraints
    # Valid range: (-3/k, Q]
    # If depth <= -3/k, it is invalid. We clip to just above -3/k.
    # If depth > Q, it is effectively "no quote" or very wide. 
    # The problem says "distances ... np.inf means no quote".
    # Usually, if we want to not quote, we set it to a very large number or inf.
    # However, the objective rewards depth earned. 
    # If we set it to Q, we are quoting at the boundary.
    # If the optimal depth is > Q, we should quote at Q? 
    # Or should we quote at inf?
    # "np.inf means no quote".
    # If we quote at Q, we might get a fill.
    # If the optimal is > Q, it means we want to be very passive.
    # Quoting at Q is the most passive allowed finite quote.
    # So clip to Q.
    
    lo = -3.0 / k
    hi = float(Q)
    
    # Clip to valid range
    bid = np.clip(bid, lo, hi)
    ask = np.clip(ask, lo, hi)

    # Boundary conditions
    # No bid fill at q = Q
    bid = np.where(q >= Q, np.inf, bid)
    # No ask fill at q = -Q
    ask = np.where(q <= -Q, np.inf, ask)

    # Final validity check
    # Ensure no NaN or Inf in the finite parts
    bid = np.where(np.isfinite(bid) & (bid > lo), bid, np.inf)
    ask = np.where(np.isfinite(ask) & (ask > lo), ask, np.inf)

    return bid, ask
# EVOLVE-BLOCK-END
