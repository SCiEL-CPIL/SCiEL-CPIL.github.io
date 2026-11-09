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

    # Inventory skew coefficient
    # S = gamma * sigma^2 * q * tau
    S = gamma * sigma * sigma * q * tau

    # Optimal depth for the "active" side (side we want to fill)
    # d_active = 1/k - sign(q) * |S| ?
    # If q > 0, we sell (Ask). d_a = 1/k - S.
    # If q < 0, we buy (Bid). d_b = 1/k + S (since S is negative, this is 1/k - |S|).
    # So d_active = 1/k - S * sign(q)? No.
    # If q > 0, S > 0. d_a = 1/k - S.
    # If q < 0, S < 0. d_b = 1/k - S (wait, S is negative, so 1/k - (-|S|) = 1/k + |S|).
    # Let's check the formula for q < 0.
    # C = |S|. d_b = C - 1/k?
    # My previous derivation: d_b = C - 1/k.
    # If C is small, d_b is negative.
    # If C is large, d_b is positive.
    
    # Let's re-verify the "active" side formula.
    # Maximize e^{-kd} (d + C) for Ask (q>0). Peak at 1/k - C.
    # Maximize e^{-kd} (C - d) for Bid (q<0). Peak at C - 1/k.
    
    # So:
    # If q > 0:
    #   d_ask = 1/k - S
    #   d_bid = Q (passive)
    # If q < 0:
    #   d_bid = -S - 1/k  (since S is negative, -S is positive C. C - 1/k)
    #   d_ask = Q (passive)
    # If q == 0:
    #   d_bid = 1/k
    #   d_ask = 1/k

    lo = -3.0 / k
    hi = float(Q)
    
    # Initialize depths
    bid = np.full(n, hi, dtype=np.float64)
    ask = np.full(n, hi, dtype=np.float64)
    
    # Mask for q > 0
    mask_pos = q > 0
    # Mask for q < 0
    mask_neg = q < 0
    # Mask for q == 0
    mask_zero = q == 0
    
    # For q > 0:
    # Ask is active: d_a = 1/k - S
    # Bid is passive: d_b = Q
    if np.any(mask_pos):
        d_a_pos = (1.0 / k) - S[mask_pos]
        # Clip to valid range
        d_a_pos = np.clip(d_a_pos, lo, hi)
        ask[mask_pos] = d_a_pos
        bid[mask_pos] = hi # Passive
        
    # For q < 0:
    # Bid is active: d_b = -S - 1/k
    # Ask is passive: d_a = Q
    if np.any(mask_neg):
        d_b_neg = -S[mask_neg] - (1.0 / k)
        # Clip to valid range
        d_b_neg = np.clip(d_b_neg, lo, hi)
        bid[mask_neg] = d_b_neg
        ask[mask_neg] = hi # Passive
        
    # For q == 0:
    # Both active/symmetric
    if np.any(mask_zero):
        d_zero = 1.0 / k
        bid[mask_zero] = d_zero
        ask[mask_zero] = d_zero
        
    # Boundary constraints
    # No bid fill at q = Q
    bid = np.where(q >= Q, np.inf, bid)
    # No ask fill at q = -Q
    ask = np.where(q <= -Q, np.inf, ask)
    
    # Guarantee validity
    bid = np.where(np.isfinite(bid) & (bid > lo), bid, np.inf)
    ask = np.where(np.isfinite(ask) & (ask > lo), ask, np.inf)
    
    return bid, ask
# EVOLVE-BLOCK-END
