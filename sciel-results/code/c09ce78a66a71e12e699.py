"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Optimal market making with inventory risk using a quadratic value function approximation.
    
    The optimal depth for bid/ask is derived from balancing the immediate gain (depth)
    against the change in inventory risk penalty.
    
    d_bid = 1/k + gamma * sigma^2 * (T - t) * q
    d_ask = 1/k - gamma * sigma^2 * (T - t) * q
    
    Depths are clamped to be non-negative.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    
    # Time remaining
    time_left = T - t
    
    # Inventory array
    q = np.asarray(inventory, dtype=float)
    
    # Base depth (risk-neutral)
    base_depth = 1.0 / k
    
    # Risk offset coefficient
    # This approximates the marginal change in the value function due to inventory
    # V(q) ~ -0.5 * gamma * sigma^2 * (T-t) * q^2
    # V'(q) ~ -gamma * sigma^2 * (T-t) * q
    risk_offset = gamma * (sigma ** 2) * time_left * q
    
    # Calculate optimal depths
    # If risk_offset is large positive (long inventory), bid depth increases (wider), ask depth decreases (tighter)
    # If risk_offset is large negative (short inventory), bid depth decreases (tighter), ask depth increases (wider)
    
    bid_depths = base_depth + risk_offset
    ask_depths = base_depth - risk_offset
    
    # Clamp depths to be non-negative
    # If calculated depth is negative, it implies we strongly prefer a fill on that side,
    # so we quote as close to mid as possible (depth 0).
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Optional: Cap depths to avoid extremely low probabilities if k is small?
    # The problem doesn't restrict max depth, but np.inf is for no quote.
    # Large depths are fine, they just have low prob.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
