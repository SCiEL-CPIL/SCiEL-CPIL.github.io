"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Optimal market making with inventory risk using a quadratic value function approximation.
    
    Improved derivation based on HJB equation for V(t, q) = -a(t) q^2 + c(t).
    a(t) = 0.5 * gamma * sigma^2 * (T - t)
    
    Optimal depths:
    d_bid = 1/k + a(t) * (2*q + 1)
    d_ask = 1/k + a(t) * (1 - 2*q)
    
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
    
    # Risk coefficient a(t)
    a_t = 0.5 * gamma * (sigma ** 2) * time_left
    
    # Calculate optimal depths
    # d_bid = 1/k + a_t * (2*q + 1)
    # d_ask = 1/k + a_t * (1 - 2*q)
    
    bid_depths = base_depth + a_t * (2.0 * q + 1.0)
    ask_depths = base_depth + a_t * (1.0 - 2.0 * q)
    
    # Clamp depths to be non-negative
    # If calculated depth is negative, it implies we strongly prefer a fill on that side,
    # so we quote as close to mid as possible (depth 0).
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
