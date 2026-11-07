"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 290  # Safety margin before 330s kill


def quote(t, inventory, mid, params):
    """Inventory-aware quoting: adjust depth based on current inventory."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    q = inventory.astype(np.float64)
    base_depth = 1.0 / k
    
    # Inventory adjustment: widen quotes in direction that increases |q|
    # When q > 0 (long): bid wider (less likely to fill), ask tighter (more likely to fill)
    # When q < 0 (short): bid tighter, ask wider
    beta = 0.02
    bid_depth = base_depth + beta * q
    ask_depth = base_depth - beta * q
    
    # Clamp to minimum positive depth to avoid numerical issues
    min_depth = 0.01
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # At boundaries, no quote on the side that would push inventory further
    bid_depth[inventory >= Q] = np.inf
    ask_depth[inventory <= -Q] = np.inf
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
