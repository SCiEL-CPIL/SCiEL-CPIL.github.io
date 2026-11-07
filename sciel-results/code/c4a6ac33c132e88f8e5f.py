"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 290  # Safety margin before 330s kill


def quote(t, inventory, mid, params):
    """
    Inventory-aware quoting with adaptive depth adjustment.
    
    Strategy:
    - Base depth from 1/k (risk-neutral)
    - Adjust based on inventory: widen in direction that increases |q|,
      tighten in direction that reduces |q|
    - Use a nonlinear (tanh-like) adjustment to handle large inventories
    - Scale adjustment with remaining time and volatility
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    dt = T / steps
    remaining_time = T - t
    time_fraction = remaining_time / T  # 1 at start, 0 at end
    
    q = inventory.astype(np.float64)
    
    # Base depth: risk-neutral level
    base_depth = 1.0 / k
    
    # Inventory adjustment strength:
    # - Scale with gamma (higher penalty -> stronger adjustment)
    # - Scale with sigma (higher vol -> more risk -> stronger adjustment)
    # - Scale with remaining time (more time to manage inventory)
    # - Normalize by Q so adjustment is consistent across regimes
    adj_scale = gamma * sigma * time_fraction / (Q + 1.0)
    
    # Nonlinear adjustment: use tanh to saturate at large |q|
    # This prevents depths from going negative
    q_normalized = q / Q
    adjustment = np.tanh(2.0 * q_normalized)  # saturates at +/-1
    
    # Apply adjustment:
    # When q > 0 (long position):
    #   - bid_depth should increase (less likely to buy more)
    #   - ask_depth should decrease (more likely to sell)
    # When q < 0 (short position):
    #   - bid_depth should decrease (more likely to buy to reduce short)
    #   - ask_depth should increase (less likely to sell more)
    bid_depth = base_depth + adj_scale * adjustment
    ask_depth = base_depth - adj_scale * adjustment
    
    # Ensure minimum positive depth (avoid numerical issues and negative depths)
    min_depth = 0.005
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # Cap maximum depth to avoid extremely wide quotes
    max_depth = 5.0 / k
    bid_depth = np.minimum(bid_depth, max_depth)
    ask_depth = np.minimum(ask_depth, max_depth)
    
    # At boundaries, no quote on the side that would push inventory further
    bid_depth[inventory >= Q] = np.inf
    ask_depth[inventory <= -Q] = np.inf
    
    # Safety check for deadline
    if time.time() > _DEADLINE:
        # Return simple fallback
        depth = np.full(len(inventory), base_depth)
        depth[inventory >= Q] = np.inf
        depth2 = np.full(len(inventory), base_depth)
        depth2[inventory <= -Q] = np.inf
        return depth, depth2
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
