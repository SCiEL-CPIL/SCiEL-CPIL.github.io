"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = None

def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy based on HJB / quadratic value function.
    
    The optimal policy for this discrete model can be approximated by a quadratic
    value function V(t, q) = a(t) + b(t)*q + (gamma/2)*sigma^2*t_rem*q^2 + ...
    The optimal depth balances marginal fill profit against marginal inventory risk.
    
    For a bid quote at depth d_bid:
      - Fill probability ~ 1 - exp(-A*exp(-k*d_bid)*dt)
      - If filled: gain d_bid, inventory increases by 1
      - Risk cost: increase in expected penalty from inventory change
    
    The first-order condition gives:
      d_bid = (1/k) * log( A*dt / (1 - gamma*sigma^2*t_rem*q) )  [for small dt]
      d_ask = (1/k) * log( A*dt / (1 + gamma*sigma^2*t_rem*q) )
    
    But we need to handle the case where the argument becomes non-positive.
    Also, we should add a base depth to ensure positive spreads.
    
    Refined formula:
      Let x = gamma * sigma^2 * t_rem * q
      d_bid = (1/k) * log( c / (1 - x) )
      d_ask = (1/k) * log( c / (1 + x) )
    where c is a calibration constant.
    
    From the HJB analysis, c should be related to A*dt and the risk parameter.
    For the given regimes, we calibrate c to match the optimal single-step behavior.
    
    Key insight: the optimal depth when inventory is zero should balance the
    expected fill profit against the risk. For small dt:
      Optimal y = k*d maximizes y*(1-exp(-y)) - gamma*sigma^2*dt*(1+y^2/2)
    For very small dt, this approaches y* = 1.5936 (the static optimum).
    But for larger effective dt or higher gamma, the optimal y decreases.
    
    We use a time-varying calibration that accounts for the risk penalty.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are very close to the limit, return a safe fallback
    if time.time() - _start_time > 325.0:
        k = params['k']
        base = 1.0 / k
        return np.full(len(inventory), base), np.full(len(inventory), base)
    
    N = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Time remaining
    t_rem = T - t
    
    # Inventory as float
    inv_float = inventory.astype(np.float64)
    
    # Compute the skew argument
    x = gamma * sigma * sigma * t_rem * inv_float
    
    # Calibration constant c:
    # For the static single-step problem with no risk (gamma=0),
    # optimal y solves: (1-exp(-y)) - y*exp(-y) = 0 => y* = 1.5936
    # With risk, the effective constant is reduced.
    # We use c = exp(y_eff) where y_eff accounts for risk.
    # y_eff = y* - gamma*sigma^2*dt*(something)
    # For simplicity, we use a time-varying base that starts at the static optimum
    # and adjusts for the cumulative risk effect.
    
    # The key term in the HJB is:
    # dV/dt + A*exp(-k*d)*(d - V_q) + ... = 0
    # where V_q = gamma*sigma^2*t_rem*q + b(t)
    # At q=0, b(t) is small, so the main correction is from the risk term.
    
    # Empirical calibration: use y_base that depends on the regime parameters.
    # For high gamma and sigma, we want wider spreads (larger y_base).
    # For low gamma and sigma, we want tighter spreads (smaller y_base).
    
    # The risk penalty per step is ~ gamma * sigma^2 * dt * q^2
    # At q=1, this is gamma * sigma^2 * dt
    # The fill profit is ~ A * exp(-k*d) * d
    # Balancing: A * exp(-y) * (y/k) ~ gamma * sigma^2 * dt
    # => exp(-y) ~ gamma * sigma^2 * dt * k / A
    # => y ~ log(A / (gamma * sigma^2 * dt * k))
    
    # But this is only valid when the argument is positive and reasonable.
    # We clamp y to a reasonable range.
    
    risk_term = gamma * sigma * sigma * dt * k / A
    if risk_term > 0:
        y_from_risk = np.log(1.0 / risk_term)
    else:
        y_from_risk = 1.5936  # static optimum
    
    # Clamp y_from_risk to reasonable bounds
    y_from_risk = np.clip(y_from_risk, 0.5, 3.0)
    
    # Blend with the static optimum for stability
    # When t_rem is large, the risk accumulates, so we want wider spreads
    # When t_rem is small, we can be tighter
    t_factor = t_rem / T  # 1 at start, 0 at end
    
    # Effective base y: interpolate between static optimum and risk-adjusted
    y_base = 1.5936 * (1 - 0.5 * t_factor) + y_from_risk * 0.5 * t_factor
    
    # Also ensure y_base doesn't get too small
    y_base = max(y_base, 0.8)
    
    c = np.exp(y_base)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_bid = 1.0 - x
        denom_ask = 1.0 + x
        
        raw_bid = np.log(c / denom_bid) / k
        raw_ask = np.log(c / denom_ask) / k
    
    # Fix invalid values: if denom <= 0, depth should be inf (no quote)
    bid_depths = np.where(denom_bid <= 0, np.inf, raw_bid)
    ask_depths = np.where(denom_ask <= 0, np.inf, raw_ask)
    
    # Clamp to valid range: depth > -3/k
    min_depth = -3.0 / k + 1e-9
    max_depth = 100.0 / k
    
    # Only clamp finite values; leave inf as inf
    finite_bid = np.isfinite(bid_depths)
    finite_ask = np.isfinite(ask_depths)
    
    if np.any(finite_bid):
        bid_depths[finite_bid] = np.clip(bid_depths[finite_bid], min_depth, max_depth)
    if np.any(finite_ask):
        ask_depths[finite_ask] = np.clip(ask_depths[finite_ask], min_depth, max_depth)
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    bid_depths[at_max_inventory] = np.inf
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure no NaN or -inf remain (safety net)
    bad_bid = np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths <= -3.0/k)
    bad_ask = np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths <= -3.0/k)
    
    if np.any(bad_bid):
        bid_depths[bad_bid] = min_depth
    if np.any(bad_ask):
        ask_depths[bad_ask] = min_depth
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
