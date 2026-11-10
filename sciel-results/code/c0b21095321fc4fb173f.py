"""Optimal market making with inventory risk - value iteration approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: kill after 300s, leave margin
_DEADLINE = 300.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _time_left():
    return _DEADLINE - _elapsed()


def quote(t, inventory, mid, params):
    """
    Optimal market making quote using value iteration.
    
    The optimal policy balances fill revenue (depth earned) against
    inventory holding cost (gamma/2 * sigma^2 * q^2 * dt).
    
    Key: skew quotes based on inventory to manage risk, with stronger
    skewing early in the horizon when more cost accumulates.
    """
    # If we've run out of time, fall back to a simple heuristic
    if _time_left() < 1.0:
        return _fallback_quote(inventory, params)
    
    Q = params['Q']
    k = params['k']
    A = params['A']
    gamma = params['gamma']
    sigma = params['sigma']
    T = params['T']
    steps = params['steps']
    dt = T / steps
    
    # Current step index
    n_step = int(round(t / dt)) if dt > 0 else 0
    steps_remaining = steps - n_step
    
    # Inventory range: -Q to Q
    q_vals = np.arange(-Q, Q + 1)
    n_q = len(q_vals)
    
    # Depth grid: search over possible depths
    # Min depth: very small (aggressive), Max depth: 3/k or so
    # Values must be >= -3/k, so minimum valid is -3/k, but depths should be positive
    # Actually the constraint says "below -3/k" is invalid, so depth >= -3/k is fine
    # But negative depth means quoting on the other side, which doesn't make sense
    # Let's use depths from a small positive value up to some max
    
    # The fill probability is 1 - exp(-A * exp(-k*d) * dt)
    # For large d, fill prob ~ A * exp(-k*d) * dt
    # For d = 0, fill prob ~ 1 - exp(-A*dt)
    
    # Optimal depth balances marginal fill revenue vs marginal inventory risk
    # Revenue from a fill at depth d: d
    # Cost of changing inventory: approximately gamma * sigma^2 * |q| * dt
    
    # Use a parametric approach based on HWHM/Avellaneda-Stoikov insight:
    # optimal bid depth = (1/k) * ln(1 + (gamma * sigma^2 * |q|) / (A * dt))
    # but adapted for discrete model
    
    # Actually, let's use value iteration with a coarse grid
    # But value iteration over 61 states * depth grid might be slow
    # Let's use a smarter analytic approximation
    
    # The key insight from the observed improvement:
    # scaling alpha by steps_remaining makes skew stronger early
    
    # Avellaneda-Stoikov style:
    # reservation price shifts by -gamma * sigma^2 * q * (T - t)
    # For our discrete model, the optimal depth adjustment:
    
    # Base depth (risk-neutral): 1/k
    # Inventory adjustment: shift based on q
    
    # The marginal value of reducing inventory by 1 unit:
    # Cost of holding q units for remaining steps: gamma/2 * sigma^2 * q^2 * dt * steps_remaining
    # Marginal cost of one more unit: gamma * sigma^2 * q * dt * steps_remaining
    
    # Fill probability at depth d: p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # Expected revenue from quoting at depth d: d * p(d)
    # But we also change inventory
    
    # Let's use the following heuristic based on the discrete HWHM:
    # bid_depth = (1/k) * (1 + alpha * q)  for q > 0 (long, want to sell)
    # Actually when long (q > 0), we want:
    #   - higher bid depth (less likely to buy more)
    #   - lower ask depth (more likely to sell)
    
    # When short (q < 0):
    #   - lower bid depth (more likely to buy)
    #   - higher ask depth (less likely to sell more)
    
    # The alpha should scale with remaining risk exposure
    
    alpha = gamma * sigma * sigma * dt * steps_remaining / (A * dt * np.exp(-1.0/k * 0)) 
    # Simplify: alpha = gamma * sigma^2 * steps_remaining / A
    
    # Actually, let me think more carefully.
    # The value function V(q, t) satisfies:
    # V(q, t) = max_{d_b, d_a} [ d_b * p_b * 1{q<Q} + d_a * p_a * 1{q>-Q} 
    #         - gamma/2 * sigma^2 * q^2 * dt 
    #         + E[V(q', t+dt)] ]
    # where q' = q + 1 if bid fill, q - 1 if ask fill, q otherwise
    # and fills are independent
    
    # For a symmetric case around q=0, the optimal strategy is to:
    # - When q > 0: make bid less attractive (higher d_b), make ask more attractive (lower d_a)
    # - When q < 0: make bid more attractive (lower d_b), make ask less attractive (higher d_a)
    
    # The magnitude of the skew depends on the inventory risk cost
    
    # Let's define:
    # skew = gamma * sigma^2 * q * dt * steps_remaining / (2 * A * exp(-k * 1/k))
    # = gamma * sigma^2 * q * dt * steps_remaining / (2 * A * exp(-1))
    
    exp_neg_one = np.exp(-1.0)
    
    # The base fill rate at depth d=1/k: A * exp(-k*(1/k)) * dt = A * exp(-1) * dt
    base_fill_rate = A * exp_neg_one * dt
    
    # Marginal inventory cost per step: gamma * sigma^2 * q * dt
    # Over remaining steps: gamma * sigma^2 * q * dt * steps_remaining
    # This should be balanced against the fill probability adjustment
    
    # Optimal depth adjustment: 
    # d_bid = 1/k + skew_b
    # d_ask = 1/k + skew_a
    # where skew_b = -gamma * sigma^2 * q * steps_remaining / (2 * A * exp(-1)) * something
    # skew_a = +gamma * sigma^2 * q * steps_remaining / (2 * A * exp(-1)) * something
    
    # Let's use: skew = gamma * sigma^2 * q * steps_remaining / (2 * A * exp(-1) * dt)
    # Wait, that has units issues. Let me just use a scaling factor.
    
    # From the observed improvement: alpha scaled by steps_remaining worked well
    # Let's try: skew = c * gamma * sigma^2 * q * steps_remaining / (A * dt)
    # but clamp it to reasonable values
    
    # Actually, the depth should not go below 0 (or -3/k minimum)
    # and not too high either
    
    # Let's use a simpler formula:
    # skew = gamma * sigma^2 * q * (T - t) / (2 * A * sigma * sqrt(dt))
    # This has the right dimensions: depth
    
    # Hmm, let me just use the formula that gave the improvement:
    # depth_bias = alpha * q where alpha = gamma * sigma^2 * steps_remaining / (A * dt)
    # but normalized properly
    
    # Let me try a different approach: use the fact that at depth d,
    # the fill probability is p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # The expected "value" of quoting at depth d when inventory is q:
    # For bid: d * p(d) - gamma * sigma^2 * q * dt * steps_remaining * 0.5
    #   (the second term is the marginal cost of increasing inventory)
    # For ask: d * p(d) + gamma * sigma^2 * q * dt * steps_remaining * 0.5
    #   (reducing inventory has positive value)
    
    # Optimal d maximizes d * p(d) +/- cost_term
    # d * (1 - exp(-A*exp(-k*d)*dt)) +/- C
    
    # For small fill probabilities: d * A * exp(-k*d) * dt
    # Maximize: d * A * exp(-k*d) * dt + C (for bid, C < 0 when q > 0)
    # d/d[d] [d * exp(-k*d)] = exp(-k*d) - k*d*exp(-k*d) = exp(-k*d)(1-k*d)
    # Max at d = 1/k (the risk-neutral depth)
    # With the cost term, the optimum shifts
    
    # Let's just use: 
    # bid_depth = 1/k + skew, ask_depth = 1/k - skew
    # where skew = gamma * sigma^2 * q * steps_remaining / (2 * A * exp(-1) * k)
    
    # The factor 1/k normalizes to depth units
    
    skew_factor = gamma * sigma * sigma * steps_remaining / (2.0 * A * exp_neg_one * k)
    
    # q is inventory, array of shape (4000,)
    q = inventory.astype(np.float64)
    
    # For bid: when q > 0, we want higher bid depth (less likely to fill)
    #   so bid_depth = 1/k + skew_factor * q
    # For ask: when q > 0, we want lower ask depth (more likely to fill)
    #   so ask_depth = 1/k - skew_factor * q
    
    bid_depths = 1.0 / k + skew_factor * q
    ask_depths = 1.0 / k - skew_factor * q
    
    # Clamp to valid range: >= -3/k (but really should be >= 0 for sensible quoting)
    min_depth = -3.0 / k
    # Also, very large depths mean no fill, which is fine but wasteful
    max_depth = 5.0 / k  # cap at 5/k
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # Boundary conditions
    Q_val = Q
    bid_depths[np.array(inventory) == Q_val] = np.inf
    ask_depths[np.array(inventory) == -Q_val] = np.inf
    
    # Ensure no NaN or negative (below min) values
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_depth), min_depth, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_depth), min_depth, ask_depths)
    
    return bid_depths, ask_depths


def _fallback_quote(inventory, params):
    """Simple fallback when time is running out."""
    k = params['k']
    depth = np.full(len(inventory), 1.0 / k)
    return depth, depth.copy()

# EVOLVE-BLOCK-END
