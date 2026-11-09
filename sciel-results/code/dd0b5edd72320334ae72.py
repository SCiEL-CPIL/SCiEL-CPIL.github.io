"""Optimal market making with inventory risk via backward induction."""
import numpy as np
import time

def quote(t, inventory, mid, params):
    """
    Compute optimal bid and ask depths for each path.
    
    Uses a pre-computed lookup table from backward induction over the
    discrete inventory state space, with depth optimization via first-order
    conditions of the Bellman equation.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # Time remaining in steps
    n_current = int(round(t / dt))
    n_remaining = steps - n_current
    
    # Use pre-computed or on-the-fly value function
    # For efficiency, we compute optimal depths based on inventory and time remaining
    
    # The key insight: optimal depth balances marginal expected profit vs. inventory risk
    # For a given inventory q, the optimal bid depth b and ask depth a satisfy:
    #   Marginal expected fill profit = marginal inventory risk cost
    
    # Let's derive: if we quote bid at depth b, expected immediate gain from bid fill:
    #   P_bid * (b - delta_V) where delta_V is the change in value from q -> q+1
    #   P_bid = 1 - exp(-A * exp(-k*b) * dt)
    #   For small dt: P_bid ≈ A * exp(-k*b) * dt
    #   Expected gain ≈ A * exp(-k*b) * dt * (b - V(q+1) + V(q))
    #   
    # Inventory penalty per step: (gamma/2) * sigma^2 * q^2 * dt
    # 
    # The value function V(q, n) satisfies:
    # V(q, n) = max_{b,a} [ (gamma/2)*sigma^2*q^2*dt  (cost)
    #                         + P_bid*(b - (V(q+1,n-1) - V(q,n-1)))
    #                         + P_ask*(a - (V(q-1,n-1) - V(q,n-1)))
    #                         + V(q, n-1) ]
    #
    # For the optimal b, taking derivative wrt b:
    # d/db [A*exp(-k*b)*dt*(b - dV)] = 0
    # => -k*exp(-k*b)*dt*(b - dV) + exp(-k*b)*dt = 0
    # => -k*(b - dV) + 1 = 0
    # => b = dV + 1/k where dV = V(q+1,n-1) - V(q,n-1)
    #
    # Similarly for ask: a = -dV_ask + 1/k where dV_ask = V(q-1,n-1) - V(q,n-1)
    #
    # But we need to be careful about signs. Let me re-derive.
    
    # Actually, let me think about this more carefully with the value function.
    # V(q, n) = maximum expected objective from step n to end, given inventory q at step n.
    # 
    # At step n, we choose bid_depth b >= 0 and ask_depth a >= 0.
    # Bid fill prob: p_b = 1 - exp(-A * exp(-k*b) * dt)
    # Ask fill prob: p_a = 1 - exp(-A * exp(-k*a) * dt)
    #
    # If bid fills: earn b, inventory becomes q+1 (if q < Q)
    # If ask fills: earn a, inventory becomes q-1 (if q > -Q)
    #
    # The immediate cost is (gamma/2) * sigma^2 * q^2 * dt
    #
    # Bellman: V(q,n) = max_{b,a} { -gamma/2*sigma^2*q^2*dt
    #     + p_b * [b + V(min(q+1,Q), n-1)]
    #     + p_a * [a + V(max(q-1,-Q), n-1)]
    #     + (1-p_b)*(1-p_a) * V(q, n-1) }
    #
    # This is complex. For a practical approach, let's use an approximate
    # analytic solution based on the marginal value of inventory.
    
    # Approximation: Let v(q) be the marginal value of inventory.
    # The optimal bid depth: b* = max(0, 1/k + v(q+1) - v(q))  [shifted by inventory cost]
    # The optimal ask depth: a* = max(0, 1/k - (v(q) - v(q-1)))
    
    # For a quadratic penalty, the value function is approximately quadratic in q:
    # V(q) ≈ c0 + c1*q + c2*q^2
    # Then v(q) = c1 + 2*c2*q
    # 
    # The inventory penalty per step is (gamma/2)*sigma^2*q^2*dt
    # Over n_remaining steps, total penalty ≈ (gamma/2)*sigma^2*q^2*dt*n_remaining
    # So c2 ≈ -(gamma/2)*sigma^2*dt*n_remaining
    #
    # But this is getting complex. Let me use a simpler heuristic that captures
    # the key physics:
    #
    # 1. Base depth: 1/k (risk-neutral)
    # 2. Inventory adjustment: shift quotes to reduce inventory
    #    - If q > 0 (long), make bid less attractive (wider) or pull, make ask more attractive
    #    - If q < 0 (short), opposite
    # 3. Time adjustment: near the end, be more conservative
    
    # The inventory risk cost of being at q for one more step is:
    # dR = gamma * sigma^2 * q * dt  (marginal cost of one more unit of inventory)
    #
    # The expected gain from a bid fill is approximately A * exp(-k*b) * dt * b
    # Setting marginal gain = marginal risk:
    # A * exp(-k*b) * dt * (1 - k*b) ≈ gamma * sigma^2 * q * dt
    #
    # For simplicity, let's use a linear approximation in inventory:
    # b_opt ≈ 1/k - (gamma * sigma^2 * q) / (A * k)  [clamped to valid range]
    # a_opt ≈ 1/k + (gamma * sigma^2 * q) / (A * k)
    
    # But we also need to account for time remaining. Near the end, there's less
    # time to reduce inventory, so we should be more conservative.
    
    # Let me implement a proper backward induction for the value function.
    
    # Pre-compute value function if not cached
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    
    if not hasattr(quote, '_vfunc_cache'):
        quote._vfunc_cache = {}
    
    if cache_key not in quote._vfunc_cache:
        # Compute value function via backward induction
        # V[q_idx][n] where q_idx = q + Q (0 to 2Q)
        n_levels = 2 * Q + 1
        V = np.zeros((n_levels, steps + 1), dtype=np.float64)
        
        # Terminal condition: no more steps, no terminal charge
        # V[:, steps] = 0 (already initialized)
        
        for n in range(1, steps + 1):
            for q_idx in range(n_levels):
                q = q_idx - Q
                # Inventory penalty for this step
                penalty = 0.5 * gamma * sigma * sigma * q * q * dt
                
                best_val = -np.inf
                # We need to optimize over b and a. For efficiency, 
                # use the first-order condition approximation.
                # 
                # Let's compute the value for the optimal depths using the
                # marginal value approach.
                
                # Marginal value of inventory at this state
                if q < Q:
                    dv_bid = V[q_idx + 1, n - 1] - V[q_idx, n - 1]
                else:
                    dv_bid = 0.0  # can't go above Q
                
                if q > -Q:
                    dv_ask = V[q_idx - 1, n - 1] - V[q_idx, n - 1]
                else:
                    dv_ask = 0.0  # can't go below -Q
                
                # Optimal bid depth from FOC: b* = 1/k - dv_bid (if positive)
                # Optimal ask depth from FOC: a* = 1/k + dv_ask (if positive)
                # But we need to be careful with the signs.
                
                # Let me re-derive properly.
                # V(q,n) = max_{b,a} { penalty_cost + E[fill_gains] + E[future_V] }
                # 
                # For small dt, p_b ≈ A*exp(-k*b)*dt, p_a ≈ A*exp(-k*a)*dt
                # 
                # V(q,n) ≈ -penalty + p_b*(b + V(q+1,n-1)) + p_a*(a + V(q-1,n-1))
                #          + (1-p_b)*(1-p_a)*V(q,n-1)
                #
                # ≈ -penalty + A*exp(-k*b)*dt*(b + V(q+1,n-1) - V(q,n-1))
                #          + A*exp(-k*a)*dt*(a + V(q-1,n-1) - V(q,n-1))
                #          + V(q,n-1)
                #
                # Let dV_b = V(q+1,n-1) - V(q,n-1), dV_a = V(q-1,n-1) - V(q,n-1)
                #
                # V(q,n) = V(q,n-1) - penalty + A*dt*exp(-k*b)*(b + dV_b) + A*dt*exp(-k*a)*(a + dV_a)
                #
                # FOC for b: d/db [exp(-k*b)*(b + dV_b)] = 0
                # => -k*exp(-k*b)*(b + dV_b) + exp(-k*b) = 0
                # => -k*(b + dV_b) + 1 = 0
                # => b = 1/k - dV_b
                #
                # Similarly: a = 1/k - dV_a
                #
                # But dV_a = V(q-1,n-1) - V(q,n-1), so a = 1/k - dV_a = 1/k + (V(q,n-1) - V(q-1,n-1))
                
                b_opt = 1.0 / k - dv_bid
                a_opt = 1.0 / k - dv_ask
                
                # Clamp to valid range: depth must be >= some minimum (to avoid -inf issues)
                # The problem says values below -3/k are invalid, but depth should be positive
                # for a meaningful quote. np.inf means no quote.
                
                # If b_opt <= 0, we might still want to quote at a small depth or not quote at all
                # Let's allow small positive depths but cap them.
                
                # Compute the value with these optimal depths
                if b_opt > 0:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                else:
                    p_b = 0.0
                    b_opt = 0.0
                
                if a_opt > 0:
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                else:
                    p_a = 0.0
                    a_opt = 0.0
                
                # Future value
                if q < Q:
                    V_next_bid = V[q_idx + 1, n - 1]
                else:
                    V_next_bid = V[q_idx, n - 1]  # no fill possible
                
                if q > -Q:
                    V_next_ask = V[q_idx - 1, n - 1]
                else:
                    V_next_ask = V[q_idx, n - 1]  # no fill possible
                
                V_next_same = V[q_idx, n - 1]
                
                expected_future = (p_b * V_next_bid + p_a * V_next_ask + 
                                  (1 - p_b) * (1 - p_a) * V_next_same)
                
                fill_gains = p_b * b_opt + p_a * a_opt
                
                V[q_idx, n] = -penalty + fill_gains + expected_future
        
        quote._vfunc_cache[cache_key] = V
    
    V = quote._vfunc_cache[cache_key]
    
    # Now compute optimal depths for each path
    n_levels = 2 * Q + 1
    
    # Get inventory indices
    inv_idx = np.clip(inventory + Q, 0, n_levels - 1).astype(int)
    
    # Current step index
    n_idx = min(n_current, steps)
    
    # Compute marginal values
    # For bid: dv_bid = V(q+1, n-1) - V(q, n-1)
    # For ask: dv_ask = V(q-1, n-1) - V(q, n-1)
    
    # We need V at step n-1 (previous step) for the marginal values
    if n_current > 0:
        n_prev = n_current - 1
    else:
        n_prev = 0  # use step 0 as base
    
    # Actually, we should use the value function at the current decision point.
    # At step n, we're deciding quotes for this step, so we look at V(q, n) vs V(q, n-1).
    # The marginal value is from the next step's value.
    
    # Let me re-think: at step n (0-indexed), we have n_remaining = steps - n steps left.
    # V[q, n] is the value with n steps remaining.
    # So at the start (n=0), we have steps steps remaining, so we use V[:, steps].
    # After one step, we have steps-1 steps remaining, so V[:, steps-1].
    
    # Wait, I indexed V as V[q_idx, n] where n is the step index (0 to steps).
    # V[:, 0] = terminal (no steps left) = 0
    # V[:, n] = value with n steps remaining? Or value at step n?
    
    # Let me re-index: V[q_idx, n] = optimal value from state q with n steps remaining.
    # V[:, 0] = 0 (no steps left)
    # V[:, n] for n = 1, ..., steps
    
    # At the beginning (step 0), we have steps steps remaining, so we use V[:, steps].
    # At step m, we have steps - m steps remaining, so we use V[:, steps - m].
    
    # I computed V[q_idx, n] for n = 1 to steps, where n is the number of steps remaining.
    # So V[q_idx, n_remaining] gives the value.
    
    n_rem = steps - n_current
    
    # Get marginal values
    # dv_bid = V(q+1, n_rem) - V(q, n_rem)  [but this is the value difference, not the marginal cost]
    
    # Actually, from the derivation:
    # b_opt = 1/k - (V(q+1, n_rem-1) - V(q, n_rem-1))
    # where n_rem-1 is the number of steps remaining AFTER this step.
    
    if n_current < steps:
        n_after = steps - (n_current + 1)  # steps remaining after this step
    else:
        n_after = 0
    
    # For the current step, we use the value function at n_after steps remaining
    # to compute the marginal values.
    
    # But wait, in my backward induction, V[q_idx, n] is the value with n steps remaining.
    # At step n_current, before making a decision, we have n_rem = steps - n_current steps remaining.
    # After this step, we'll have n_rem - 1 steps remaining.
    # The marginal value should be computed from V[:, n_rem - 1].
    
    # Let me just use n_after directly.
    
    if n_after > 0:
        V_current = V[:, n_after]
    else:
        V_current = np.zeros(n_levels)
    
    # Compute dv_bid and dv_ask for each inventory level
    dv_bid = np.zeros(n_levels)
    dv_ask = np.zeros(n_levels)
    
    for q_idx in range(n_levels):
        q = q_idx - Q
        if q < Q:
            dv_bid[q_idx] = V_current[q_idx + 1] - V_current[q_idx]
        else:
            dv_bid[q_idx] = 0.0
        
        if q > -Q:
            dv_ask[q_idx] = V_current[q_idx - 1] - V_current[q_idx]
        else:
            dv_ask[q_idx] = 0.0
    
    # Optimal depths
    b_opt = 1.0 / k - dv_bid[inv_idx]
    a_opt = 1.0 / k - dv_ask[inv_idx]
    
    # Clamp: depths must be positive for a valid quote. If <= 0, use a small positive value
    # or set to inf (no quote). The problem says np.inf means no quote.
    # Let's use a small epsilon to avoid invalid values, but allow inf for no quote.
    
    min_depth = 1e-6  # minimum positive depth
    max_depth = 10.0 / k  # maximum reasonable depth
    
    bid_depths = np.where(b_opt > min_depth, np.clip(b_opt, min_depth, max_depth), np.inf)
    ask_depths = np.where(a_opt > min_depth, np.clip(a_opt, min_depth, max_depth), np.inf)
    
    # Ensure valid: values below -3/k are invalid, but our depths are positive or inf
    # Also ensure no NaN or -inf
    bid_depths = np.nan_to_num(bid_depths, nan=np.inf, posinf=np.inf, neginf=np.inf)
    ask_depths = np.nan_to_num(ask_depths, nan=np.inf, posinf=np.inf, neginf=np.inf)
    
    return bid_depths, ask_depths
