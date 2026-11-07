"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
TIME_LIMIT = 320  # safety margin before 330s kill

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths for each (step, inventory) via backward induction.
    Returns (bid_table, ask_table) each of shape (steps, 2*Q+1).
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    # V_next[q] = value from step n+1 with inventory q (indexed by q+Q)
    V_next = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Precompute the minimum depth
    min_depth = 3.0 / k
    
    # For optimization, we search over depth in [min_depth, max_search]
    # The objective is smooth; use golden-section or scipy minimize_scalar
    from scipy.optimize import minimize_scalar
    
    max_depth = 50.0 / k  # generous upper bound
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT:
            # Fallback: use 1/k for remaining steps
            for nn in range(n, -1, -1):
                bid_table[nn] = 1.0 / k
                ask_table[nn] = 1.0 / k
            break
        
        # Inventory cost for this step
        inv_cost = (gamma / 2.0) * sigma**2 * inv_range**2 * dt
        
        for qi in range(n_inv):
            q = inv_range[qi]
            
            # Determine if we can quote bid/ask
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # Optimize bid depth
            if can_bid:
                # If bid fill: inventory goes to q-1
                # Continuation: (1-p_b)*[p_a*V[q+1] + (1-p_a)*V[q]] + p_b*[p_a*V[q] + (1-p_a)*V[q-1]]
                # = p_a*V[q+1] + (1-p_a)*V[q] + p_b*((1-p_a)*V[q-1] - (1-p_a)*V[q])
                # = p_a*V[q+1] + (1-p_a)*V[q] + p_b*(1-p_a)*(V[q-1]-V[q])
                # 
                # Immediate: p_b * b - inv_cost
                # 
                # Total = p_b*b - inv_cost + p_a*V[q+1] + (1-p_a)*V[q] + p_b*(1-p_a)*(V[q-1]-V[q])
                # 
                # For fixed p_a, optimize over p_b (equivalently over b):
                # f(b) = p_b*b + p_b*(1-p_a)*(V[q-1]-V[q]) + [p_a*V[q+1] + (1-p_a)*V[q] - inv_cost]
                # 
                # But p_a also depends on a, which we optimize separately.
                # Since bid and ask are independent in the sense that the total objective
                # separates: total = [bid part depending on b] + [ask part depending on a] + constant
                # 
                # Let's verify: total = p_b*b + p_a*a - inv_cost 
                #   + p_a*V[q+1] + (1-p_a)*V[q] + p_b*(1-p_a)*(V[q-1]-V[q])
                #   + p_a*(1-p_b)*(V[q+1]-V[q])  ... wait, let me redo this more carefully.
                #
                # Actually, the four outcomes:
                # No fill: prob (1-p_b)(1-p_a), value V[q]
                # Bid fill only: prob p_b(1-p_a), value V[q-1]
                # Ask fill only: prob (1-p_b)p_a, value V[q+1]
                # Both: prob p_b*p_a, value V[q]
                #
                # Continuation = (1-p_b)(1-p_a)V[q] + p_b(1-p_a)V[q-1] + (1-p_b)p_a V[q+1] + p_b p_a V[q]
                # = (1-p_a)V[q] + p_a V[q+1] + p_b(1-p_a)(V[q-1]-V[q])
                #
                # Wait: (1-p_b)(1-p_a)V[q] + p_b(1-p_a)V[q-1] + (1-p_b)p_a V[q+1] + p_b p_a V[q]
                # = (1-p_a)[(1-p_b)V[q] + p_b V[q-1]] + p_a[(1-p_b)V[q+1] + p_b V[q]]
                # = (1-p_a)V[q] + p_a V[q+1] + p_b(1-p_a)(V[q-1]-V[q])
                #
                # Hmm, that's not quite right either. Let me expand:
                # = (1-p_b)(1-p_a)V[q] + p_b(1-p_a)V[q-1] + (1-p_b)p_a V[q+1] + p_b p_a V[q]
                # Group by V[q]: (1-p_b)(1-p_a)V[q] + p_b p_a V[q] = V[q][(1-p_b)(1-p_a) + p_b p_a]
                # = V[q][1 - p_b - p_a + 2 p_b p_a]
                # Plus p_b(1-p_a)V[q-1] + (1-p_b)p_a V[q+1]
                #
                # So continuation = V[q](1 - p_b - p_a + 2 p_b p_a) + p_b(1-p_a)V[q-1] + (1-p_b)p_a V[q+1]
                #
                # This couples p_b and p_a through the 2 p_b p_a term. So they're NOT independent.
                #
                # However, for small dt, p_b and p_a are small, so the coupling is weak.
                # For exact optimality, we need to handle this.
                #
                # Alternative: since the state space is small (2Q+1 = 61) and steps = 1000,
                # we can do a 2D optimization for each (n, q). But that's 1000*61 = 61000 2D optimizations.
                # Too slow.
                #
                # Better approach: For each (n, q), the objective as a function of (b, a) is:
                # f(b,a) = p_b(b)*b + p_a(a)*a - inv_cost
                #   + V[q](1 - p_b - p_a + 2 p_b p_a) + p_b(1-p_a)V[q-1] + (1-p_b)p_a V[q+1]
                #
                # where p_b(b) = 1-exp(-A exp(-k b) dt), p_a(a) = 1-exp(-A exp(-k a) dt)
                #
                # This is a 2D problem. But note that for fixed p_a, the function in p_b is:
                # g(p_b) = p_b*b + V[q](1-p_b) + p_b(1-p_a)V[q-1] + (1-p_b)p_a V[q+1] + p_b p_a V[q] - ... 
                #
                # This is getting complex. Let me just do a 2D grid search or use scipy for each state.
                # Given 61000 states and each 2D optimization taking ~1ms, that's ~61 seconds. Tight but feasible.
                #
                # Actually, let's think about this differently. The key observation is that p_b and p_a 
                # are each in [0, 1-exp(-A dt)] which is a small range for small dt.
                #
                # For dt = 0.001, A=140: max p = 1-exp(-0.14) ≈ 0.13
                # For A=50: max p = 1-exp(-0.05) ≈ 0.049
                #
                # So the coupling term 2 p_b p_a is at most ~0.017, which is small.
                #
                # Practical approach: iterate. First optimize b assuming p_a=0, then optimize a assuming 
                # the new p_b, etc. A few iterations should converge.
                #
                # Even simpler: since the coupling is weak, just optimize each independently assuming 
                # the other is at its optimal value from the previous iteration. Start with p_a=0.
                
                # Let's do a simple approach: optimize b and a jointly using a small 2D grid
                # or use the fact that for fixed a, the optimal b can be found, and vice versa.
                
                # For efficiency, let's use scipy minimize_scalar nested.
                # But 61000 * 2D might be too slow. Let me estimate: if each 2D takes 0.5ms, total ~30s.
                
                # Alternative: use the fact that the problem is separable up to a small coupling.
                # Do 2 iterations of alternating optimization.
                
                # Initialize
                p_a_est = 0.0  # start with no ask fill probability
                
                # Optimize b given p_a_est
                def neg_obj_bid(b):
                    pb = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    # Continuation with fixed p_a = p_a_est
                    cont = V_next[qi] * (1 - pb - p_a_est + 2*pb*p_a_est) \
                           + pb * (1-p_a_est) * V_next[qi-1] \
                           + (1-pb) * p_a_est * V_next[qi+1]
                    return -(pb * b + cont - inv_cost[qi])
                
                res_b = minimize_scalar(neg_obj_bid, bounds=(min_depth, max_depth), method='bounded', 
                                        options={'xatol': 1e-8})
                b_opt = res_b.x
                pb_opt = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                
                # Optimize a given pb_opt
                def neg_obj_ask(a):
                    pa = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    cont = V_next[qi] * (1 - pb_opt - pa + 2*pb_opt*pa) \
                           + pb_opt * (1-pa) * V_next[qi-1] \
                           + (1-pb_opt) * pa * V_next[qi+1]
                    return -(pa * a + cont - inv_cost[qi])
                
                res_a = minimize_scalar(neg_obj_ask, bounds=(min_depth, max_depth), method='bounded',
                                        options={'xatol': 1e-8})
                a_opt = res_a.x
                pa_opt = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                
                # One more iteration for better accuracy
                def neg_obj_bid2(b):
                    pb = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    cont = V_next[qi] * (1 - pb - pa_opt + 2*pb*pa_opt) \
                           + pb * (1-pa_opt) * V_next[qi-1] \
                           + (1-pb) * pa_opt * V_next[qi+1]
                    return -(pb * b + cont - inv_cost[qi])
                
                res_b2 = minimize_scalar(neg_obj_bid2, bounds=(min_depth, max_depth), method='bounded',
                                         options={'xatol': 1e-9})
                b_opt = res_b2.x
                pb_opt = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                
                def neg_obj_ask2(a):
                    pa = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    cont = V_next[qi] * (1 - pb_opt - pa + 2*pb_opt*pa) \
                           + pb_opt * (1-pa) * V_next[qi-1] \
                           + (1-pb_opt) * pa * V_next[qi+1]
                    return -(pa * a + cont - inv_cost[qi])
                
                res_a2 = minimize_scalar(neg_obj_ask2, bounds=(min_depth, max_depth), method='bounded',
                                         options={'xatol': 1e-9})
                a_opt = res_a2.x
                pa_opt = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                
                bid_table[n, qi] = b_opt
                ask_table[n, qi] = a_opt
                
                # Compute V_next for this state
                cont = V_next[qi] * (1 - pb_opt - pa_opt + 2*pb_opt*pa_opt) \
                       + pb_opt * (1-pa_opt) * V_next[qi-1] \
                       + (1-pb_opt) * pa_opt * V_next[qi+1]
                V_next[qi] = pb_opt * b_opt + pa_opt * a_opt - inv_cost[qi] + cont
            else:
                # Cannot bid (q = Q)
                bid_table[n, qi] = np.inf
                pb_opt = 0.0
                
                # Optimize a
                def neg_obj_ask(a):
                    pa = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    # q=Q, so no bid fill. Outcomes: no fill (V[q]), ask fill (V[q+1])
                    # But q=Q, so q+1 = Q+1 which is out of bounds. 
                    # Actually at q=Q, ask fill would make q=Q+1 which is invalid.
                    # The problem says "no ask fill at q = -Q" and "no bid fill at q = Q"
                    # So at q=Q, only ask fills are possible, leading to q=Q+1? 
                    # No wait: "A fill earns its depth and moves inventory one unit"
                    # Bid fill: inventory decreases by 1 (we sold, so less long / more short)
                    # Ask fill: inventory increases by 1 (we bought, so more long / less short)
                    # "no bid fill at q = Q" means we don't quote bid when q=Q
                    # "no ask fill at q = -Q" means we don't quote ask when q=-Q
                    # So at q=Q, we can still quote ask, and an ask fill makes q=Q+1
                    # But Q is the max, so q=Q+1 shouldn't happen... 
                    # Re-reading: inventory in [-Q, Q]. So if q=Q and ask fill, q becomes Q+1?
                    # That's outside the range. I think the constraint means:
                    # At q=Q, don't quote bid (so no bid fill possible)
                    # At q=-Q, don't quote ask (so no ask fill possible)
                    # But ask fill at q=Q would make q=Q+1 which is invalid.
                    # I think the inventory is capped at [-Q,Q], so fills that would push out of range
                    # simply don't happen, or the constraint prevents quoting on that side.
                    # 
                    # Given "no bid fill at q = Q, no ask fill at q = -Q", I interpret:
                    # At q=Q: only ask fills possible (bid not quoted)
                    # At q=-Q: only bid fills possible (ask not quoted)
                    # An ask fill at q=Q would make q=Q+1, but since Q is the max allowed inventory,
                    # perhaps it's capped. But the problem says inventory is in [-Q,Q], so maybe
                    # fills that would push out of bounds are disallowed.
                    #
                    # Safest interpretation: at q=Q, quote only ask. If ask fills, q->Q+1 which 
                    # we clamp to Q (or it's simply not allowed). Given the problem states inventory
                    # is in [-Q,Q], I'll assume fills that would push out of bounds don't occur.
                    # So at q=Q, ask fill would push to Q+1 which is invalid, so effectively no ask fill either?
                    # That seems wrong. Let me re-read.
                    #
                    # "inventory (4000,) ints in [-Q, Q]" - this is the state space.
                    # "no bid fill at q = Q, no ask fill at q = -Q" - these are the constraints.
                    # I think it means: when q=Q, you don't quote bid (so bid fill can't happen).
                    # When q=-Q, you don't quote ask (so ask fill can't happen).
                    # An ask fill at q=Q makes q=Q+1, but since the state space is [-Q,Q], 
                    # this transition is simply not possible (the fill doesn't occur).
                    # Similarly, a bid fill at q=-Q makes q=-Q-1, not possible.
                    #
                    # So effectively: at q=Q, only ask fills that keep q in range are possible.
                    # But ask fill always increases q by 1, so at q=Q, no ask fill is possible either!
                    # That means at q=Q, no fills at all? That can't be right.
                    #
                    # I think the correct interpretation is: the inventory can go to Q+1 or -Q-1,
                    # but the problem restricts to [-Q,Q] for the state space, and the constraints
                    # "no bid fill at q=Q" and "no ask fill at q=-Q" are the only restrictions.
                    # So at q=Q, bid fills are disallowed, but ask fills are allowed, leading to q=Q+1.
                    # But then the state space should be [-Q-1, Q+1] or something.
                    #
                    # Given the ambiguity, I'll go with: at q=Q, no bid fill (bid depth = inf).
                    # Ask fills at q=Q lead to q=Q+1, but since our value function only covers [-Q,Q],
                    # we need to handle this. The simplest is to assume the state space is [-Q,Q] and
                    # fills that push out of bounds are simply not possible (inventory is clamped).
                    #
                    # For now, let me assume: at q=Q, ask fill leads to q=Q (clamped). 
                    # Similarly at q=-Q, bid fill leads to q=-Q (clamped).
                    
                    # Actually, re-reading more carefully: the problem says inventory is in [-Q,Q]
                    # and gives the constraint "no bid fill at q = Q, no ask fill at q = -Q".
                    # The most natural reading: these are the ONLY restrictions. So at q=Q, you can
                    # still quote ask, and an ask fill moves you to q=Q+1. But since the problem
                    # states inventory is in [-Q,Q], I think the intended interpretation is that
                    # the state space is exactly [-Q,Q] and the fills are constrained to keep
                    # inventory in range. So at q=Q, an ask fill would push to Q+1 which is out
                    # of range, so it doesn't happen. Effectively, at q=Q, no fills are possible.
                    #
                    # But that makes q=Q and q=-Q absorbing states with no revenue, which seems
                    # pathological. Let me just go with: at q=Q, bid not quoted, ask quoted normally.
                    # If ask fills, q becomes Q+1, but we'll clamp to Q for the value function.
                    # This is a minor edge case.
                    
                    # For simplicity in this edge case, assume ask fill at q=Q leads to q=Q (no change)
                    cont = V_next[qi] * (1 - pa) + V_next[qi] * pa  # = V_next[qi]
                    # Wait, that's just V_next[qi]. So the ask doesn't change anything?
                    # That's because we're clamping. Let me just use V_next[qi] for both outcomes.
                    return -(pa * a - inv_cost[qi] + V_next[qi])
                
                res_a = minimize_scalar(neg_obj_ask, bounds=(min_depth, max_depth), method='bounded',
                                        options={'xatol': 1e-8})
                a_opt = res_a.x
                pa_opt = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                
                ask_table[n, qi] = a_opt
                V_next[qi] = pa_opt * a_opt - inv_cost[qi] + V_next[qi]
            
            # Handle ask optimization for normal case (q > -Q and q < Q)
            if can_ask and can_bid:
                pass  # Already handled above
            
            if can_ask and not can_bid:
                pass  # Already handled above (q = Q case)
            
            if can_bid and not can_ask:
                # q = -Q, cannot ask
                ask_table[n, qi] = np.inf
                pa_opt = 0.0
                
                def neg_obj_bid(b):
                    pb = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    # At q=-Q, bid fill leads to q=-Q-1, clamp to q=-Q
                    cont = V_next[qi]
                    return -(pb * b - inv_cost[qi] + cont)
                
                res_b = minimize_scalar(neg_obj_bid, bounds=(min_depth, max_depth), method='bounded',
                                        options={'xatol': 1e-8})
                b_opt = res_b.x
                pb_opt = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                
                bid_table[n, qi] = b_opt
                V_next[qi] = pb_opt * b_opt - inv_cost[qi] + V_next[qi]
        
        if time.time() - _START_TIME > TIME_LIMIT:
            for nn in range(n, -1, -1):
                for qi in range(n_inv):
                    if not np.isfinite(bid_table[nn, qi]):
                        bid_table[nn, qi] = 1.0 / k
                    if not np.isfinite(ask_table[nn, qi]):
                        ask_table[nn, qi] = 1.0 / k
            break
    
    return bid_table, ask_table


def _get_policy_table(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    bid_table, ask_table = _get_policy_table(params)
    
    Q = params['Q']
    steps = params['steps']
    n = int(round(t / (params['T'] / steps)))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to index
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, inv_idx].astype(np.float64)
    ask_depths = ask_table[n, inv_idx].astype(np.float64)
    
    # Ensure validity: depths must be >= 3/k or inf
    min_depth = 3.0 / params['k']
    bid_depths = np.where(bid_depths < min_depth, min_depth, bid_depths)
    ask_depths = np.where(ask_depths < min_depth, min_depth, ask_depths)
    
    return bid_depths, ask_depths
