"""Market-making with inventory risk: full DP with exact optimal policy lookup."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 310.0

def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)

_POLICY_CACHE = {}

def _solve_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Solves the full discrete DP to find optimal bid and ask depths for each
    inventory level at each time step.
    Returns (opt_bids, opt_asks) each of shape (steps, 2*Q+1), or None on timeout.
    """
    if _remaining_budget() < 15.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k
    nq = 2 * Q + 1
    qs = np.arange(-Q, Q + 1, dtype=float)

    risk_coeff = 0.5 * gamma * sigma * sigma * dt

    V_next = np.zeros(nq)

    opt_bids = np.zeros((steps, nq))
    opt_asks = np.zeros((steps, nq))

    # Precompute index arrays
    idx = np.arange(nq)
    idx_lo = np.clip(idx - 1, 0, nq - 1)
    idx_hi = np.clip(idx + 1, 0, nq - 1)

    # Precompute exp(-k * b) and exp(-k * a) over a fine grid of depths? 
    # No, we need to optimize over b and a. The analytic solution is b* = (1/k)ln(k*A*dt*dV_bid).
    # Let's verify this is the exact optimum for the given reward structure.
    # Reward: b + a - risk_coeff * q^2
    # Continuation: p_b V(q-1) + p_a V(q+1) + (1-p_b-p_a) V(q)
    # p_b = 1 - exp(-A e^{-kb} dt)
    # Let x = e^{-kb}. p_b = 1 - A x dt.
    # b = -ln(x)/k.
    # Objective for b: -ln(x)/k + (1 - A x dt) V(q-1) + (1 - A x dt) ... wait, the continuation depends on p_b and p_a jointly.
    # The objective is separable in b and a?
    # V(q) = b + a - R + p_b V(q-1) + p_a V(q+1) + (1-p_b-p_a) V(q)
    # = b + a - R + p_b (V(q-1) - V(q)) + p_a (V(q+1) - V(q)) + V(q)
    # Let dV_bid = V(q-1) - V(q), dV_ask = V(q+1) - V(q).
    # V(q) = b + a - R + (1 - e^{-kb} A dt) dV_bid + (1 - e^{-ka} A dt) dV_ask + V(q)
    # The terms involving b are: b + (1 - e^{-kb} A dt) dV_bid = b + dV_bid - A dt e^{-kb} dV_bid.
    # To maximize over b: d/db [b - A dt dV_bid e^{-kb}] = 1 - A dt dV_bid (-k) e^{-kb} = 1 + k A dt dV_bid e^{-kb}.
    # Set to 0: e^{-kb*} = -1 / (k A dt dV_bid).
    # This requires dV_bid < 0. If dV_bid > 0, the term is b + positive * e^{-kb}, which is increasing in b? 
    # Wait, if dV_bid > 0, we want to minimize the penalty, so make p_b small -> b large.
    # If dV_bid < 0, we want to maximize the gain, so make p_b large -> b small.
    # Let's re-evaluate.
    # Term: b + dV_bid - A dt dV_bid e^{-kb}.
    # If dV_bid < 0, let C = -dV_bid > 0. Term: b - C - A dt (-C) e^{-kb} = b - C + A dt C e^{-kb}.
    # Derivative: 1 - A dt C k e^{-kb}. Set to 0: e^{-kb*} = 1/(A dt C k) = 1/(A dt (-dV_bid) k).
    # So b* = (1/k) ln(A dt (-dV_bid) k).
    # This matches the current program's formula: term_b = k * A * dt * dV_bid. If dV_bid < 0, term_b < 0, log is invalid.
    # The current program uses: term_b = k * A * dt * dV_bid. valid_b = term_b > 0.
    # If dV_bid > 0, term_b > 0, b_opt = (1/k) ln(term_b).
    # Let's check the derivative again.
    # f(b) = b + dV_bid (1 - e^{-kb} A dt)
    # f'(b) = 1 + dV_bid A dt k e^{-kb}
    # If dV_bid > 0, f'(b) > 0 always. So f is increasing in b. Max b = max_depth.
    # If dV_bid < 0, let C = -dV_bid. f(b) = b - C + C A dt e^{-kb}.
    # f'(b) = 1 - C A dt k e^{-kb}.
    # If C A dt k e^{-kb} = 1, i.e. e^{-kb} = 1/(C A dt k).
    # If 1/(C A dt k) > 1, i.e. C A dt k < 1, then e^{-kb} > 1 is impossible for b>=0.
    # In this case f'(b) < 0 for all b>=0? No, e^{-kb} <= 1, so C A dt k e^{-kb} <= C A dt k < 1.
    # So f'(b) = 1 - something < 1 > 0. f is increasing. Max b = max_depth.
    # If C A dt k >= 1, then there is a root. b* = (1/k) ln(C A dt k).
    # So the condition for interior optimum is dV_bid < 0 AND (-dV_bid) A dt k >= 1.
    # Otherwise b* = max_depth.
    
    # The current program's logic:
    # term_b = k * A * dt * dV_bid
    # valid_b = term_b > 0  (i.e. dV_bid > 0)
    # b_opt = (1/k) ln(term_b)
    # This is WRONG. If dV_bid > 0, we should set b = max_depth.
    # If dV_bid < 0, we check if (-dV_bid) A dt k >= 1. If so, b = (1/k) ln((-dV_bid) A dt k). Else b = max_depth.
    
    # Let's fix this logic.
    
    for n in range(steps - 1, -1, -1):
        V_lo = V_next[idx_lo]
        V_mid = V_next[idx]
        V_hi = V_next[idx_hi]
        
        dV_bid = V_lo - V_mid
        dV_ask = V_hi - V_mid
        
        # Optimal b
        # f'(b) = 1 + k A dt dV_bid e^{-kb}
        # If dV_bid >= 0, f' > 0, max b.
        # If dV_bid < 0, f' = 1 - k A dt (-dV_bid) e^{-kb}.
        # Root exists if k A dt (-dV_bid) >= 1.
        # b* = (1/k) ln(k A dt (-dV_bid)).
        
        b_opt = np.full(nq, max_depth)
        # Condition: dV_bid < 0 AND k A dt (-dV_bid) >= 1
        cond_b = (dV_bid < 0) & (k * A * dt * (-dV_bid) >= 1.0)
        b_opt[cond_b] = (1.0 / k) * np.log(k * A * dt * (-dV_bid[cond_b]))
        b_opt = np.clip(b_opt, 0.0, max_depth)
        
        # Optimal a
        # Symmetric. f'(a) = 1 + k A dt dV_ask e^{-ka}
        # If dV_ask >= 0, max a.
        # If dV_ask < 0, root if k A dt (-dV_ask) >= 1.
        # a* = (1/k) ln(k A dt (-dV_ask)).
        
        a_opt = np.full(nq, max_depth)
        cond_a = (dV_ask < 0) & (k * A * dt * (-dV_ask) >= 1.0)
        a_opt[cond_a] = (1.0 / k) * np.log(k * A * dt * (-dV_ask[cond_a]))
        a_opt = np.clip(a_opt, 0.0, max_depth)
        
        opt_bids[n] = b_opt
        opt_asks[n] = a_opt
        
        p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
        
        p_b[nq - 1] = 0.0
        p_a[0] = 0.0
        
        reward = (b_opt + a_opt) - risk_coeff * qs**2
        cont = p_b * V_lo + p_a * V_hi + (1.0 - p_b - p_a) * V_mid
        V_next = reward + cont
        
        if n % 100 == 0 and _remaining_budget() < 5.0:
            return None
            
    return opt_bids, opt_asks

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        result = _solve_policy(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[key] = result
        
    data = _POLICY_CACHE[key]
    
    if data is None:
        # Fallback to analytic shift
        inv_k = 1.0 / k
        max_depth = 3.0 * inv_k
        shift = gamma * sigma * sigma * 0.5 / (A * k)
        q = inventory.astype(float)
        bid = np.clip(inv_k + q * shift, 0.0, max_depth)
        ask = np.clip(inv_k - q * shift, 0.0, max_depth)
        bid = np.nan_to_num(bid, nan=inv_k, posinf=max_depth, neginf=0.0)
        ask = np.nan_to_num(ask, nan=inv_k, posinf=max_depth, neginf=0.0)
        return bid, ask
        
    opt_bids, opt_asks = data
    
    dt = T / steps
    n = int(t / dt)
    n = max(0, min(n, steps - 1))
    
    q = inventory
    idx = q + Q
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = opt_bids[n][idx]
    ask_depths = opt_asks[n][idx]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
