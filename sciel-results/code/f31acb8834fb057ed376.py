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
    steps = int(params["steps"])
    
    dt = T / steps

    n = len(inventory)
    q = np.asarray(inventory, dtype=np.float64)

    # Time remaining
    tau = np.clip(T - float(t), 0.0, T)

    # Analytic solution for the quadratic value function coefficient C(tau)
    # C'(tau) = alpha * C(tau)^2 - beta
    # alpha = 4 * A * dt * k / e
    # beta = gamma * sigma^2 / 2
    # C(tau) = - sqrt(beta/alpha) * tanh(sqrt(alpha*beta) * tau)
    
    alpha = 4.0 * A * dt * k / np.e
    beta = 0.5 * gamma * sigma * sigma
    
    if alpha == 0:
        # If A or dt is 0, no fills, no skew
        S = np.zeros_like(q)
    else:
        sqrt_ab = np.sqrt(alpha * beta)
        sqrt_ba = np.sqrt(beta / alpha)
        # Handle small tau to avoid numerical issues with tanh
        # tanh(x) ~ x for small x
        arg = sqrt_ab * tau
        tanh_arg = np.tanh(arg)
        
        C = -sqrt_ba * tanh_arg
        # S = V_q = 2 * C * q
        S = 2.0 * C * q

    # Optimal depths
    # d_bid = 1/k - S
    # d_ask = 1/k + S
    base_depth = 1.0 / k
    
    bid = base_depth - S
    ask = base_depth + S

    # Constraints
    lo = -3.0 / k
    hi = float(Q)
    
    # Clip to valid range
    bid = np.clip(bid, lo, hi)
    ask = np.clip(ask, lo, hi)

    # Boundary conditions
    # No bid fill at q = Q
    bid = np.where(q >= Q, np.inf, bid)
    # No ask fill at q = -Q
    ask = np.where(q <= -Q, np.inf, ask)

    # Final validity check
    bid = np.where(np.isfinite(bid) & (bid > lo), bid, np.inf)
    ask = np.where(np.isfinite(ask) & (ask > lo), ask, np.inf)

    return bid, ask
# EVOLVE-BLOCK-END
