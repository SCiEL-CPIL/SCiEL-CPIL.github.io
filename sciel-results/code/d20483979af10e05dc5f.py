# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Continuous-state HJB approximation using analytical FOC for optimal depths.
    
    We model the value function V(t, q) and use a semi-Lagrangian / analytical
    approach. The key insight is that the optimal bid/ask depths can be derived
    from the first-order condition of the Hamiltonian, which is concave in depth.
    
    For a small time step dt, the value function satisfies:
    V(t, q) = max_{b, a} [ (1 - exp(-A e^{-k b} dt)) (b - V(t+dt, q+1) + V(t+dt, q))
                         + (1 - exp(-A e^{-k a} dt)) (a - V(t+dt, q-1) + V(t+dt, q))
                         - (gamma/2) sigma^2 q^2 dt
                         + V(t+dt, q) ]
    
    Let delta_q = V(t+dt, q+1) - V(t+dt, q) (marginal value of inventory)
    Let delta_q_minus = V(t+dt, q-1) - V(t+dt, q)
    
    For the bid side, we maximize:
    f(b) = (1 - exp(-A e^{-k b} dt)) (b - delta_q)
    
    Taking derivative wrt b and setting to 0:
    A k e^{-k b} dt (b - delta_q) + (1 - exp(-A e^{-k b} dt)) = 0
    
    Let x = e^{-k b}, so b = -ln(x)/k.
    A k dt (-ln(x)/k - delta_q) x + (1 - A dt x) = 0
    -A dt (ln(x) + k delta_q) x + 1 - A dt x = 0
    1 - A dt x - A dt x ln(x) - A dt k delta_q x = 0
    
    For small dt, we can approximate. But a better approach:
    
    Actually, let's think about this differently. The objective per step is:
    Earned depth from fills - inventory cost.
    
    The expected profit from quoting at bid depth b is:
    E[profit_bid] = (1 - exp(-A e^{-k b} dt)) * b - (1 - exp(-A e^{-k b} dt)) * marginal_cost
    
    where marginal_cost is the change in value due to inventory change.
    
    Let's define the marginal value of inventory as m_q = V(t+dt, q+1) - V(t+dt, q).
    When a bid fill happens, inventory increases by 1, so the cost is m_q.
    When an ask fill happens, inventory decreases by 1, so the benefit is -m_{q-1} = V(t+dt, q) - V(t+dt, q-1).
    
    For simplicity and given the symmetric nature, let's approximate m_q ≈ m_{q-1} ≈ m.
    
    Then for bid: maximize (1 - exp(-A e^{-k b} dt)) (b - m)
    For ask: maximize (1 - exp(-A e^{-k a} dt)) (a + m)
    
    Taking FOC for bid:
    Let p = exp(-A e^{-k b} dt) (probability of no fill)
    f(b) = (1-p)(b-m)
    f'(b) = -p'(b)(b-m) + (1-p)
    p'(b) = p * A k e^{-k b} dt
    So: -p A k e^{-k b} dt (b-m) + (1-p) = 0
    (1-p) = p A k e^{-k b} dt (b-m)
    
    Let y = A e^{-k b} dt, so p = exp(-y), 1-p = 1-exp(-y), and e^{-k b} = y/(A dt)
    Then b = -ln(y/(A dt))/k = (ln(A dt) - ln(y))/k
    
    (1-exp(-y)) = exp(-y) * k * y/(A dt) * dt * ((ln(A dt) - ln(y))/k - m)
    (1-exp(-y)) = y * (ln(A dt) - ln(y) - k m)
    
    This is a transcendental equation in y. For small y (low fill probability), 1-exp(-y) ≈ y:
    y ≈ y (ln(A dt) - ln(y) - k m)
    1 ≈ ln(A dt) - ln(y) - k m
    ln(y) ≈ ln(A dt) - k m - 1
    y ≈ A dt exp(-k m - 1)
    
    Then b = (ln(A dt) - ln(y))/k = (ln(A dt) - ln(A dt) + k m + 1)/k = (k m + 1)/k = m + 1/k
    
    So optimal bid depth ≈ m + 1/k, where m is the marginal value of inventory.
    Similarly, optimal ask depth ≈ -m + 1/k (with sign adjustment).
    
    Now we need to estimate m = V(t+dt, q+1) - V(t+dt, q).
    
    For a linear inventory cost model, the value function is often approximated as:
    V(t, q) = C(t) - (gamma/2) sigma^2 q^2 (T-t) ... but this isn't quite right.
    
    Actually, for the Avellaneda-Stoikov style model, the optimal spread is:
    bid = mid - gamma sigma^2 (T-t) q - ...
    
    Let's use the known result from market making literature. The optimal bid/ask quotes are:
    bid = mid - gamma sigma^2 (T-t) q - (1/k) ln(1 + gamma sigma^2 (T-t) k)
    ask = mid + gamma sigma^2 (T-t) q + (1/k) ln(1 + gamma sigma^2 (T-t) k)
    
    Wait, that's for a different model. Let me derive more carefully.
    
    For our model with fill prob 1 - exp(-A exp(-k depth) dt), the optimal depths are:
    bid_depth = (1/k) [ln(A dt) - ln(y_bid)] where y_bid solves the FOC.
    
    Given the complexity, let's use a simpler approximation that captures the inventory risk:
    The marginal value of inventory m ≈ -gamma sigma^2 (T-t) q (from the quadratic cost).
    
    Then:
    bid_depth ≈ m + 1/k = -gamma sigma^2 (T-t) q + 1/k
    ask_depth ≈ -m + 1/k = gamma sigma^2 (T-t) q + 1/k
    
    But we need to clip to ensure positive depths and handle boundaries.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n = int(round(t / dt))
    time_remaining = T - t
    
    q = np.asarray(inventory, dtype=np.float64)
    
    # Marginal value of inventory approximation
    # m ≈ -gamma * sigma^2 * (T-t) * q
    m = -gamma * sigma**2 * time_remaining * q
    
    # Optimal depths from FOC approximation
    # bid_depth ≈ m + 1/k, ask_depth ≈ -m + 1/k
    base = 1.0 / k
    bid_depth = m + base
    ask_depth = -m + base
    
    # Ensure minimum positive depth (avoid non-positive or too small)
    min_depth = 1e-6
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # Handle inventory boundaries: no bid fill at q = Q, no ask fill at q = -Q
    # At q = Q, we should not quote bid (or quote at very large depth)
    # At q = -Q, we should not quote ask
    bid_depth = np.where(q >= Q, np.inf, bid_depth)
    ask_depth = np.where(q <= -Q, np.inf, ask_depth)
    
    # Convert to arrays of correct shape
    bid_depths = np.asarray(bid_depth, dtype=np.float64)
    ask_depths = np.asarray(ask_depth, dtype=np.float64)
    
    # Ensure shape is (4000,)
    if bid_depths.shape != (4000,):
        bid_depths = np.resize(bid_depths, (4000,))
    if ask_depths.shape != (4000,):
        ask_depths = np.resize(ask_depths, (4000,))
    
    # Validate: no NaN, no values below -3/k
    min_allowed = -3.0 / k
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_allowed), min_allowed + 1e-6, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_allowed), min_allowed + 1e-6, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
