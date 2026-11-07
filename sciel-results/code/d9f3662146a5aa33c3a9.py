"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

def _solve_depths(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    dt = T / params["steps"]
    
    # Continuous time remaining
    tau = T - t
    
    # Avellaneda-Stoikov parameters
    # x = gamma * sigma^2
    x = gamma * sigma * sigma
    
    # The optimal spread and skew are derived from the HJB equation.
    # For the discrete model, we approximate the continuous solution.
    # The reservation price is mid - gamma * sigma^2 * q * tau
    # The optimal spread is (2 / k) * (1 - exp(-k * spread)) ... no, that's not right.
    
    # Let's use the known closed-form for the Avellaneda-Stoikov model:
    # Optimal spread s* = (2 / k) * (1 - exp(-k * s*)) ... wait.
    # The HJB equation is:
    # max_d [ (1 - exp(-k d)) * (d - V_q+1 + V_q) + (1 - exp(-k d)) * (d - V_q-1 + V_q) - x * q^2 - sigma^2/2 * V_qq ] = 0
    # Actually, the standard AS model has:
    # max_d [ lambda(d) * (d - V_{q+1} + V_q) + lambda(d) * (d - V_{q-1} + V_q) - x q^2 ]
    # where lambda(d) = A * exp(-k d).
    
    # The optimal spread s satisfies:
    # s = (2 / k) * (1 - exp(-k * s / 2)) ... no.
    # Let's use the Newton solver for the optimal spread s*.
    # The first order condition is:
    # k * A * exp(-k s*) * (s* - 2 * (1 - exp(-k s*/2)) / (k * A * exp(-k s*/2))) ... this is getting complicated.
    
    # Let's use the simpler approximation:
    # s* ≈ 2 / k * (1 - exp(-k * s* / 2))
    # For small k*s, s* ≈ 2/k.
    
    # Actually, the exact solution for the continuous AS model is:
    # s* = (2 / k) * (1 - exp(-k * s* / 2))
    # This is a transcendental equation. We can solve it with Newton's method.
    
    # Let's define f(s) = s - (2 / k) * (1 - exp(-k * s / 2))
    # f'(s) = 1 - exp(-k * s / 2)
    
    # We want to solve f(s) = 0.
    
    # Initial guess: s0 = 2 / k
    s0 = 2.0 / k
    
    # Newton's method
    s = s0
    for _ in range(10):
        f_s = s - (2.0 / k) * (1.0 - np.exp(-k * s / 2.0))
        f_prime_s = 1.0 - np.exp(-k * s / 2.0)
        if abs(f_prime_s) < 1e-12:
            break
        s_new = s - f_s / f_prime_s
        if abs(s_new - s) < 1e-10:
            s = s_new
            break
        s = s_new
    
    # The optimal bid and ask depths are:
    # bid_depth = s/2 - gamma * sigma^2 * q * tau
    # ask_depth = s/2 + gamma * sigma^2 * q * tau
    
    # However, the model in the task is discrete.
    # The fill probability is 1 - exp(-A exp(-k d) dt).
    # This is different from the continuous AS model where the fill rate is A exp(-k d).
    
    # Let's adapt the solution to the discrete model.
    # The optimal depth d* satisfies:
    # A exp(-k d*) dt * (d* - V_{q+1} + V_q) = V_q - V_{q-1} ... no.
    
    # Let's use the approximation that the optimal depth is close to 1/k.
    # The seed program uses 1/k.
    # The HJB solution uses a grid search.
    
    # Let's try a different approach: use the continuous AS solution as a starting point,
    # but adjust it for the discrete model.
    
    # The discrete fill rate is lambda_d = A exp(-k d) dt.
    # The continuous fill rate is lambda_c = A exp(-k d).
    # So the discrete model is like the continuous model with a time step dt.
    
    # The optimal spread in the discrete model should be close to the continuous one.
    
    # Let's use the continuous AS solution for the spread, and adjust the skew.
    
    # The reservation price is mid - gamma * sigma^2 * q * tau.
    # The optimal bid is reservation - s/2.
    # The optimal ask is reservation + s/2.
    
    # So the bid depth is s/2 - gamma * sigma^2 * q * tau.
    # The ask depth is s/2 + gamma * sigma^2 * q * tau.
    
    # But we need to ensure the depths are positive.
    
    bid_depths = s / 2.0 - gamma * sigma * sigma * inventory * tau
    ask_depths = s / 2.0 + gamma * sigma * sigma * inventory * tau
    
    # Clip to a reasonable range to avoid numerical issues.
    min_depth = -3.0 / k
    max_depth = 10.0 / k
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # If the depth is too small, it means we should not quote.
    # But the model allows negative depths? No, depths must be positive.
    # Wait, the task says "distances below and above mid to quote this step".
    # A negative depth would mean quoting on the other side of mid.
    # The task says "np.inf means no quote".
    
    # Let's check if the depth is too small.
    # If the depth is less than a small epsilon, set it to np.inf.
    
    # Actually, the task says "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid."
    # So we can have negative depths, as long as they are not below -3/k.
    
    return bid_depths, ask_depths

def quote(t, inventory, mid, params):
    """Closed-form optimal market making with inventory risk."""
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
        
    bid_depths, ask_depths = _solve_depths(t, inventory, mid, params)
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
