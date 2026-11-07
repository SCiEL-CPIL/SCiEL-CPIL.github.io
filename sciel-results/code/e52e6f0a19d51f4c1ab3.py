"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin below 150s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    
    # Initialize consumption for the last period (t = horizon - 1)
    # At the last period, optimal consumption is all resources (m)
    # V_5(m, z) = u(m)
    # Policy c_5(m, z) = m
    consumption = np.zeros((horizon, 2, n_m))
    consumption[horizon-1, :, :] = m_grid[np.newaxis, :]
    
    # Value function for t = horizon - 1
    # V(m, z) = u(m)
    # We need V_{t+1} to solve for c_t
    
    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    def get_utility_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def get_utility_double_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: just return current best (which is filled for t >= current)
            # But we need to fill remaining t with something valid, e.g., linear rule
            # For safety, if we run out of time, we can use a simple heuristic for remaining t
            # However, we are going backwards, so t is decreasing.
            # If we break, we need to fill t-1, ..., 0.
            # Let's just break and fill with a simple rule: c = m / (horizon - t)
            for tt in range(t, -1, -1):
                 consumption[tt, :, :] = m_grid[np.newaxis, :] / (horizon - tt)
            break
        
        # V_{t+1} is known from the previous iteration (or initialization)
        # We need to compute V_{t+1}(m', z') for all z, z' and m'
        # m' = R * (m - c) + income[z']
        
        # We want to find c that maximizes:
        # u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R*(m-c) + income[z'], z')
        
        # Let F(c, m, z) = u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R*(m-c) + income[z'], z')
        # First Order Condition (FOC):
        # u'(c) + beta * sum_{z'} P(z'|z) * V'_{t+1}(m', z') * (-R) = 0
        # u'(c) = beta * R * sum_{z'} P(z'|z) * V'_{t+1}(m', z')
        
        # We need V'_{t+1}(m', z').
        # Since V_{t+1} was computed as the max over c' of u(c') + beta * E[V_{t+2}], 
        # by envelope theorem, V'_{t+1}(m', z') = u'(c^*_{t+1}(m', z'))
        # So we need the optimal consumption policy c^*_{t+1} for all m' and z'.
        
        # We have consumption[t+1, z, m_idx] which is c^*_{t+1}(m, z) on the grid m_grid.
        # We need to interpolate V'_{t+1}(m', z) for arbitrary m' = R*(m-c) + income[z'].
        
        # Step 1: Compute V'_{t+1} on the grid for each z.
        # V'_{t+1}(m, z) = u'(c^*_{t+1}(m, z))
        c_next = consumption[t+1, :, :] # shape (2, n_m)
        u_prime_next = get_utility_prime(c_next, gamma) # shape (2, n_m)
        
        # Now for each t, z, m, we need to solve for c.
        # Equation: u'(c) = beta * R * sum_{z'} P(z'|z) * interp(u_prime_next[z'], m_grid, R*(m-c) + income[z'])
        
        # This is a nonlinear equation for c.
        # Let's solve it using Newton's method or bisection for each (z, m).
        # Since n_m=101, 2 states, 6 periods, this is 6*2*101 = 1212 solves.
        # Each solve involves interpolation and a few iterations. Should be fast.
        
        for z in range(2):
            for i_m in range(n_m):
                m = m_grid[i_m]
                
                # Define the function to solve: g(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * V'_{t+1}(R*(m-c) + income[z'], z')
                def g(c):
                    if c <= 0:
                        return np.inf # u'(c) -> inf as c->0
                    if c > m:
                        return -np.inf # u'(c) is small, RHS is large (if m-c is large)
                        # Actually if c > m, m-c < 0, which is invalid.
                        # We constrain c in (0, m).
                    
                    m_next = R * (m - c)
                    expected_v_prime = 0.0
                    for zp in range(2):
                        m_zp = m_next + income[zp]
                        # Interpolate u_prime_next[zp] at m_zp
                        # m_zp can be outside [min(m_grid), max(m_grid)]
                        # np.interp clamps to boundary values. This is an approximation.
                        # Better to extrapolate? Or just clamp.
                        # Given the grid is geomspace 0.001 to 40, and R=1.02, income 0.5-1.5
                        # m' can be slightly outside. Clamping is standard for np.interp.
                        vp = np.interp(m_zp, m_grid, u_prime_next[zp])
                        expected_v_prime += transition[z, zp] * vp
                    
                    target = beta * R * expected_v_prime
                    return get_utility_prime(c, gamma) - target
                
                # Bisection search for c in (1e-12, m)
                # g(c) is monotonic decreasing?
                # u'(c) is decreasing in c.
                # RHS: m-c decreases as c increases -> m' decreases -> V'_{t+1}(m') increases (since V' is decreasing in m? No, V' is decreasing in m? 
                # V(m) is concave, so V'(m) is decreasing in m.
                # So as c increases, m' decreases, V'(m') increases.
                # So RHS increases as c increases.
                # LHS (u'(c)) decreases as c increases.
                # So g(c) = LHS - RHS is strictly decreasing.
                # We can use bisection.
                
                c_low = 1e-12
                c_high = m
                
                # Check boundaries
                # If g(c_low) < 0, then even at min c, u'(c) < RHS. This shouldn't happen if m>0.
                # If g(c_high) > 0, then even at max c, u'(c) > RHS.
                
                for _ in range(30):
                    c_mid = (c_low + c_high) / 2.0
                    val = g(c_mid)
                    if val > 0:
                        c_low = c_mid
                    else:
                        c_high = c_mid
                    if c_high - c_low < 1e-10:
                        break
                
                c_opt = (c_low + c_high) / 2.0
                consumption[t, z, i_m] = c_opt
        
    # Final check and clipping
    # Ensure 0 < c <= m + 1e-10
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)
    
    # Ensure c[5] is exactly the grid
    consumption[5, :, :] = resource_grid[np.newaxis, :]
    
    return consumption
# EVOLVE-BLOCK-END
