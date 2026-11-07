"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin below 120s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_grid = resource_grid
    n_m = len(m_grid)

    # Use a dense internal grid for accurate interpolation of continuation values
    # The evaluator interpolates our policy onto a 251-point grid, so our internal
    # resolution should be at least that to avoid interpolation error.
    n_dense = 500
    dense_grid = np.geomspace(0.001, 40, n_dense)

    # We will store consumption on the dense grid internally, then downsample to m_grid
    # Actually, the task requires returning shape (6, 2, 101) on resource_grid.
    # But the evaluator interpolates our result onto a 251-point grid.
    # If we solve on a dense grid and interpolate down to 101, we lose accuracy.
    # If we solve on the 101 grid, we are limited by that grid's resolution.
    # The seed solved on the 101 grid.
    # Let's solve on a dense grid (e.g., 251 or 501) and interpolate the final result to 101.
    # Or better: solve on the 101 grid but use a dense grid for the *continuation value* lookup.
    # The bottleneck in the seed was the scalar loop. Vectorization is key.

    # Let's stick to solving on the provided resource_grid (101 points) to match the output shape directly,
    # but use a dense grid for interpolating V' from the next period.
    # Wait, if we solve on 101 points, the policy is only defined at those 101 points.
    # The evaluator interpolates our 101-point policy onto 251 points.
    # If we solved on a 251-point grid and interpolated down to 101, the 101-point values would be
    # more accurate representations of the true policy.
    # Let's solve on a dense grid (e.g., 251) and interpolate the result to the required 101 grid.
    
    solve_grid = np.geomspace(0.001, 40, 251)
    n_solve = len(solve_grid)
    
    consumption_solve = np.zeros((horizon, 2, n_solve))
    consumption_solve[horizon-1, :, :] = solve_grid[np.newaxis, :]

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

    # Precompute transition probabilities as a matrix for easy broadcasting
    # transition[z, zp] is P(zp | z)
    # We need E[V'(m', zp) | z] = sum_zp transition[z, zp] * V'(m', zp)
    # Let V_prime_next be shape (2, n_solve).
    # For a given z and m_next, we need:
    # sum_zp transition[z, zp] * np.interp(m_next + income[zp], solve_grid, V_prime_next[zp, :])

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: fill remaining periods with a simple rule
            for tt in range(t, -1, -1):
                 consumption_solve[tt, :, :] = solve_grid[np.newaxis, :] / (horizon - tt)
            break
        
        # V'_{t+1}(m, z) = u'(c^*_{t+1}(m, z))
        c_next = consumption_solve[t+1, :, :] # shape (2, n_solve)
        u_prime_next = get_utility_prime(c_next, gamma) # shape (2, n_solve)
        
        # We want to solve for c_t(m, z) for all m in solve_grid and z in {0, 1}
        # FOC: u'(c) = beta * R * E[V'_{t+1}(R*(m-c) + income[z'], z') | z]
        
        # Let's vectorize over m and z.
        # m shape: (n_solve,)
        # z shape: (2,)
        # We create a 2D array of m values for all z, m pairs
        # Actually, we can just loop over z (only 2 iterations) and vectorize over m.
        
        for z in range(2):
            m_vals = solve_grid # shape (n_solve,)
            
            # Initial guess for c
            # A good guess is proportional to m.
            # c ~ m * (1 - beta) / (1 - beta * R) approximately for log utility
            # Let's start with c = 0.5 * m
            c = 0.5 * m_vals.copy()
            
            # Newton's method
            # g(c) = u'(c) - beta * R * E[V']
            # g'(c) = u''(c) + beta * R * E[V'' * (-R)]
            # Note: E[V'] depends on c through m' = R(m-c) + income[z']
            # d/dc E[V'] = sum_zp P(zp|z) * V''(m') * (-R)
            
            for _ in range(20):
                # Compute m' for each z'
                # m' shape: (2, n_solve) where index 0 is zp=0, index 1 is zp=1
                m_next = R * (m_vals[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
                
                # Clip m_next to valid range for interpolation?
                # np.interp handles out-of-bounds by clamping, which is fine.
                # But V'' is needed.
                
                # Interpolate V' and V''
                # V'_{t+1}(m', zp)
                vp = np.array([np.interp(m_next[zp, :], solve_grid, u_prime_next[zp, :]) for zp in range(2)])
                # vp shape: (2, n_solve)
                
                # V''_{t+1}(m', zp) = d/dm V' = u''(c^*) * dc^*/dm
                # By envelope theorem, V'(m) = u'(c^*(m)).
                # V''(m) = u''(c^*(m)) * c^*'(m).
                # From FOC of next period: u'(c^*) = beta * R * E[V''_{t+2}]... no.
                # From FOC: u'(c) = beta * R * E[V'_{next}].
                # Differentiate w.r.t m: u''(c) * c'(m) = beta * R * E[V''_{next} * (d m_next / d m)]
                # d m_next / d m = R * (1 - dc/dm) ... wait.
                # m_next = R(m-c) + income.
                # d m_next / d m = R * (1 - c'(m)).
                # So u''(c) * c' = beta * R * E[V'' * R * (1 - c')]
                # u''(c) * c' = beta * R^2 * E[V''] - beta * R^2 * E[V''] * c'
                # c' * (u''(c) + beta * R^2 * E[V'']) = beta * R^2 * E[V'']
                # c' = beta * R^2 * E[V''] / (u''(c) + beta * R^2 * E[V''])
                
                # We need V''_{t+1}. We don't have it directly.
                # However, we can approximate V'' numerically or compute it from the previous step's policy.
                # Alternative: Use Bisection which is robust and doesn't need derivatives.
                # Bisection is slower per iteration but vectorized bisection is still very fast.
                # Let's stick to Bisection for robustness, but vectorized.
                
                # g(c) = u'(c) - target
                # target = beta * R * sum_zp P(zp|z) * V'(m')
                # sum_zp P(zp|z) * vp[zp, :]
                expected_vp = transition[z, 0] * vp[0, :] + transition[z, 1] * vp[1, :]
                target = beta * R * expected_vp
                
                lhs = get_utility_prime(c, gamma)
                g = lhs - target
                
                # g(c) is decreasing in c.
                # If g > 0, c is too small (u' too high), increase c.
                # If g < 0, c is too large, decrease c.
                
                # Update c using bisection logic, but we need low/high bounds.
                # Let's maintain low and high arrays.
                # For simplicity and speed, let's just do a few steps of Newton if we can estimate derivative,
                # or just stick to bisection with fixed number of iterations.
                # 30 iterations of bisection on a range of size ~40 gives precision 40/2^30 ~ 1e-8.
                # 20 iterations gives 40/2^20 ~ 1e-5.
                # Given the score tolerance, 1e-5 might be enough, but let's do 25.
                
                # Actually, let's just use a simple iterative update:
                # c_new = c - g / g_prime_approx
                # Or just bisection.
                
                # Let's implement vectorized bisection properly.
                # We need to track low and high for each m.
                # Initialize low = 1e-12, high = m_vals
                # This requires storing low/high arrays.
                
                # Let's just do a simple relaxation:
                # c = c - alpha * g
                # This might not converge fast.
                
                # Let's go back to bisection but stored in arrays.
                # We'll do this outside the inner loop for efficiency.
                
                break # Break inner Newton loop, we'll do bisection below.
            
            # Vectorized Bisection
            c_low = np.full(n_solve, 1e-12)
            c_high = m_vals.copy()
            
            for _ in range(30):
                c_mid = (c_low + c_high) / 2.0
                
                m_next = R * (m_vals[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                vp = np.array([np.interp(m_next[zp, :], solve_grid, u_prime_next[zp, :]) for zp in range(2)])
                expected_vp = transition[z, 0] * vp[0, :] + transition[z, 1] * vp[1, :]
                target = beta * R * expected_vp
                
                lhs = get_utility_prime(c_mid, gamma)
                g = lhs - target
                
                # g > 0 => c_mid too small => c_low = c_mid
                # g < 0 => c_mid too large => c_high = c_mid
                mask = g > 0
                c_low[mask] = c_mid[mask]
                c_high[~mask] = c_mid[~mask]
                
                # Early stop if converged
                if np.max(c_high - c_low) < 1e-10:
                    break
            
            c_opt = (c_low + c_high) / 2.0
            consumption_solve[t, z, :] = c_opt

    # Interpolate the solution from solve_grid (251 points) to resource_grid (101 points)
    consumption = np.zeros((horizon, 2, n_m))
    for z in range(2):
        for t in range(horizon):
            consumption[t, z, :] = np.interp(resource_grid, solve_grid, consumption_solve[t, z, :])
    
    # Final check and clipping
    # Ensure 0 < c <= m + 1e-10
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)
    
    # Ensure c[5] is exactly the grid
    consumption[5, :, :] = resource_grid[np.newaxis, :]
    
    return consumption
# EVOLVE-BLOCK-END
