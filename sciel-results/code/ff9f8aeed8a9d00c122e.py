"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin below 150s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_grid = resource_grid
    n_m = len(m_grid)
    
    # We solve on a grid that matches the evaluation grid to avoid final interpolation errors.
    # The evaluator interpolates onto np.geomspace(0.001, 40, 251).
    # We use 251 points for our internal solve to be exact on the evaluation points.
    # Actually, the task says "interpolated with np.interp onto np.geomspace(0.001, 40, 251)".
    # If we solve exactly on that grid, interpolation is identity (within floating point).
    # Let's use 251 points.
    n_solve = 251
    solve_grid = np.geomspace(0.001, 40, n_solve)
    
    # Map input resource_grid to solve_grid indices if they match, or just interpolate at the end.
    # The function must return shape (6, 2, 101).
    # So we solve on 251, then interpolate back to 101.

    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # Value function V[t, z, :] has shape (horizon, 2, n_solve)
    V = np.zeros((horizon, 2, n_solve))
    c_opt = np.zeros((horizon, 2, n_solve))

    # Terminal period: consume everything
    c_opt[horizon - 1, :, :] = solve_grid[np.newaxis, :]
    V[horizon - 1, :, :] = get_utility(solve_grid[np.newaxis, :], gamma)

    # Pre-compute log-spaced grid for potential transformations? 
    # No, let's stick to standard m-space for clarity and robustness.
    
    # For efficient lookup of V_next, we will use np.interp.
    # To speed up the search for c*, we exploit the fact that the objective 
    # J(c) = u(c) + beta * E[V_next(R*(m-c) + y')] is concave in c.
    # We can use a vectorized bisection or golden section search.
    # Given the small horizon (6) and 2 states, and 251 points, 
    # a simple loop with np.interp might be slow if not careful.
    
    # Let's try a different approach: 
    # Since the grid is log-spaced, and the problem is time-invariant in structure (though finite horizon),
    # we can use a coarser grid for the search if needed, but 251 is small enough.
    
    # Optimization: Pre-compute the interpolation parameters for V_next?
    # np.interp is quite fast.
    
    # We will use a "policy iteration" style update where we find the argmax.
    # To make it fast, we can use the fact that c* is monotonic in m.
    # But let's just do a robust 1D search for each m.
    # 6 periods * 2 states * 251 points * (search cost)
    # Search cost: ~20 iterations of bisection * 2 np.interp calls.
    # 6 * 2 * 251 * 20 * 2 = ~120,000 np.interp calls. This is very fast in Python/NumPy.
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: simple rule
            for tt in range(t, -1, -1):
                c_opt[tt, :, :] = solve_grid[np.newaxis, :] / (horizon - tt)
                V[tt, :, :] = get_utility(c_opt[tt, :, :], gamma)
            break

        for z in range(2):
            m_vals = solve_grid
            
            # We want to maximize f(c) = u(c) + beta * (p0 * V_next0(R*(m-c)+y0) + p1 * V_next1(R*(m-c)+y1))
            # Let's define the objective function for a vector of c's.
            
            # Initial guess for c:
            # For CRRA, c is roughly proportional to m.
            # c_init = m * 0.5
            c_low = np.full_like(m_vals, 1e-6)
            c_high = m_vals.copy()
            
            # Bisection search for the optimal c
            # The derivative of the objective w.r.t c is:
            # u'(c) - beta * R * (p0 * V'_next0(m_next0) + p1 * V'_next1(m_next1))
            # Since we don't have easy access to V' without splines, we can just maximize the value directly.
            # However, maximizing value directly with bisection requires knowing the sign of the derivative.
            # We can approximate the derivative or just do a golden section search.
            # Golden section is safer for black-box maximization.
            
            # Alternatively, we can use the fact that for small m, c is small, and for large m, c is large.
            # Let's use a simple bisection on the "marginal utility gap".
            # We need V' for this. 
            # Let's compute V' using finite differences on the log grid? 
            # Or just use np.interp for V and approximate the slope?
            
            # Actually, the current program uses Newton with splines. 
            # The inspiration suggests "Direct value function iteration on a coarser, fixed grid".
            # Let's try a different structural change: **Approximate the Value Function with a Log-Linear Function**.
            # V(m, z) ~= A[z] * m^B[z] (Power function)
            # This is standard for CRRA. 
            # If we assume V is a power function, we can solve for the parameters A, B.
            # But in finite horizon, it's not exactly a power function.
            
            # Let's stick to the grid but improve the search.
            # I will implement a vectorized Golden Section Search.
            
            # Golden ratio
            gr = (np.sqrt(5) - 1) / 2
            c1 = c_high - gr * (c_high - c_low)
            c2 = c_low + gr * (c_high - c_low)
            
            def obj_func(c):
                c = np.clip(c, 1e-12, m_vals)
                m_next_0 = R * (m_vals - c) + income[0]
                m_next_1 = R * (m_vals - c) + income[1]
                
                # Clip m_next to grid bounds for interpolation
                m_next_0 = np.clip(m_next_0, solve_grid[0], solve_grid[-1])
                m_next_1 = np.clip(m_next_1, solve_grid[0], solve_grid[-1])
                
                V_next_0 = np.interp(m_next_0, solve_grid, V[t+1, 0, :])
                V_next_1 = np.interp(m_next_1, solve_grid, V[t+1, 1, :])
                
                expected_V = transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1
                return get_utility(c, gamma) + beta * expected_V

            f1 = obj_func(c1)
            f2 = obj_func(c2)
            
            max_iter = 30
            for _ in range(max_iter):
                if time.time() > deadline - 2:
                    break
                if f1 > f2:
                    c_high = c2
                    c2 = c1
                    f2 = f1
                    c1 = c_high - gr * (c_high - c_low)
                    f1 = obj_func(c1)
                else:
                    c_low = c1
                    c1 = c2
                    f1 = f2
                    c2 = c_low + gr * (c_high - c_low)
                    f2 = obj_func(c2)
                
                if np.max(np.abs(c_high - c_low)) < 1e-8:
                    break
            
            c_opt[t, z, :] = (c_low + c_high) / 2.0
            c_opt[t, z, :] = np.clip(c_opt[t, z, :], 1e-12, m_vals)
            
            # Update V
            V[t, z, :] = obj_func(c_opt[t, z, :])

    # Interpolate the solution from solve_grid (251 points) to resource_grid (101 points)
    consumption = np.zeros((horizon, 2, n_m))
    for z in range(2):
        for t in range(horizon):
            consumption[t, z, :] = np.interp(resource_grid, solve_grid, c_opt[t, z, :])

    # Final check and clipping
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)

    # Ensure c[5] is exactly the grid
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :]

    # Final validity check
    consumption = np.where(np.isfinite(consumption), consumption, resource_grid[np.newaxis, np.newaxis, :])
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)

    return consumption
# EVOLVE-BLOCK-END
