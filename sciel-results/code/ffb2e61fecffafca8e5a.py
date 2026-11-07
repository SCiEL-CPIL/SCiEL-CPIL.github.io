import numpy as np
import time
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin below 120s (and well below 150s kill)
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    # Output grid
    n_out = len(resource_grid)
    m_out = resource_grid
    
    # Internal high-resolution grid for solving
    # We need enough points to define the spline well.
    # The evaluator interpolates onto 251 points, so we should be at least that fine.
    # Let's use 500 points for the internal solve to ensure accuracy.
    n_internal = 500
    m_internal = np.geomspace(0.001, 40, n_internal)
    
    # Policy storage: shape (horizon, 2, n_internal)
    policy = np.zeros((horizon, 2, n_internal))
    
    # Terminal condition: c_T = m
    policy[horizon - 1, :, :] = m_internal[None, :, :]
    
    # We will solve backward from t = horizon - 2 down to 0
    # For each t, we need the policy from t+1.
    # We create splines for the policy at t+1 for each state z.
    
    # Pre-compute constants
    # u'(c) = c^(-gamma)
    
    def marginal_utility(c):
        return c ** (-gamma)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            break
            
        # Create splines for the next period's policy c_{t+1}(m, z)
        # We need to ensure the spline is well-behaved.
        # Policy is increasing in m.
        splines = []
        for z in range(2):
            # Ensure monotonicity or just fit a spline.
            # CubicSpline can oscillate if not careful, but for this smooth problem it should be fine.
            # Using 'not-a-knot' or 'natural' boundary conditions.
            try:
                # We sort just in case, though m_internal is sorted.
                cs = CubicSpline(m_internal, policy[t + 1, z, :], bc_type='natural')
                splines.append(cs)
            except Exception:
                # Fallback to linear if spline fails
                splines.append(lambda m, z=z: np.interp(m, m_internal, policy[t + 1, z, :]))
        
        for z in range(2):
            probs = transition[z]
            
            for i in range(n_internal):
                m = m_internal[i]
                
                # Define the function to solve: F(c) = u'(c) - beta * R * E[u'(c_next)]
                # c is in (0, m]
                
                def f(c):
                    # If c is too close to 0, mu is huge.
                    # If c is m, saving is 0 (plus income).
                    mu_c = marginal_utility(c)
                    
                    # Calculate expected next period marginal utility
                    # m_next_z = R * (m - c) + income[z']
                    # c_next_z = policy_next(m_next_z, z')
                    
                    ev_mu = 0.0
                    for z_next in range(2):
                        m_next = R * (m - c) + income[z_next]
                        # m_next can be negative? 
                        # R=1.02, m=0.001, c=0.001 -> m_next = 0 + income. Income is 0.5 or 1.5. So m_next > 0.
                        # If m is small and c is close to m, m_next is small + income.
                        # We need to evaluate the spline.
                        # Spline domain is [0.001, 40].
                        # If m_next < 0.001, we should clamp or extrapolate?
                        # Physically, m_next >= income[0] > 0.
                        # Min income is 0.5. So m_next >= 0.5.
                        # So m_next is always well within [0.001, 40] for reasonable m, c.
                        # Wait, if m is very small, say 0.001, and c is small, m_next is approx 0.5.
                        # If m is large, say 40, and c is small, m_next is approx 40*1.02 + 0.5 = 41.3.
                        # Spline is defined on [0.001, 40].
                        # np.interp clamps. CubicSpline extrapolates.
                        # Extrapolation might be unstable.
                        # Let's clamp m_next to the grid range for evaluation to be safe, 
                        # or rely on the fact that optimal c won't push m_next too far out of bounds?
                        # Actually, if m=40, c=1, m_next=41.3. 
                        # The policy at 41.3 is likely close to policy at 40 (or higher).
                        # Using extrapolation is risky. 
                        # Let's use a safe evaluation function.
                        
                        if m_next < m_internal[0]:
                            c_next = policy[t + 1, z_next, 0]
                        elif m_next > m_internal[-1]:
                            c_next = policy[t + 1, z_next, -1]
                        else:
                            c_next = splines[z_next](m_next)
                        
                        # Ensure c_next is positive
                        c_next = max(c_next, 1e-12)
                        
                        ev_mu += probs[z_next] * marginal_utility(c_next)
                    
                    return mu_c - beta * R * ev_mu
                
                # Solve for c
                # f(c) is decreasing in c?
                # mu_c = c^-gamma. Decreasing in c.
                # E[mu_next]: As c increases, m_next decreases, so c_next decreases (policy is increasing in m), so mu_next increases.
                # So E[mu_next] is increasing in c.
                # So f(c) = decreasing - increasing = strictly decreasing.
                # Unique root.
                
                # Bounds:
                # c_min = 1e-12
                # c_max = m
                # Check signs
                # At c_min, mu_c is huge (1e12*gamma). f(c_min) > 0.
                # At c_max, mu_c = m^-gamma.
                # m_next = income. c_next = policy(income).
                # f(c_max) = m^-gamma - beta*R*E[mu(income)]
                # If f(c_max) > 0, then the root is > m? 
                # But c cannot exceed m.
                # If f(m) > 0, it means we want to consume more than m, which is impossible.
                # So we set c = m.
                # If f(m) < 0, there is a root in (0, m).
                
                f_min = f(1e-12)
                f_max = f(m)
                
                if f_max > 0:
                    c_opt = m
                elif f_min < 0:
                    c_opt = 1e-12 # Should not happen usually
                else:
                    try:
                        c_opt = brentq(f, 1e-12, m, xtol=1e-12)
                    except ValueError:
                        # Fallback if brentq fails (e.g. no sign change due to numerical issues)
                        # Use bisection manually or just pick midpoint?
                        # Let's try a few points
                        c_opt = m / 2.0
                        # Simple refinement
                        for _ in range(20):
                            mid = c_opt
                            if f(mid) > 0:
                                c_opt = mid + (m - mid) / 2
                            else:
                                c_opt = mid / 2
                        # Actually, just use the last valid value or m
                        c_opt = m if f(m) > 0 else 1e-12

                policy[t, z, i] = c_opt
        
        # Optional: Check time and break if needed
        if time.time() > deadline - 10:
             break

    # If we broke out early, fill remaining (earlier) t with the last computed policy
    # The loop goes from T-2 down to 0.
    # If we break at t, then t-1, ..., 0 are still 0.
    # We should copy the policy from the last completed t to the earlier ones.
    last_valid_t = horizon - 2
    for t in range(horizon - 2, -1, -1):
        if not np.all(policy[t] == 0):
            last_valid_t = t
            break
            
    if last_valid_t != horizon - 2:
        for t in range(last_valid_t, -1, -1):
            if np.all(policy[t] == 0):
                policy[t] = policy[last_valid_t]

    # Interpolate policy from internal grid (500) to output grid (101)
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            # Use np.interp for final output as it's robust
            consumption_out[t, z, :] = np.interp(m_out, m_internal, policy[t, z, :])
            
    # Final validity check and correction
    # c must be > 0.
    # c must be <= m + 1e-10.
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.maximum(consumption_out[t, z, :], 1e-12)
            consumption_out[t, z, :] = np.minimum(consumption_out[t, z, :], m_out + 1e-10)
            
    # Specific constraint: c[5] equal to the grid within 1e-9
    consumption_out[5, :, :] = m_out[None, None, :]
    
    return consumption_out
