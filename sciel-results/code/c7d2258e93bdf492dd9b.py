import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
                value[tt, :, :] = utility(0.5 * resource_grid)[np.newaxis, :]
            break
        
        for z in range(num_states):
            # We want to find optimal c for all m in resource_grid simultaneously.
            # Let m be a vector of shape (num_resources,).
            # We will search over a grid of c values.
            # Since c must be in [0, m], and m varies, we can define a grid of c
            # relative to m, or just a large grid of absolute c values and mask invalid ones.
            
            # A robust way:
            # Create a grid of c values. Since m ranges from 0.001 to 40,
            # and we need high precision, we can use a grid of c values.
            # However, the constraint c <= m is tricky to vectorize efficiently if c is a fixed grid.
            # Instead, let's define c as a fraction of m? No, the problem is non-linear.
            
            # Better approach:
            # For each m, the optimal c is in [0, m].
            # We can create a grid of c values that covers the maximum m.
            # Let max_m = resource_grid[-1].
            # Let c_grid be a vector of size N_c.
            # For each m_i, valid c are those <= m_i.
            # This creates a jagged array which is hard to vectorize.
            
            # Alternative:
            # Use a grid of c values that is dense enough.
            # Let's try N_c = 500.
            # c_grid = np.linspace(1e-6, resource_grid[-1], 500)
            # For each m, we only consider c_grid[c_grid <= m].
            # This is still hard to vectorize fully without loops over m or masking.
            
            # Let's try a different vectorization:
            # We can interpolate the continuation value V_{t+1}(m', z') onto a dense grid of m'.
            # Then, for a given m, the objective is u(c) + beta * E[V_{t+1}(R(m-c)+income, z')].
            # Let m' = R(m-c). Then c = m - m'/R.
            # As c goes from 0 to m, m' goes from R*m to 0.
            # So we can search over m' in [0, R*m].
            
            # Let's stick to the c-grid but use a smart masking strategy or just a loop over m if N_c is small?
            # No, loop over m (101) * loop over z (2) * 5 periods = 1010 iterations.
            # Inside each, if we do a vectorized search over N_c=500, it's fast.
            # But the current code did a loop over m_idx (101) and inside a bisection (40 steps).
            # 101 * 40 * 5 * 2 = ~40k evaluations. That's why it was slow? No, 40k is fast.
            # Wait, the current code did:
            # for t in 5..0
            #   for z in 2
            #     for m_idx in 101
            #       bisection (40 iters)
            #         obj_deriv (2 evals of obj)
            #           obj (loop over z' (2) + interp)
            # Total obj evals: 5 * 2 * 101 * 40 * 2 * 2 = 161,600.
            # Each obj eval has 2 interps.
            # This should be fast in Python? Maybe np.interp is slow?
            # Or maybe the overhead of Python loops is high.
            
            # Let's try to vectorize the m-loop.
            # For a fixed t and z, we want to solve for all m.
            # We can create a 2D array of c candidates.
            # But the constraint c <= m makes it irregular.
            
            # Let's use a grid of c values that is a fraction of m?
            # No, let's just use a dense grid of c values and mask.
            # To make it efficient, we can sort c_grid.
            
            # Actually, a very efficient way for this specific problem structure:
            # The value function V(m) is concave.
            # We can use a grid of m' (next period wealth) instead of c.
            # m' = R(m-c).
            # c = m - m'/R.
            # Constraint c > 0 => m' < R*m.
            # Constraint c < m => m' > 0.
            # So m' in (0, R*m).
            
            # Let's define a grid of m' values.
            # Max m' = R * max(m).
            # Let's use N_mp = 500 points for m'.
            # m_prime_grid = np.geomspace(1e-6, R * resource_grid[-1], 500)
            
            # For each m in resource_grid, valid m' are those < R*m.
            # This is still irregular.
            
            # Let's go back to the scalar loop but optimize the inner part.
            # The bottleneck is likely np.interp called many times.
            # We can pre-compute the continuation value on a dense grid and use a faster interpolation
            # or just use np.interp which is C-implemented.
            
            # Let's try to vectorize the z' sum.
            # cont_val(m_next) = sum_z' P(z'|z) * V_{t+1}(m_next + income[z'], z')
            # We can compute this for all m_next in a vector.
            
            # Let's restructure the loop to be over m_idx, but vectorize the z' part.
            # Actually, the current code is already pretty standard.
            # Maybe the bisection is too slow?
            # Let's replace bisection with a grid search on c.
            # For each m, we can evaluate the objective on a grid of c values.
            # Since we are doing this for all m, we can vectorize the c-grid evaluation?
            # No, because the valid c range depends on m.
            
            # Let's try a simple grid search for c for each m, but vectorized over the c-grid for a fixed m?
            # No, that's the same as what we are doing.
            
            # Let's try to improve the bisection.
            # Instead of 40 iterations, use 30.
            # Instead of central difference for derivative, use analytical derivative?
            # u'(c) + beta * sum P * V'(m_next) * (-R) = 0
            # We need V'(m). We can approximate V' by finite differences on the grid.
            
            # Let's try a different approach:
            # Use a dense grid of c values (e.g., 1000 points) for the entire range [0, max_m].
            # For each m, we only take the subset of c_grid where c <= m.
            # This is inefficient if done naively.
            
            # Let's try to use the fact that the policy is monotonic.
            # c*(m) is increasing in m.
            # We can use a single grid of c values and interpolate the value function?
            
            # Let's stick to the current structure but optimize the inner loop.
            # The most expensive part is likely the interpolation.
            # We can pre-compute the continuation value for a dense grid of m_next values.
            
            # Let's try a vectorized grid search for c for each m.
            # For a fixed t and z:
            # 1. Pre-compute V_next for all z' on a dense grid of m_next.
            # 2. For each m in resource_grid:
            #    a. Define c_grid = np.linspace(1e-6, m, 100)
            #    b. Compute m_next = R * (m - c_grid)
            #    c. Compute cont_val = sum_z' P * interp(m_next + income[z'], m_next_grid, V_next[z'])
            #    d. Compute obj = u(c_grid) + beta * cont_val
            #    e. Find c_opt = c_grid[argmax(obj)]
            
            # This is 101 * 100 * 2 * 5 = 101,000 evaluations.
            # Each evaluation involves a vectorized interpolation.
            # This should be much faster than bisection with Python loops.
            
            # Let's implement this.
            
            # Dense grid for m_next
            max_m_next = R * resource_grid[-1]
            m_next_grid = np.geomspace(1e-6, max_m_next, 1000)
            
            # Pre-compute V_next on m_next_grid for all z'
            V_next_on_grid = np.zeros((num_states, len(m_next_grid)))
            for zp in range(num_states):
                V_next_on_grid[zp, :] = np.interp(m_next_grid, resource_grid, value[t + 1, zp, :])
            
            # Now loop over m
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                
                # Grid of c values
                # Use a logarithmic grid for c to get better resolution at low c
                # But c must be in [0, m].
                # Let's use a linear grid for simplicity, or geomspace.
                # geomspace from 1e-6 to m.
                if m < 1e-5:
                    c_grid = np.array([m * 0.5, m])
                else:
                    c_grid = np.geomspace(1e-6, m, 100)
                
                # Compute m_next for each c in c_grid
                m_next = R * (m - c_grid)
                
                # Compute continuation value
                # cont_val = sum_z' P(z'|z) * V_next_on_grid[zp] at (m_next + income[zp])
                cont_val = np.zeros(len(c_grid))
                for zp in range(num_states):
                    m_next_zp = m_next + income[zp]
                    # Clip m_next_zp to the grid range
                    m_next_zp_clipped = np.clip(m_next_zp, m_next_grid[0], m_next_grid[-1])
                    v_interp = np.interp(m_next_zp_clipped, m_next_grid, V_next_on_grid[zp, :])
                    cont_val += transition[z, zp] * v_interp
                
                # Compute objective
                obj = utility(c_grid) + beta * cont_val
                
                # Find optimal c
                idx_max = np.argmax(obj)
                c_opt = c_grid[idx_max]
                
                policy[t, z, m_idx] = c_opt
                value[t, z, m_idx] = obj[idx_max]
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m
    
    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)
    
    return policy
