import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']

    m_grid_out = np.array(resource_grid)
    n_m_out = len(m_grid_out)
    n_z = len(income)

    # Internal grid for value function - use a fine grid
    # Since the algorithm is fast, we can afford a larger grid
    n_m_int = 8000
    m_internal = np.geomspace(0.001, 50, n_m_int)

    # Utility and Marginal Utility functions
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def marg_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # Value function marginal storage: V_prime[t, z, m_idx]
    # We don't really need V itself, just V_prime for the Euler equation
    # And C for the final output.
    V_prime = np.zeros((horizon, n_z, n_m_int))
    C = np.zeros((horizon, n_z, n_m_int))

    # Terminal period: consume everything
    C[horizon - 1, :, :] = m_internal[np.newaxis, :]
    V_prime[horizon - 1, :, :] = marg_utility(m_internal)[np.newaxis, :]

    # Pre-compute transition probabilities weighted by beta
    beta_trans = beta * transition

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # Initialize C with a guess. 
            # A good guess is the previous period's policy interpolated, or just m.
            # Let's use the previous period's C as a starting point if available, 
            # but since we are going backward, t+1 is the "previous" in time but "next" in calc.
            # We can use C[t+1, z, :] as a rough guess, but m changes.
            # Simple guess: C = m_internal * 0.5
            # Better guess: Use the result from the last iteration of the fixed point loop
            # But for the first m, we need a start.
            
            c_curr = np.full(n_m_int, 0.5) * m_internal
            
            # Fixed point iteration for the Euler equation
            # u'(c) = beta * R * sum( P(z'|z) * V_prime[t+1, z', R(m-c)+y] )
            
            # Number of iterations
            n_iter = 50
            
            for _ in range(n_iter):
                # Compute m_next for all z'
                # m_next[z'] = R * (m_internal - c_curr) + income[z']
                # Shape: (n_z, n_m_int)
                m_next = R * (m_internal[np.newaxis, :] - c_curr[np.newaxis, :]) + income[:, np.newaxis]
                
                # Clip m_next to the internal grid range for interpolation
                m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                
                # Interpolate V_prime[t+1, z', :] at m_next
                # V_prime_next[z', m_idx]
                V_prime_next = np.zeros_like(m_next)
                for zp in range(n_z):
                    V_prime_next[zp, :] = np.interp(m_next_clipped[zp, :], m_internal, V_prime[t + 1, zp, :])
                
                # Compute Target = beta * R * sum( transition[z, zp] * V_prime_next[zp, :] )
                # transition[z, :] shape (n_z,). V_prime_next shape (n_z, n_m_int).
                # We need to sum over zp.
                # target = beta * R * np.sum(transition[z, :, np.newaxis] * V_prime_next, axis=0)
                
                # Optimization: precompute weights
                # weights = beta * R * transition[z, :]
                # target = np.dot(weights, V_prime_next)
                
                target = np.dot(beta * R * transition[z, :], V_prime_next)
                
                # Update c_curr
                # u'(c) = target => c = target^(-1/gamma)
                if abs(gamma - 1.0) < 1e-10:
                    c_new = 1.0 / target
                else:
                    c_new = np.power(target, -1.0 / gamma)
                
                # Clip to feasible region [1e-12, m_internal]
                c_new = np.clip(c_new, 1e-12, m_internal)
                
                # Damping if necessary? 
                # For standard params, it converges.
                # If it oscillates, we might need damping. 
                # Let's check for divergence.
                # If max change is huge, damp it.
                # diff = c_new - c_curr
                # if np.any(np.abs(diff) > 0.1 * m_internal):
                #    c_new = c_curr + 0.5 * diff
                # c_curr = c_new
                c_curr = c_new
                
            C[t, z, :] = c_curr
            V_prime[t, z, :] = marg_utility(c_curr)

        # Time check
        if time.time() - start_time > deadline - 5:
            break

    # Interpolate policy C from m_internal to m_grid_out
    consumption = np.zeros((horizon, n_z, n_m_out))

    for t in range(horizon):
        for z in range(n_z):
            c_out = np.interp(m_grid_out, m_internal, C[t, z, :])
            consumption[t, z, :] = c_out

            # Enforce constraints on the output grid
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid_out)
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)

    # Terminal period must consume everything exactly
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]

    return consumption
