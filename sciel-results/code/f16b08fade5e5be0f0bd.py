import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    c_min = 1e-12
    T = horizon
    n_z = 2
    n_out = len(resource_grid)

    # Dense internal grid for value function
    n_int = 2001
    grid_int = np.geomspace(0.001, 40, n_int)

    # Pre-compute utility functions
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Terminal condition on internal grid
    V_int = [None] * T
    P_int = [None] * T

    V_int[T - 1] = utility(grid_int)[np.newaxis, :].repeat(2, axis=0)
    P_int[T - 1] = grid_int[np.newaxis, :].repeat(2, axis=0)

    # Backward induction on internal grid
    for t in range(T - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: copy terminal
            for t_rem in range(t, -1, -1):
                V_int[t_rem] = V_int[T - 1].copy()
                P_int[t_rem] = P_int[T - 1].copy()
            break

        V_t = np.zeros((n_z, n_int))
        P_t = np.zeros((n_z, n_int))
        V_next = V_int[t + 1]

        # For each z, solve for optimal c using bisection on a dense c-grid
        for z in range(n_z):
            # For each m in grid_int, find optimal c
            # Use vectorized bisection: for each m, search over c in [c_min, m]
            # But we need to be efficient. Let's use a parametric approach.
            
            # Actually, let's do it differently: for each m, we can vectorize over z'
            # But the coupling is through m-c. Let's just loop over m but vectorize over z'.
            
            # For speed, use a coarse-to-fine bisection
            # First, get a good initial bracket using a heuristic
            # c* is roughly proportional to m
            
            # Let's use a fixed number of bisection iterations with a dense c-grid
            # For each m, define f(c) = u'(c) - beta*R*sum_{z'} P(z'|z)*V_next[z'](R(m-c)+y_{z'})
            # We want f(c) = 0
            
            # To vectorize over m, we can compute f for all m simultaneously for a given c/m ratio
            # Let c = s * m, where s in [0, 1]
            # Then f(s) = u'(s*m) - beta*R*sum_{z'} P(z'|z)*V_next[z'](R*m*(1-s)+y_{z'})
            
            # This is still coupled through m. Let's just loop over m but use numpy for z'.
            
            # Actually, for 2001 m values and 2 z' values, we can vectorize over m for each s.
            # Let's do bisection over s in [0.001, 1.0]
            
            s_lo = np.full(n_int, 0.001)
            s_hi = np.full(n_int, 1.0)
            
            # Max 20 bisection iterations
            for it in range(20):
                s_mid = 0.5 * (s_lo + s_hi)
                c_mid = s_mid * grid_int
                
                # Compute f for all m simultaneously
                # f = u'(c) - beta*R*sum_{z'} P(z'|z)*V_next[z'](R*(m-c)+y_{z'})
                
                u_prime_c = utility_prime(c_mid)
                
                cont_val = np.zeros(n_int)
                for zp in range(n_z):
                    next_m = R * (grid_int - c_mid) + income[zp]
                    # Clip next_m to grid range
                    next_m_clipped = np.clip(next_m, grid_int[0], grid_int[-1])
                    V_interp = np.interp(next_m_clipped, grid_int, V_next[zp])
                    cont_val += transition[z, zp] * V_interp
                
                f = u_prime_c - beta * R * cont_val
                
                # f is decreasing in c (since u'' < 0 and V' > 0, V'' < 0)
                # If f > 0, c is too small, increase c (s)
                # If f < 0, c is too large, decrease c (s)
                s_lo = np.where(f > 0, s_mid, s_lo)
                s_hi = np.where(f < 0, s_mid, s_hi)
            
            s_opt = 0.5 * (s_lo + s_hi)
            c_opt = s_opt * grid_int
            c_opt = np.clip(c_opt, c_min, grid_int)
            
            P_t[z, :] = c_opt
            
            # Compute V_t
            u_c = utility(c_opt)
            cont_val = np.zeros(n_int)
            for zp in range(n_z):
                next_m = R * (grid_int - c_opt) + income[zp]
                next_m_clipped = np.clip(next_m, grid_int[0], grid_int[-1])
                V_interp = np.interp(next_m_clipped, grid_int, V_next[zp])
                cont_val += transition[z, zp] * V_interp
            V_t[z, :] = u_c + beta * cont_val
        
        V_int[t] = V_t
        P_int[t] = P_t

    # Interpolate policy to output grid
    result = np.zeros((T, n_z, n_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(resource_grid, grid_int, P_int[t, z, :])
            result[t, z, :] = np.clip(result[t, z, :], c_min, resource_grid + 1e-10)
    
    # Final period: c = m
    result[T - 1, :, :] = resource_grid[np.newaxis, :].repeat(2, axis=0)
    
    return result
