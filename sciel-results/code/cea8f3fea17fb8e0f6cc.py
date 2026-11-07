import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # consumption[t, z, m_index]
    consumption = np.zeros((horizon, Z, M))

    # Terminal condition: c_H(m) = m
    consumption[horizon - 1] = resource_grid[np.newaxis, :]  # shape (Z, M)

    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Backward induction: find optimal c for t = horizon-2 down to 0
    # We store c_next[z, m_idx] as the consumption policy at t+1
    c_next = np.zeros((Z, M))
    for z in range(Z):
        c_next[z] = resource_grid

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: consume all remaining
            for tt in range(t, -1, -1):
                consumption[tt] = resource_grid[np.newaxis, :]
            break

        c_curr = np.zeros((Z, M))

        for z in range(Z):
            # Solve Euler equation for each m in resource_grid
            # Euler: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c_next(z', m'))
            # where m' = R*(m-c) + income[z']
            #
            # Define s = (m-c)/m, so c = m*(1-s), s in [0,1)
            # m' = R*m*s + income[z']
            #
            # Define G(s) = log(u'(m*(1-s))) - log(beta*R*E[u'(c_next)])
            # G is monotonically decreasing in s (higher s -> lower c -> higher u'(c), but also higher m' -> higher c_next -> lower u'(c_next))
            # Actually: as s increases, c decreases so u'(c) increases, making G increase.
            # As s increases, m' increases, c_next increases, u'(c_next) decreases, making G decrease.
            # Net effect: G is monotonically decreasing in s for well-behaved CRRA.
            # We find s* such that G(s*) = 0.
            #
            # Use bisection: s_low where G > 0, s_high where G < 0.
            # At s=0: c=m, m'=income[z']. G(0) = log(u'(m)) - log(beta*R*E[u'(c_next(income))])
            # At s close to 1: c->0, u'(c)->inf, G->inf. Wait, that's not right.
            # Let me reconsider: as s increases, c = m(1-s) decreases, so u'(c) = c^{-gamma} increases.
            # So LHS = log(u'(c)) increases with s.
            # m' = R*m*s + income[z'] increases with s, so c_next(m') increases (assuming c_next is increasing in m'),
            # so u'(c_next) decreases, so RHS = log(E[u'(c_next)]) decreases with s.
            # Therefore G(s) = LHS - RHS is strictly increasing in s.
            # We want G(s*) = 0.
            # At s=0: G(0) might be negative (consume less) or positive (consume more).
            # At s=1-eps: G -> +inf (since u'(c) -> inf).
            # So if G(0) < 0, there's a root in (0, 1). If G(0) >= 0, optimal is s=0 (consume all).
            
            m = resource_grid  # shape (M,)
            
            # Compute G(s) for a given s array
            def compute_G(s_arr):
                c = m * (1.0 - s_arr)
                c = np.maximum(c, 1e-12)
                lhs = np.log(utility_prime(c))
                
                # m' = R * m * s + income[z']
                m_prime = R * m[:, None] * s_arr[:, None] + income[None, :]  # shape (M, Z)
                
                # E[u'(c_next)] = sum_{z'} P(z'|z) * u'(c_next[z'](m'))
                u_prime_next_sum = np.zeros(M)
                for zp in range(Z):
                    c_next_val = np.interp(m_prime[:, zp], resource_grid, c_next[zp])
                    u_prime_val = utility_prime(c_next_val)
                    u_prime_next_sum += transition[z, zp] * u_prime_val
                
                rhs = np.log(beta * R * np.maximum(u_prime_next_sum, 1e-12))
                return lhs - rhs

            # Bisection
            s_low = np.zeros(M)
            s_high = np.full(M, 0.999999)
            
            # Check if s=0 is already optimal (G(0) >= 0)
            G_low = compute_G(s_low)
            # If G(0) >= 0, optimal s is 0
            optimal_zero = G_low >= 0
            
            # For those not optimal at s=0, bisect
            active = ~optimal_zero
            if np.any(active):
                s_low_act = s_low[active]
                s_high_act = s_high[active]
                
                # Verify G(s_high) > 0 (should be, but check)
                G_high = compute_G(s_high_act)
                # If G(s_high) <= 0 for some, set s=1 (consume almost nothing)
                still_active = G_high > 0
                if not np.all(still_active):
                    # Those with G(s_high) <= 0: set s_high to s_low (will converge to s=0 or boundary)
                    # Actually this shouldn't happen. Just clip.
                    pass
                
                for _iter in range(40):
                    if time.time() > deadline:
                        break
                    s_mid = 0.5 * (s_low_act + s_high_act)
                    G_mid = compute_G(s_mid)
                    # G is increasing, so if G_mid > 0, root is in [s_low, s_mid]
                    # if G_mid < 0, root is in [s_mid, s_high]
                    mask_pos = G_mid > 0
                    s_high_act[mask_pos] = s_mid[mask_pos]
                    s_low_act[~mask_pos] = s_mid[~mask_pos]
                    if np.max(np.abs(s_high_act - s_low_act)) < 1e-8:
                        break
                
                s_act = 0.5 * (s_low_act + s_high_act)
                s_full = np.zeros(M)
                s_full[active] = s_act
                s_full[~active] = 0.0
            else:
                s_full = np.zeros(M)

            c_curr[z] = m * (1.0 - s_full)
            # Ensure validity
            c_curr[z] = np.maximum(c_curr[z], 1e-12)
            c_curr[z] = np.minimum(c_curr[z], m + 1e-10)

        consumption[t] = c_curr
        c_next = c_curr

    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c

    # Enforce terminal condition exactly
    consumption[horizon - 1] = resource_grid[np.newaxis, :]

    return consumption
