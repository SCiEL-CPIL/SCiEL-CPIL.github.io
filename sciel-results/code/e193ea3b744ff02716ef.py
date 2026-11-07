import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_z = len(income)
    
    # Fine grid for computation
    # Extend max m to cover potential m' values
    # m' = R(m-c) + income[z]. Max m=40, c=0, income=1.5 -> 1.02*40+1.5 = 42.3
    m_max_fine = 45.0
    m_min_fine = 0.001
    N_fine = 2000
    m_fine = np.geomspace(m_min_fine, m_max_fine, N_fine)
    
    # Utility and derivative
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # C[t, z, m_idx]
    C = np.zeros((horizon, N_z, len(resource_grid)))
    
    # We compute on m_fine first, then interpolate to resource_grid
    # C_fine[t, z, m_idx_fine]
    C_fine = np.zeros((horizon, N_z, N_fine))
    
    # Initialize last period t = horizon - 1
    t = horizon - 1
    for z in range(N_z):
        C_fine[t, z, :] = m_fine
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Prepare splines or interpolators for C_next (which is C_fine[t+1])
        # We will use np.interp for speed and robustness, or CubicSpline for accuracy.
        # Let's use CubicSpline for C_next to get smooth policies.
        C_next_splines = []
        for z in range(N_z):
            C_next_splines.append(CubicSpline(m_fine, C_fine[t+1, z, :], extrapolate=False))
            
        for z in range(N_z):
            C_curr_fine = np.zeros(N_fine)
            for i_m in range(N_fine):
                m = m_fine[i_m]
                
                def euler_diff(c_val):
                    # c_val is scalar
                    m_prime = R * (m - c_val) + income
                    # Clip m_prime to range of m_fine for interpolation?
                    # CubicSpline with extrapolate=False returns NaN outside.
                    # We need to handle m_prime outside [m_min_fine, m_max_fine].
                    # m_prime_min = 0.5 (inside).
                    # m_prime_max = 42.3 (inside 45).
                    # So we are safe.
                    
                    # Get c_next for each z_prime
                    c_next_vals = np.zeros(N_z)
                    for zp in range(N_z):
                        mp = m_prime[zp]
                        # np.clip to be safe, though should be in range
                        mp_clipped = np.clip(mp, m_fine[0], m_fine[-1])
                        c_next_vals[zp] = C_next_splines[zp](mp_clipped)
                        
                    # Calculate expected marginal utility of future consumption
                    # E[ R * u'(c_next) ]
                    # Note: The Euler equation is u'(c_t) = beta * E[ R * u'(c_{t+1}) ]
                    # Wait, the standard Euler equation for this model:
                    # u'(c_t) = beta * E[ R * u'(c_{t+1}) ]
                    # Yes, because m' = R(m-c) + y. dm'/dc = -R.
                    # d/d c [ u(c) + beta V(m') ] = u'(c) + beta V'(m') * (-R) = 0
                    # V'(m') = E[ u'(c_{t+1}) * ... ]? 
                    # Actually V'(m) = E[ u'(c_{t+1}(m')) * R ]? 
                    # Let's derive:
                    # V_t(m) = max_c u(c) + beta E[ V_{t+1}(R(m-c) + y') ]
                    # FOC: u'(c) - beta E[ V'_{t+1}(m') * R ] = 0
                    # u'(c) = beta R E[ V'_{t+1}(m') ]
                    # By envelope theorem, V'_{t+1}(m') = E[ u'(c_{t+1}(m'')) * ... ]?
                    # Actually, V'_{t+1}(m) is the marginal value of wealth.
                    # It is equal to E[ u'(c_{t+1}) * (dm_{t+2}/dm_{t+1}) ]?
                    # No, simpler:
                    # V_{t+1}(m) = max_{c'} u(c') + beta E[V_{t+2}(...)]
                    # V'_{t+1}(m) = u'(c'_{opt})? No.
                    # The marginal value of m is the marginal utility of the optimal consumption?
                    # No, it's the shadow price.
                    # However, for the Euler equation linking t and t+1:
                    # u'(c_t) = beta * R * E[ V'_{t+1}(m') ]
                    # And V'_{t+1}(m') is the marginal utility of consumption in t+1?
                    # Yes, if we are in an interior solution, V'(m) = u'(c_opt(m)).
                    # So u'(c_t) = beta * R * E[ u'(c_{t+1}) ]
                    
                    exp_mu = 0.0
                    for zp in range(N_z):
                        prob = transition[z, zp]
                        if prob == 0: continue
                        mu_next = util_prime(c_next_vals[zp])
                        exp_mu += prob * mu_next
                        
                    return util_prime(c_val) - beta * R * exp_mu

                # Solve for c
                # Bounds: [1e-12, m]
                c_low = 1e-12
                c_high = m
                
                # Check boundaries
                val_low = euler_diff(c_low)
                val_high = euler_diff(c_high)
                
                if val_high > 0:
                    # Derivative is positive at m, so we want to increase c, but can't.
                    # Optimum is at boundary c = m.
                    c_opt = m
                elif val_low < 0:
                    # Derivative is negative at low c, so we want to decrease c, but can't.
                    # Optimum is at boundary c = 1e-12.
                    c_opt = c_low
                else:
                    # Root exists
                    try:
                        c_opt = brentq(euler_diff, c_low, c_high, xtol=1e-10)
                    except ValueError:
                        # Fallback to minimize_scalar if brentq fails
                        from scipy.optimize import minimize_scalar
                        res = minimize_scalar(lambda x: -util(x) - beta * sum(transition[z, zp] * 0 for zp in range(N_z)), bounds=(c_low, c_high), method='bounded')
                        # This fallback is bad, better to just use midpoint or linear search?
                        # Let's just use a simple grid search fallback
                        c_grid = np.linspace(c_low, c_high, 50)
                        vals = [euler_diff(c) for c in c_grid]
                        # Find sign change
                        idx = 0
                        for i in range(len(c_grid)-1):
                            if (vals[i] - 0) * (vals[i+1] - 0) <= 0:
                                idx = i
                                break
                        c_opt = c_grid[idx]
                        
                C_curr_fine[i_m] = c_opt
            
            C_fine[t, z, :] = C_curr_fine
    
    # Interpolate C_fine to resource_grid
    for t in range(horizon):
        for z in range(N_z):
            # C_fine[t, z, :] is on m_fine
            # We want C[t, z, :] on resource_grid
            C[t, z, :] = np.interp(resource_grid, m_fine, C_fine[t, z, :])
            
    return C
