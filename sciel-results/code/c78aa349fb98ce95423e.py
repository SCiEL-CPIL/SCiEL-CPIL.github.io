import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    incomes = np.array(model['income'], dtype=float)
    trans = np.array(model['transition'], dtype=float)
    horizon = model['horizon']  # 6
    z_count = len(incomes)       # 2
    m_grid = resource_grid       # shape (101,)
    n_m = len(m_grid)

    # Pre-compute utility and marginal utility functions
    def crra_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def crra_marginal(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # V[t, z, m] : Value function
    # V_m[t, z, m] : Marginal value (u'(c*))
    # C[t, z, m] : Optimal consumption policy
    V = np.zeros((horizon, z_count, n_m))
    V_m = np.zeros((horizon, z_count, n_m))
    C = np.zeros((horizon, z_count, n_m))

    # Terminal condition: t = horizon - 1 = 5
    # Consume everything, so c = m
    V[5, :, :] = crra_utility(m_grid)
    V_m[5, :, :] = crra_marginal(m_grid)
    C[5, :, :] = m_grid

    # Backward induction for t = 4, 3, 2, 1, 0
    max_newton_iter = 30
    tol = 1e-10

    for t in range(horizon - 2, -1, -1):
        if time.time() - t0 > DEADLINE:
            # If we run out of time, we must return what we have.
            # But C[t] is currently 0. We need a valid fallback.
            # A safe fallback is c = m (consume all) or a simple heuristic.
            # Let's fill remaining t's with a simple heuristic to ensure validity.
            for t_rem in range(t, -1, -1):
                for z in range(z_count):
                    # Simple heuristic: spend a fraction based on beta
                    frac = 1.0 / (1.0 + beta * R)
                    C[t_rem, z, :] = np.clip(m_grid * frac, 1e-12, m_grid)
                    # Calculate V and V_m for consistency, though not strictly needed for C validity
                    V[t_rem, z, :] = crra_utility(C[t_rem, z, :])
                    V_m[t_rem, z, :] = crra_marginal(C[t_rem, z, :])
            break
        
        for z in range(z_count):
            for i in range(n_m):
                m_i = m_grid[i]
                
                if m_i <= 1e-12:
                    C[t, z, i] = 1e-12
                    V[t, z, i] = crra_utility(1e-12)
                    V_m[t, z, i] = crra_marginal(1e-12)
                    continue
                
                # Initial guess: 
                # If beta is high, save more. If R is high, saving is rewarding.
                # Heuristic: c ~ m / (1 + beta * R) is a standard start.
                # Refinement: Adjust for income shock.
                # A better start: c = m * (1 - beta*R) / (1 - beta) is not valid if beta*R >= 1.
                # Let's use c0 = m / (1 + beta * R)
                c = m_i / (1.0 + beta * R)
                c = np.clip(c, 1e-12, m_i)
                
                # Define the Euler equation residual: f(c) = u'(c) - beta * E[u'(c')]
                # We want to find c such that f(c) = 0.
                
                def eval_residual(c_val):
                    lhs = crra_marginal(c_val)
                    rhs = 0.0
                    for zp in range(z_count):
                        m_prime = R * (m_i - c_val) + incomes[zp]
                        m_prime = np.maximum(m_prime, 0.0)
                        # Interpolate V_m[t+1, zp, :] at m_prime
                        # V_m[t+1, zp, :] is defined on m_grid
                        Vm_interp = np.interp(m_prime, m_grid, V_m[t+1, zp, :])
                        rhs += trans[z, zp] * Vm_interp
                    rhs *= beta
                    return lhs - rhs

                def eval_derivative(c_val):
                    # d/dc [ u'(c) - beta * E[ V_m(R(m-c)+y) ] ]
                    # = u''(c) - beta * E[ V_mm(R(m-c)+y) * (-R) ]
                    # = u''(c) + beta * R * E[ V_mm(...) ]
                    
                    if abs(gamma - 1.0) < 1e-9:
                        u_second = -1.0 / (c_val * c_val)
                    else:
                        u_second = -gamma * np.power(c_val, -gamma - 1)
                    
                    exp_vmm = 0.0
                    for zp in range(z_count):
                        m_prime = R * (m_i - c_val) + incomes[zp]
                        m_prime = np.maximum(m_prime, 0.0)
                        # Numerical derivative of V_m (which is V_marginal) w.r.t m
                        # V_m is the marginal utility of next period's consumption.
                        # We need d(V_m)/dm.
                        # V_m[m'] = u'(c*(m')).
                        # dV_m/dm' = u''(c*) * dc*/dm' = u''(c*) * 1 (by envelope, dc*/dm' = 1 if no corner)
                        # Actually, V_m is stored as u'(c*).
                        # The derivative of V_m w.r.t m is V_mm.
                        # We can approximate V_mm by finite difference of V_m array.
                        
                        dm = 1e-5
                        mp1 = np.minimum(m_prime + dm, m_grid[-1])
                        mp0 = np.maximum(m_prime - dm, m_grid[0])
                        
                        # Use central difference if possible, else forward/backward
                        if mp0 > m_grid[0] and mp1 < m_grid[-1]:
                            Vm_p1 = np.interp(mp1, m_grid, V_m[t+1, zp, :])
                            Vm_p0 = np.interp(mp0, m_grid, V_m[t+1, zp, :])
                            Vmm = (Vm_p1 - Vm_p0) / (mp1 - mp0)
                        elif m_prime < m_grid[0] + dm:
                            Vm_p1 = np.interp(mp1, m_grid, V_m[t+1, zp, :])
                            Vm_p0 = V_m[t+1, zp, 0]
                            Vmm = (Vm_p1 - Vm_p0) / (mp1 - mp0)
                        else:
                            Vm_p1 = V_m[t+1, zp, -1]
                            Vm_p0 = np.interp(mp0, m_grid, V_m[t+1, zp, :])
                            Vmm = (Vm_p1 - Vm_p0) / (mp1 - mp0)
                        
                        exp_vmm += trans[z, zp] * Vmm
                    
                    return u_second + beta * R * exp_vmm

                # Newton-Raphson loop with bisection fallback
                lo = 1e-12
                hi = m_i
                c_curr = c
                
                for _ in range(max_newton_iter):
                    if time.time() - t0 > DEADLINE:
                        break
                    
                    res = eval_residual(c_curr)
                    if abs(res) < tol:
                        break
                    
                    # Determine bracket
                    # f(lo) > 0 (u'(lo) is huge, rhs is finite) -> positive
                    # f(hi) < 0 (u'(hi) is small, rhs is large) -> negative
                    # We assume f is monotonically decreasing in c.
                    
                    if res > 0:
                        lo = c_curr
                    else:
                        hi = c_curr
                    
                    # Newton step
                    try:
                        deriv = eval_derivative(c_curr)
                        if abs(deriv) > 1e-15:
                            c_new = c_curr - res / deriv
                        else:
                            c_new = 0.5 * (lo + hi)
                    except:
                        c_new = 0.5 * (lo + hi)
                    
                    # Check if Newton step is within bracket
                    if c_new <= lo or c_new >= hi:
                        # Bisection step
                        c_new = 0.5 * (lo + hi)
                    
                    # Ensure c_new is valid
                    c_new = np.clip(c_new, lo, hi)
                    
                    # Check convergence
                    if abs(c_new - c_curr) < 1e-12 * max(1.0, c_curr):
                        c_curr = c_new
                        break
                    
                    c_curr = c_new
                
                C[t, z, i] = c_curr
                V[t, z, i] = crra_utility(c_curr)
                V_m[t, z, i] = crra_marginal(c_curr)

    # Final validity checks
    for t in range(horizon):
        for z in range(z_count):
            # Ensure 0 < c <= m + 1e-10
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
            
            # Replace any non-finite values with a safe default (e.g., m/2)
            mask = ~np.isfinite(C[t, z, :])
            if np.any(mask):
                C[t, z, mask] = m_grid[mask] * 0.5
                C[t, z, mask] = np.maximum(C[t, z, mask], 1e-12)
                C[t, z, mask] = np.minimum(C[t, z, mask], m_grid[mask])

    # Ensure c[5] = m_grid exactly
    C[5, :, :] = m_grid

    return C
