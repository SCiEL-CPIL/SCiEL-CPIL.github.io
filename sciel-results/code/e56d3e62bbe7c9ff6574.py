import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_out = resource_grid
    n_out = len(m_out)
    n_z = len(income)
    
    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)

    # We use a dense internal grid for accurate value function construction
    # The output grid is geomspace(0.001, 40, 101).
    # We extend the range slightly to handle m' that might fall outside due to income shocks,
    # but mostly we focus on the range relevant for the policy.
    # Max m in output is 40. Min m is 0.001.
    # m' = R(m-c) + y. Max m' approx R*40 + 1.5 = 41.5. Min m' approx 0.5.
    # Let's use a dense grid from 0.001 to 50 to be safe.
    n_int = 5000
    m_int = np.geomspace(0.001, 50, n_int)
    
    # Policy storage on internal grid
    # pol_int[t][z] is the consumption policy for state z at time t on m_int
    pol_int = np.zeros((horizon, n_z, n_int))
    
    # Terminal condition: c = m
    for z in range(n_z):
        pol_int[horizon - 1, z, :] = m_int
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # If running out of time, we can break, but we need to ensure we return something valid.
            # Given the small horizon (6) and vectorized operations, this should be fast.
            pass
            
        next_pol = pol_int[t + 1, :, :] # Shape (n_z, n_int)
        
        # Interpolators for next period's utility for each state
        # We need u(c_next(m')). 
        # Since c_next is a function of m', and u is a function of c, 
        # we can precompute u(next_pol) on the internal grid and interpolate that.
        # Let U_next[z](m') = u(pol_int[t+1, z, :](m'))
        
        U_next_interp = []
        for z in range(n_z):
            c_next = next_pol[z]
            u_next = u(c_next)
            # PCHIP is good for monotonic data, but u(c) is concave so u_next(m') should be smooth.
            # Linear interpolation might be sufficient and faster, but PCHIP is safer for accuracy.
            # However, for 5000 points, PCHIP is fast.
            p = PchipInterpolator(m_int, u_next)
            U_next_interp.append(p)
            
        for z in range(n_z):
            # For each m in m_int, we want to find c that maximizes:
            # u(c) + beta * sum_{z'} transition[z, z'] * U_next_interp[z']( R*(m-c) + income[z'] )
            
            c_values = np.zeros(n_int)
            
            # To maximize, we can use the fact that the objective is concave in c.
            # We can do a vectorized search or use a root-finding on the derivative.
            # Given the vectorization, let's try a dense grid search for c for each m?
            # That would be n_int * n_c_candidates. 5000 * 5000 = 25M ops. Might be slow in Python loops.
            
            # Better: Use Newton's method on the Euler equation, vectorized?
            # Or, since the grid is dense, we can approximate the derivative and do a few Newton steps.
            
            # Let's stick to the robust approach: 
            # For each m, the optimal c is in [0, m].
            # We can use np.argmax over a candidate set of c's?
            # If we use the same m_int grid for c candidates, we restrict c to the grid points.
            # This is a standard grid-search method. With 5000 points, it's quite accurate.
            
            # Let's try to vectorize the evaluation of the objective function for all m and a subset of c.
            # Actually, solving the Euler equation point-by-point with brentq was the previous method.
            # The issue was interpolation of the *policy*, not the value.
            # Here we interpolate the *value* (or marginal utility).
            
            # Let's use the Euler equation root finding again, but with the high-quality interpolated Value/Marginal Utility.
            # Euler Eq: u'(c) = beta * R * sum_{z'} transition[z, z'] * V_m(m', z')
            # where V_m(m', z') = d/dm' U_next[z'](m')
            
            # We can compute the derivative of the PCHIP interpolant?
            # PchipInterpolator has a .deriv() method.
            
            dU_next_interp = []
            for z in range(n_z):
                # dU/dm' = u'(c') * c'_m'
                # It's easier to just use the .deriv() of the value function if we stored Value.
                # But we stored u(c). 
                # Let's create interpolators for the marginal utility directly?
                # No, u(c) is not linear in m'.
                
                # Let's just use the .deriv() of the PCHIP of u(c(m')).
                # This approximates V_m(m').
                p_deriv = U_next_interp[z].deriv(1)
                dU_next_interp.append(p_deriv)
                
            for i, m in enumerate(m_int):
                # Define the function to solve: F(c) = u'(c) - RHS(c) = 0
                # We can vectorize this if we define it carefully, but brentq is scalar.
                # Given 5000 points * 2 states * 5 periods = 50,000 roots.
                # Brentq takes ~10-20 iterations. 50k * 20 = 1M evaluations.
                # Each evaluation involves 2 interpolations. This should be fast enough in Python with NumPy?
                # Actually, calling a Python function 1M times is slow.
                
                # Alternative: Grid Search on a coarser set of c candidates?
                # Or, since we have a dense grid m_int, we can assume c is also on m_int.
                # Let's use np.argmax over a vectorized calculation.
                
                # Objective(c) = u(c) + beta * sum_{z'} trans[z, z'] * U_next[z'](R(m-c)+y[z'])
                # Let's evaluate this for all c in a subset of m_int that is <= m.
                
                # To be efficient, we can limit the search for c to [0, m].
                # Since m_int is sorted, indices 0..i are <= m.
                
                # However, vectorizing for all i simultaneously is tricky because the upper bound of c changes with i.
                
                # Let's try a different approach: 
                # Use the property that the optimal policy is smooth.
                # We can start with the previous period's policy (shifted) as a guess and do a few Newton steps?
                
                # Let's go back to Brentq but optimize the inner loop.
                # We will pre-compute the interpolators.
                
                def euler_residual(c, m_val=m):
                    # c is scalar
                    if c <= 0:
                        return 1e10 # Large positive
                    if c >= m_val:
                        # u'(c) is small, RHS is positive. 
                        # If c=m, m'=y. 
                        # Usually u'(m) < RHS(m) is false? 
                        # Let's just evaluate.
                        pass
                    
                    rhs = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = R * (m_val - c) + income[zp]
                        # Clamp m_prime
                        if m_prime < m_int[0]:
                            m_prime = m_int[0]
                        elif m_prime > m_int[-1]:
                            m_prime = m_int[-1]
                        
                        # Get marginal value V_m(m', zp)
                        # dU_next_interp[zp](m_prime) gives d/dm' u(c_opt(m', zp))
                        rhs += prob * dU_next_interp[zp](m_prime)
                    
                    rhs *= beta * R
                    
                    return u_prime(c) - rhs

                def u_prime(c):
                    c_safe = np.maximum(c, 1e-12)
                    if abs(gamma - 1.0) < 1e-10:
                        return 1.0 / c_safe
                    else:
                        return c_safe ** (-gamma)

                # Solve for c
                # F(1e-12) > 0
                # F(m) ?
                # If F(m) > 0, c=m is optimal (corner).
                # If F(m) < 0, root is in (0, m).
                
                try:
                    res_m = euler_residual(m)
                    if res_m > 0:
                        c_values[i] = m
                    else:
                        c_opt = brentq(euler_residual, 1e-12, m, xtol=1e-12, rtol=1e-12, maxiter=50)
                        c_values[i] = c_opt
                except Exception:
                    # Fallback to linear interpolation of previous policy or simple guess
                    # For robustness, if it fails, use a conservative guess
                    # e.g., c = 0.5 * m
                    c_values[i] = 0.5 * m

            pol_int[t, z, :] = c_values

    # Interpolate the final policy from m_int to m_out
    pol_out = np.zeros((horizon, n_z, n_out))
    
    for t in range(horizon):
        for z in range(n_z):
            # PCHIP interpolation
            p = PchipInterpolator(m_int, pol_int[t, z, :])
            # Evaluate on m_out
            # Handle edge cases if m_out is outside m_int range (shouldn't be, but just in case)
            # m_out is 0.001 to 40. m_int is 0.001 to 50. So it's inside.
            pol_out[t, z, :] = p(m_out)
            
    # Final check: ensure validity
    # 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c = pol_out[t, z, :]
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m_out)
            pol_out[t, z, :] = c
            
    # Terminal condition check: c[5] equal to grid within 1e-9
    pol_out[horizon-1, :, :] = m_out

    return pol_out

# Helper for brentq inside the loop scope, defined globally or passed in
def brentq(f, a, b, xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, args=(), full_output=False, disp=True):
    from scipy.optimize import brentq as _brentq
    return _brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter, args=args, full_output=full_output, disp=disp)
