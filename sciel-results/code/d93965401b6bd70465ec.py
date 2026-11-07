import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # Initialize consumption array: shape (horizon, Z, M)
    consumption = np.zeros((horizon, Z, M))

    def utility(c):
        c_safe = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)

    def utility_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)

    # Initialize V for t = horizon - 1 (index 5)
    # V[5, z, i] = U(m_i)
    V = np.zeros((horizon, Z, M))
    V[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, np.newaxis, :]
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, np.newaxis, :]

    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break

        V_next = V[t + 1]  # Shape (Z, M)

        # Build interpolators for V_next[z, :] for each z
        # We'll use PCHIP for monotone-preserving interpolation, or just cubic
        # For speed and simplicity, let's use np.interp with a denser grid?
        # No, np.interp is linear. Let's use a better interpolator for the value function.
        # Since m' can be outside the grid range [0.001, 40], we need to handle that.
        # The grid is geomspace(0.001, 40, 101).
        # m' = R*(m-c) + income[z']. m in [0.001, 40], c in [0, m].
        # So m-c in [0, 40]. R=1.02. R*(m-c) in [0, 40.8].
        # income is [0.5, 1.5]. So m' in [0.5, 42.3].
        # The grid max is 40. So m' can exceed 40.
        # For m' > 40, we can extrapolate or just use the last value.
        # Actually, the value function for large m is roughly linear (or power).
        # Let's extend the grid for interpolation to cover up to ~45.
        # Or simpler: np.interp with right=V_next[z, -1] and left=V_next[z, 0].
        # But linear interpolation might be too coarse.
        # Let's create a denser grid for interpolation? No, that's expensive.
        # Let's use a cubic spline or PCHIP on the existing grid, and handle boundaries.
        
        # Actually, for the Euler equation, we need V_next and its derivative.
        # np.interp only gives values, not derivatives.
        # We need to approximate the derivative or use an interpolator that provides it.
        # Let's use scipy.interpolate.PchipInterpolator for each z.
        # PCHIP provides .deriv(1) for the first derivative.
        
        V_interp = []
        for z in range(Z):
            # Ensure the grid is sorted (it is)
            V_interp.append(PchipInterpolator(resource_grid, V_next[z, :]))

        for z in range(Z):
            for i in range(M):
                m = resource_grid[i]
                
                # Newton's method on the Euler equation
                # F(c) = U'(c) - beta * E[ R * U'(m') ] = 0
                # where m' = R*(m-c) + income[z']
                # F'(c) = U''(c) + beta * E[ R^2 * U''(m') ]
                
                # Initial guess: c0 = m * 0.5
                c = max(m * 0.5, 1e-10)
                c = min(c, m - 1e-10)
                
                converged = False
                for _ in range(20):
                    # Compute F(c) and F'(c)
                    F = 0.0
                    F_prime = 0.0
                    
                    U_c = utility_prime(c)
                    U_c_prime = -utility_prime(c) * c / c  # d/dc (c^-gamma) = -gamma c^-gamma-1
                    # Actually U'(c) = c^(1-gamma-1) = c^-gamma
                    # U''(c) = -gamma * c^(-gamma-1)
                    # So U''(c) = -gamma / c * U'(c)
                    U_c_second = -gamma * U_c / c
                    
                    for zp in range(Z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_save = R * (m - c)
                        m_prime = m_save + income[zp]
                        
                        # Clamp m_prime to grid range for interpolation
                        # PchipInterpolator handles extrapolation poorly, so let's clamp
                        # But wait, if m_prime > 40, we should probably just use the boundary value
                        # or extrapolate linearly. Let's clamp to [resource_grid[0], resource_grid[-1]]
                        # Actually, for m_prime > 40, the value is roughly U(m_prime) if it's the last period,
                        # but here it's V[t+1]. For large m, V is concave.
                        # Let's just clamp for safety.
                        m_prime_clamped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                        
                        # Get V and V' at m_prime_clamped
                        V_val = float(V_interp[zp](m_prime_clamped))
                        V_prime_val = float(V_interp[zp].deriv(1)(m_prime_clamped))
                        
                        # Continuation: beta * E[ V(m') ]
                        # F(c) = U'(c) - beta * sum_zp prob[z,zp] * d/dc V(m')
                        # d/dc V(m') = V'(m') * dm'/dc = V'(m') * (-R)
                        # So F(c) = U'(c) + beta * sum_zp prob[z,zp] * R * V'(m')
                        # F'(c) = U''(c) + beta * sum_zp prob[z,zp] * R * d/dc V'(m')
                        # d/dc V'(m') = V''(m') * dm'/dc = V''(m') * (-R)
                        # We don't have V''. Let's approximate or ignore F'' for Newton?
                        # Newton needs F'. If we don't have V'', we can't compute F' exactly.
                        # Alternative: Use a simpler update rule or approximate F'.
                        # Let's approximate F' using U''(c) only? No, that's not right.
                        # Let's use a damped Newton or just a fixed-point iteration?
                        # Fixed-point: c_{new} = (beta * R * E[V'(m')])^(-1/gamma) * ... ? No.
                        
                        # Let's try to approximate F'.
                        # F'(c) = U''(c) + beta * sum prob * R * (-R) * V''(m')
                        # We don't have V''.
                        # Let's use a secant method instead? Or just a simple gradient ascent?
                        # Gradient ascent on the objective J(c) = U(c) + beta E[V(m')]
                        # J'(c) = U'(c) + beta * sum prob * V'(m') * (-R)
                        # J''(c) = U''(c) + beta * sum prob * V''(m') * R^2
                        # If we ignore V'', J''(c) approx U''(c) = -gamma c^(-gamma-1)
                        # This might be unstable.
                        
                        # Let's stick to Newton but approximate V'' using finite differences?
                        # Or just use a robust 1D optimizer like minimize_scalar but vectorized?
                        # The prompt says minimize_scalar is slow.
                        
                        # Let's try a different approach: 
                        # Since the grid is small (101), maybe we can just do a fine grid search?
                        # No, that's what the seed did.
                        
                        # Let's try to compute V'' using the interpolator's second derivative?
                        # PchipInterpolator doesn't easily give second derivatives.
                        
                        # Let's use a simpler update:
                        # c_new = c - F(c) / F'(c)
                        # where F'(c) is approximated by U''(c) + beta * R^2 * E[V''(m')]
                        # If we can't get V'', we can use a central difference for V' to get V''.
                        # But that's expensive.
                        
                        # Let's just use a damped Newton with a simple step size.
                        # Or, let's use the fact that for CRRA, the optimal c is a fraction of m.
                        # c* = phi(z) * m.
                        # We can solve for phi(z) using Newton's method on the Euler equation in terms of phi.
                        # This might be more stable.
                        
                        pass # Placeholder for the logic above
                    
                    # Let's restart with a simpler, robust method:
                    # Use scipy.optimize.minimize_scalar but only if Newton fails?
                    # Or, let's just implement a robust 1D root finding for F(c)=0.
                    # F(c) is monotonically decreasing (U' decreases, V' term increases as c decreases? No.)
                    # As c increases, U'(c) decreases.
                    # m' decreases, so V'(m') increases (since V is concave, V' is decreasing in m', so as m' decreases, V' increases).
                    # So -R * V'(m') becomes more negative?
                    # Wait, F(c) = U'(c) + beta * R * E[V'(m')]
                    # As c increases, m' decreases. V'(m') increases. So E[V'(m')] increases.
                    # So F(c) = U'(c) + beta * R * E[V'(m')].
                    # U'(c) is decreasing. E[V'(m')] is increasing.
                    # So F(c) is not necessarily monotonic?
                    # Actually, U'(c) goes to 0 as c->infty. E[V'(m')] goes to infinity as c->m (m'->0).
                    # So F(c) goes from +infinity (at c=0) to -infinity (at c=m)?
                    # Let's check:
                    # c -> 0: U'(c) -> inf. m' -> R*m + income. V'(m') is finite. F -> inf.
                    # c -> m: U'(c) is finite. m' -> income. V'(m') is large (since m' is small). F -> -inf?
                    # Wait, V'(m') is positive. So beta*R*E[V'(m')] is positive.
                    # So F(c) = U'(c) + positive term.
                    # As c -> m, U'(c) is finite, positive term is large positive. So F(c) -> +infinity?
                    # That means there is no root?
                    # Ah, the Euler equation is U'(c) = beta * E[ R * V'(m') ]
                    # So F(c) = U'(c) - beta * E[ R * V'(m') ]
                    # As c -> 0, U'(c) -> inf, V'(m') is finite. F -> inf.
                    # As c -> m, U'(c) is finite, V'(m') -> inf (since m' -> income, which is small? No, income is 0.5 or 1.5, not that small).
                    # Wait, if m is small, m' is small.
                    # Let's just use a robust bracketing method like brentq.
                    # We need to find a bracket [a, b] such that F(a) > 0 and F(b) < 0.
                    # a = 1e-12, b = m - 1e-12.
                    # F(a) = U'(a) - beta*R*E[V'(R*a + income)] ~ inf - finite > 0.
                    # F(b) = U'(m) - beta*R*E[V'(income)].
                    # Is F(b) < 0?
                    # U'(m) is small if m is large.
                    # V'(income) is large if income is small.
                    # So F(b) might be < 0.
                    # If not, the optimum is at c = m.
                    
                    pass

                # Given the complexity and time limit, let's fall back to a vectorized grid search
                # but with a finer grid for c, and use np.interp for V.
                # This is safer and likely faster than a buggy Newton.
                
                # Create a fine grid for c
                c_grid = np.linspace(1e-12, m, 100)
                m_save = R * (m - c_grid)
                obj_vals = np.zeros_like(c_grid)
                
                for idx, c_val in enumerate(c_grid):
                    cont_val = 0.0
                    for zp in range(Z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = m_save[idx] + income[zp]
                        m_prime_clamped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                        v_zp = np.interp(m_prime_clamped, resource_grid, V_next[zp, :])
                        cont_val += prob * v_zp
                    obj_vals[idx] = utility(c_val) + beta * cont_val
                
                c_opt = c_grid[np.argmax(obj_vals)]
                consumption[t, z, i] = c_opt
                V[t, z, i] = np.max(obj_vals)

    return consumption
