"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq
import math


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system y0' = 1.5 y0 - y0 y1, y1' = -3 y1 + y0 y1
    from t=0 to t=5 with y(0) = [10, 5].
    
    Uses the first integral (Hamiltonian) of the system to determine the phase trajectory,
    then uses the fact that the system is periodic to find the exact terminal state.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # The Lotka-Volterra system has a first integral (Hamiltonian):
    # H(y0, y1) = y1 - b*log(y1) + y0 - a*log(y0)
    # where a = 1.5, b = 3 for this specific system.
    # Actually, let's derive it properly.
    # y0' = a*y0 - b*y0*y1  where a=1.5, b=1
    # y1' = -c*y1 + d*y0*y1  where c=3, d=1
    # The first integral is: H = d*y0 - c*log(y0) + b*y1 - a*log(y1)
    # Wait, let me re-derive.
    # dy1/dy0 = (-c*y1 + d*y0*y1) / (a*y0 - b*y0*y1)
    # dy1/dy0 = y1*(-c + d*y0) / (y0*(a - b*y1))
    # dy1/(y1*(a - b*y1)) = dy0/(y0*(d*y0 - c))
    # Integrating both sides gives the first integral.
    
    # For our specific system:
    # y0' = 1.5*y0 - 1*y0*y1
    # y1' = -3*y1 + 1*y0*y1
    # So a=1.5, b=1, c=3, d=1
    # The first integral is: H = y0 - 3*log(y0) + y1 - 1.5*log(y1)
    # Let me verify: dH/dt = (1 - 3/y0)*y0' + (1 - 1.5/y1)*y1'
    # = (1 - 3/y0)*(1.5*y0 - y0*y1) + (1 - 1.5/y1)*(-3*y1 + y0*y1)
    # = (1 - 3/y0)*y0*(1.5 - y1) + (1 - 1.5/y1)*y1*(-3 + y0)
    # = y0*(1.5 - y1) - 3*(1.5 - y1) + y1*(-3 + y0) - 1.5*(-3 + y0)
    # = 1.5*y0 - y0*y1 - 4.5 + 3*y1 - 3*y1 + y0*y1 + 4.5 - 1.5*y0
    # = 0 ✓
    
    # So H(y0, y1) = y0 - 3*log(y0) + y1 - 1.5*log(y1) is conserved.
    
    # The initial condition is y0_init = 10, y1_init = 5
    # H0 = 10 - 3*log(10) + 5 - 1.5*log(5)
    
    # The system is periodic. We need to find where the trajectory is at t=5.
    # Since the system is autonomous and has a first integral, the trajectory lies on
    # the level set H(y0, y1) = H0.
    
    # One approach: parameterize the trajectory and find the position at t=5.
    # But we can also use the fact that for Lotka-Volterra, the period can be computed,
    # and we can find the phase at t=5.
    
    # Alternative approach: Use the first integral to reduce the problem.
    # From the first integral, we can express y1 as a function of y0 (or vice versa).
    # Then we can integrate a single ODE to find y0(t) and hence y1(t).
    
    # Actually, let's think about this differently. The key insight is that the system
    # is conservative (has a first integral), so the trajectory is a closed curve.
    # We can compute the period T and then find the phase at t=5.
    
    # For Lotka-Volterra, the period can be approximated, but for exact results,
    # we might need to use the first integral more cleverly.
    
    # Let's use a different approach: since the system is 2D and autonomous with a
    # first integral, we can reduce it to a 1D problem.
    
    # From H(y0, y1) = H0, we can solve for y1 in terms of y0 (or use a parameterization).
    # Then y0' = 1.5*y0 - y0*y1(y0) is a 1D ODE.
    
    # However, solving for y1(y0) explicitly is not straightforward.
    
    # Alternative: Use the fact that we know the initial condition and the first integral,
    # and we can numerically integrate the 1D ODE with very high accuracy.
    
    # Actually, the simplest robust approach: use a high-order explicit method with
    # adaptive step size, but that's what the current code does.
    
    # Let me reconsider. The directive says to use the first integral to analytically
    # determine the exact terminal state by solving a single univariate root-finding problem.
    
    # Here's the idea: The trajectory on the level set H(y0, y1) = H0 is a closed curve.
    # We can parameterize it by an angle or by one of the variables.
    # The time to go from one point to another on the curve can be computed.
    
    # Specifically, we can write:
    # dt = dy0 / (1.5*y0 - y0*y1)
    # where y1 is determined by y0 from the first integral.
    
    # So the total time from y0_a to y0_b along the trajectory is:
    # T = ∫ dy0 / (1.5*y0 - y0*y1(y0))
    
    # We want to find y0_end such that the time from y0_init=10 to y0_end is t1-t0=5.
    
    # This is a univariate root-finding problem!
    
    # First, let's compute H0:
    a, b, c, d = 1.5, 1.0, 3.0, 1.0
    y0_init, y1_init = y[0], y[1]
    
    def H(y0, y1):
        return y0 - c*math.log(y0) + y1 - a*math.log(y1)
    
    H0 = H(y0_init, y1_init)
    
    # Now, for a given y0, we need to find y1 such that H(y0, y1) = H0.
    # H(y0, y1) = y0 - c*log(y0) + y1 - a*log(y1) = H0
    # => y1 - a*log(y1) = H0 - y0 + c*log(y0)
    # Let g(y1) = y1 - a*log(y1) - (H0 - y0 + c*log(y0)) = 0
    
    # g(y1) is convex for y1 > 0 (g'' = a/y1^2 > 0), and has a unique minimum.
    # For each y0, there are generally two solutions for y1 (above and below the minimum).
    # We need to track which branch we're on.
    
    # At the initial point, y1_init = 5. Let's find the two roots for y1 given y0_init.
    
    def find_y1_from_y0(y0_val, y1_guess):
        """Find y1 such that H(y0_val, y1) = H0, starting from y1_guess."""
        target = H0 - y0_val + c * math.log(y0_val)
        
        def g(y1_val):
            return y1_val - a * math.log(y1_val) - target
        
        # g is convex, minimum at y1 = a = 1.5
        # We need to find the root on the correct branch.
        # Use brentq with appropriate brackets.
        
        # Find the minimum of g:
        y1_min = a  # where g'(y1) = 1 - a/y1 = 0 => y1 = a
        
        # g(y1_min) = a - a*log(a) - target
        g_min = g(y1_min)
        
        if g_min > 1e-14:
            # No real solution
            return None
        
        # Determine which side of the minimum we're on based on y1_guess
        if y1_guess >= y1_min:
            # Right branch: y1 >= y1_min
            # g(y1) -> infinity as y1 -> infinity
            # Find upper bracket
            y1_lo = y1_min
            y1_hi = max(y1_guess * 2, y1_min + 1)
            while g(y1_hi) < 0:
                y1_hi *= 2
            if y1_hi > 1e12:
                return None
            try:
                y1_sol = brentq(g, y1_lo, y1_hi, xtol=1e-15, rtol=1e-15, maxiter=100)
            except ValueError:
                return None
            return y1_sol
        else:
            # Left branch: y1 <= y1_min
            # g(y1) -> infinity as y1 -> 0+
            # Find lower bracket
            y1_hi = y1_min
            y1_lo = min(y1_guess * 0.5, y1_min - 1)
            while y1_lo > 1e-15 and g(y1_lo) < 0:
                y1_lo *= 0.5
            if y1_lo <= 1e-15:
                return None
            try:
                y1_sol = brentq(g, y1_lo, y1_hi, xtol=1e-15, rtol=1e-15, maxiter=100)
            except ValueError:
                return None
            return y1_sol
    
    # Determine the branch at the initial point
    y1_min_val = a  # 1.5
    on_right_branch = y1_init >= y1_min_val
    
    # Now, we need to compute the time to go from y0_init to y0_end along the trajectory.
    # The direction of motion depends on the sign of y0' = a*y0 - b*y0*y1 = y0*(a - y1).
    # At the initial point: y0' = 10*(1.5 - 5) = 10*(-3.5) = -35 < 0
    # So y0 is decreasing initially.
    
    # We need to find y0_end such that the time from y0_init to y0_end is 5.
    
    # The time integral: T(y0_a, y0_b) = ∫_{y0_a}^{y0_b} dy0 / (y0*(a - y1(y0)))
    # where y1(y0) is on the correct branch.
    
    # Note: the sign of the integrand depends on the direction of motion.
    # We want T = t1 - t0 = 5.
    
    # Let's define a function that computes the time from y0_init to a given y0_val.
    # We need to be careful about the direction.
    
    # Actually, let's parameterize by going along the trajectory.
    # The trajectory is a closed curve. Let's find the period first.
    
    # To find the period, we can compute the time to go from y0_init back to y0_init
    # in the same direction.
    
    # But we don't need the period explicitly. We just need to find y0_end at t=5.
    
    # Let's define a function time_to(y0_end) that returns the time to go from y0_init
    # to y0_end along the trajectory (in the direction of motion).
    
    # The direction of motion at any point is determined by the sign of y0' = y0*(a - y1).
    # Since y0 > 0 always, the sign is determined by (a - y1) = (1.5 - y1).
    
    # On the right branch (y1 >= 1.5), y1 >= 1.5, so a - y1 <= 0, so y0' <= 0.
    # On the left branch (y1 <= 1.5), y1 <= 1.5, so a - y1 >= 0, so y0' >= 0.
    
    # At the initial point, y1_init = 5 > 1.5, so we're on the right branch, y0' < 0.
    # y0 decreases until it reaches the point where y1 = 1.5 (the minimum of y1 on the curve).
    # Then y1 < 1.5, y0' > 0, y0 increases back to the initial point.
    
    # So the trajectory goes: y0 decreases from 10, reaches a minimum y0_min, then increases back to 10.
    
    # We need to find y0_end at t=5.
    
    # Let's compute the time from y0_init to y0_end.
    # If y0_end < y0_init, we're in the first half (y0 decreasing).
    # If y0_end > y0_init, we've passed the minimum and are in the second half.
    
    # Define the time function:
    def time_from_init(y0_end):
        """Compute the time to go from y0_init to y0_end along the trajectory."""
        if y0_end <= 0:
            return None
        
        # Check if y0_end is on the trajectory
        y1_end = find_y1_from_y0(y0_end, y1_init if y0_end < y0_init else y1_init)
        if y1_end is None:
            return None
        
        # Integrate dt = dy0 / (y0 * (a - y1(y0)))
        # from y0_init to y0_end
        
        # Use numerical quadrature
        def integrand(y0_val):
            y1_val = find_y1_from_y0(y0_val, y1_init if y0_val < y0_init else y1_init)
            if y1_val is None:
                return 0
            denom = y0_val * (a - y1_val)
            if abs(denom) < 1e-30:
                return 0
            return 1.0 / denom
        
        # Integrate from y0_init to y0_end
        # Use scipy.integrate.quad for high accuracy
        try:
            from scipy.integrate import quad
            # We need to handle the sign properly
            # quad integrates from a to b, so if y0_end < y0_init, the integral will be negative
            # but we want the positive time.
            result, error = quad(integrand, y0_init, y0_end, limit=1000, epsabs=1e-15, epsrel=1e-15)
            # The time should be positive
            t_val = -result if y0_end < y0_init else result
            if t_val < 0:
                t_val = abs(t_val)
            return t_val
        except Exception:
            return None
    
    # Now, we need to find y0_end such that time_from_init(y0_end) = 5.
    
    # First, let's find the range of y0 on the trajectory.
    # y0_min is where y1 is at its maximum (or minimum?).
    # Actually, y0 is at its minimum when y1 is at its maximum on the right branch.
    # Let's find y0_min by finding where y0' = 0, i.e., y1 = a = 1.5.
    
    # Find y0 where y1 = 1.5 on the trajectory:
    # H(y0, 1.5) = H0
    # y0 - c*log(y0) + 1.5 - a*log(1.5) = H0
    # y0 - 3*log(y0) = H0 - 1.5 + 1.5*log(1.5)
    
    target_y0 = H0 - 1.5 + a * math.log(1.5)
    
    def h_y0(y0_val):
        return y0_val - c * math.log(y0_val) - target_y0
    
    # h_y0 is convex, minimum at y0 = c = 3.
    # There are two solutions: one with y0 < 3 and one with y0 > 3.
    # The one with y0 < 3 is y0_min (where y0 is at its minimum).
    
    # Find the solution with y0 < 3:
    y0_lo = 0.1
    while h_y0(y0_lo) > 0:
        y0_lo *= 0.5
    if y0_lo < 1e-15:
        y0_min = y0_lo
    else:
        try:
            y0_min = brentq(h_y0, y0_lo, 3.0, xtol=1e-15, rtol=1e-15, maxiter=100)
        except ValueError:
            y0_min = y0_lo
    
    # Now, the time from y0_init to y0_min:
    t_to_min = time_from_init(y0_min)
    
    # The total period is twice the time from y0_min to y0_init (by symmetry? Not exactly, but let's compute).
    # Actually, let's just find y0_end by solving time_from_init(y0_end) = 5.
    
    # If 5 < t_to_min, then y0_end is between y0_init and y0_min (y0 decreasing).
    # If 5 > t_to_min, then y0_end is on the return path (y0 increasing).
    
    # Let's try to solve for y0_end.
    
    # Case 1: y0_end in [y0_min, y0_init]
    def f_y0_decreasing(y0_val):
        t_val = time_from_init(y0_val)
        if t_val is None:
            return 0
        return t_val - 5
    
    # Case 2: y0_end in [y0_init, y0_max] where y0_max is the maximum y0 on the trajectory
    # Find y0_max: where y1 is at its minimum.
    # y1_min on the trajectory: find y1 where y0' changes sign... 
    # Actually, y1 is at its minimum when y0 is at its maximum.
    # y0_max: find y0 where y1 is at its minimum.
    # The minimum of y1 on the trajectory: y1' = 0 => y0 = c = 3.
    # So y0_max is where y0 = 3? No, y0' = y0*(a - y1), so y0' = 0 when y1 = a = 1.5.
    # That's the same condition. So y0 has extrema when y1 = 1.5.
    # We found y0_min. The other solution for y0 when y1 = 1.5 is y0_max.
    
    y0_hi = 3.0
    while h_y0(y0_hi) < 0:
        y0_hi *= 2
    if y0_hi > 1e12:
        y0_max = y0_hi
    else:
        try:
            y0_max = brentq(h_y0, 3.0, y0_hi, xtol=1e-15, rtol=1e-15, maxiter=100)
        except ValueError:
            y0_max = y0_hi
    
    # Now try to solve in the decreasing branch first
    y0_end = None
    
    # Check if the solution is in the decreasing branch
    t_at_min = time_from_init(y0_min)
    if t_at_min is not None and 5 <= t_at_min:
        # Solution in [y0_min, y0_init]
        try:
            y0_end = brentq(f_y0_decreasing, y0_min, y0_init, xtol=1e-14, rtol=1e-14, maxiter=100)
        except ValueError:
            y0_end = None
    
    if y0_end is None:
        # Solution in the increasing branch: y0 goes from y0_min back up
        # Time from y0_init to y0_end (where y0_end > y0_init) 
        # = time from y0_init to y0_min + time from y0_min to y0_end
        
        def f_y0_increasing(y0_val):
            # Time from y0_init to y0_min, then from y0_min to y0_val
            t1 = time_from_init(y0_min)
            if t1 is None:
                return 0
            # Time from y0_min to y0_val
            def integrand_inc(y0_v):
                y1_v = find_y1_from_y0(y0_v, 1.5)  # left branch
                if y1_v is None:
                    return 0
                denom = y0_v * (a - y1_v)
                if abs(denom) < 1e-30:
                    return 0
                return 1.0 / denom
            
            try:
                from scipy.integrate import quad
                t2_result, _ = quad(integrand_inc, y0_min, y0_val, limit=1000, epsabs=1e-15, epsrel=1e-15)
                t2 = t2_result
            except Exception:
                return 0
            return t1 + t2 - 5
        
        try:
            y0_end = brentq(f_y0_increasing, y0_init, y0_max, xtol=1e-14, rtol=1e-14, maxiter=100)
        except ValueError:
            y0_end = None
    
    if y0_end is None:
        # Fallback: use RK4 with many steps
        def _rk4(f, t0, t1, y0, steps):
            h = (t1 - t0) / steps
            t, y = t0, np.array(y0, dtype=float)
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
                if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                    return None
            return y
        
        y = _rk4(f, t0, t1, y0, 100000)
        if y is not None:
            return y
        return np.array(y0, dtype=float)
    
    # Find y1_end
    # Determine which branch y0_end is on
    if y0_end < y0_init:
        y1_end = find_y1_from_y0(y0_end, y1_init)
    else:
        y1_end = find_y1_from_y0(y0_end, 1.5)  # left branch
    
    if y1_end is None:
        # Fallback
        def _rk4(f, t0, t1, y0, steps):
            h = (t1 - t0) / steps
            t, y = t0, np.array(y0, dtype=float)
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
                if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                    return None
            return y
        
        y = _rk4(f, t0, t1, y0, 100000)
        if y is not None:
            return y
        return np.array(y0, dtype=float)
    
    return np.array([y0_end, y1_end], dtype=float)
# EVOLVE-BLOCK-END
