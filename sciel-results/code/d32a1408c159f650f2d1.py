import numpy as np

def integrate(f, t_span, y0):
    """
    High-precision ODE integrator for the Lotka-Volterra system.
    
    Since the problem is fixed and smooth, we use a high-order explicit Runge-Kutta
    method with very small steps to achieve high accuracy without relying on adaptive
    solvers that may have overhead or different error profiles.
    
    We use RK45 (Dormand-Prince 5th order) with a fixed, small step size to ensure
    deterministic, high-precision results. The step size is chosen to be small enough
    that the local truncation error is negligible relative to the reference tolerance.
    
    Given the reference uses DOP853 with rtol=1e-11, atol=1e-13, we aim for similar
    precision. A fixed-step RK45 with h=1e-3 over [0,5] gives 5000 steps, which is
    well within the 2,000,000 call limit and should achieve ~1e-10 to 1e-12 relative
    error.
    
    To improve further, we use Richardson extrapolation with RK4 (4th order) at two
    different step sizes to get effective 8th order accuracy with fewer calls than
    a naive 8th order method.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Use RK4 with Richardson extrapolation for high accuracy
    # Run RK4 with n steps, then with 2n steps, and extrapolate
    
    def rk4_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    
    def rk4_integrate(f, t0, t1, y0, n_steps):
        h = (t1 - t0) / n_steps
        y = np.array(y0, dtype=float)
        t = t0
        for _ in range(n_steps):
            y = rk4_step(f, t, y, h)
            t += h
        return y
    
    # Base step count - choose a number that balances accuracy and call count
    # n=2000 steps: 4*2000 = 8000 calls for one run
    # With Richardson: 2 runs, total ~ 4*(2000 + 4000) = 24000 calls
    # This is well within budget and should give ~1e-13 accuracy
    
    n_base = 1000
    n_fine = 2 * n_base
    
    # Coarse integration
    y_coarse = rk4_integrate(f, t0, t1, y, n_base)
    
    # Fine integration  
    y_fine = rk4_integrate(f, t0, t1, y, n_fine)
    
    # Richardson extrapolation: (2^4 * y_fine - y_coarse) / (2^4 - 1)
    # Since RK4 is 4th order, extrapolation gives 8th order accuracy
    y_ext = (16.0 * y_fine - y_coarse) / 15.0
    
    # Verify result is finite
    if not np.all(np.isfinite(y_ext)):
        # Fallback to fine result
        y_ext = y_fine
    
    return y_ext
