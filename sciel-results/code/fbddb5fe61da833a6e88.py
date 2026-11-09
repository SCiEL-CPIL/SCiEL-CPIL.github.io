    import numpy as np
    from scipy.integrate import ode

    def _robertson_jac(t, y):
        y1, y2, y3 = y
        return np.array([
            [-10.0, 1e4 * y3, 1e4 * y2],
            [10.0, -1e4 * y3 - 2e6 * y2, -1e4 * y2],
            [0.0, 2e6 * y2, 0.0]
        ])

    def integrate(f, t_span, y0):
        t0, t1 = t_span
        y0 = np.array(y0, dtype=float)
        
        # Use lsoda which automatically switches between stiff and non-stiff
        solver = ode(f, jac=_robertson_jac)
        solver.set_integrator('lsoda')
        solver.set_initial_value(y0, t0)
        solver.set_relative_tolerance(1e-12)
        solver.set_absolute_tolerance(1e-14)
        
        # Integrate to t1
        solver.integrate(t1)
        
        if solver.successful():
            return solver.y
        else:
            # Fallback to solve_ivp BDF if lsoda fails
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='BDF', jac=_robertson_jac, rtol=1e-12, atol=1e-14)
            if sol.success:
                return sol.y[:, -1]
            raise RuntimeError("Integration failed")
    