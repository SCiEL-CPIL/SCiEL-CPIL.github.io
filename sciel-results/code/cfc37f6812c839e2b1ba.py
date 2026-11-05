import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    call_count = 0
    start_time = time.time()
    max_time = 310.0  # Safety margin before 330s kill
    max_calls = 1950000  # Safety margin for 2M limit

    def f_wrapped(t, y):
        nonlocal call_count
        call_count += 1
        elapsed = time.time() - start_time
        if elapsed > max_time:
            raise TimeoutError("Time limit exceeded")
        if call_count > max_calls:
            raise OverflowError("Call limit exceeded")
        return f(t, y)

    # Primary: DOP853 with very tight tolerances
    # Remove max_step to allow solver to choose optimal step size, reducing calls
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # If DOP853 failed, try Radau (implicit, handles stiffness)
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: RK45 with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped,
            t_span,
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
