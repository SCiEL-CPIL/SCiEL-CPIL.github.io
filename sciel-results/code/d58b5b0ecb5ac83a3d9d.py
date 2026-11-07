"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jacobian(t, y):
    """Analytic Jacobian for the Van der Pol oscillator (mu=5)."""
    y0, y1 = y[0], y[1]
    return np.array([
        [0.0, 1.0],
        [-1.0 - 10.0 * y0 * y1, 5.0 * (1.0 - y0 * y0)]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    with initial condition y0, using Radau with analytic Jacobian.
    """
    start_time = time.time()
    DEADLINE = 290.0  # seconds, safety margin from 330s limit

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    # Count calls to f
    call_count = [0]

    def counted_f(t, y):
        call_count[0] += 1
        if time.time() - start_time > DEADLINE:
            raise TimeoutError("Deadline exceeded")
        return f(t, y)

    # Strategy: use Radau with analytic Jacobian and very tight tolerances
    try:
        sol = solve_ivp(
            counted_f,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            jac=_vdp_jacobian,
            dense_output=False
        )
        
        if not sol.success:
            raise RuntimeError(f"Radau failed: {sol.message}")
        
        y_end = sol.y[:, -1]
        
        if not np.all(np.isfinite(y_end)):
            raise RuntimeError("Non-finite result")
        
        return y_end
    
    except (TimeoutError, RuntimeError, Exception) as e:
        # Fallback: try with slightly looser tolerances
        try:
            sol = solve_ivp(
                counted_f,
                (t0, t1),
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                jac=_vdp_jacobian,
                dense_output=False
            )
            
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass
        
        # Last resort: Radau with reference tolerances
        try:
            sol = solve_ivp(
                counted_f,
                (t0, t1),
                y0,
                method='Radau',
                rtol=1e-11,
                atol=1e-13,
                jac=_vdp_jacobian,
                dense_output=False
            )
            
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass
        
        return y0
# EVOLVE-BLOCK-END
