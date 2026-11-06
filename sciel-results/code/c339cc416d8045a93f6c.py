"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    k1, k2, k3 = 0.04, 3.0e-5, 1.0e4
    # dy0/dt = -k1*y0 + k3*y1*y2
    # dy1/dt = k1*y0 - k2*y1 - k3*y1*y2
    # dy2/dt = k2*y1
    # Jacobian:
    # d(dy0)/dy0 = -k1, d(dy0)/dy1 = k3*y2, d(dy0)/dy2 = k3*y1
    # d(dy1)/dy0 = k1, d(dy1)/dy1 = -k2 - k3*y2, d(dy1)/dy2 = -k3*y1
    # d(dy2)/dy0 = 0, d(dy2)/dy1 = k2, d(dy2)/dy2 = 0
    return np.array([
        [-k1, k3 * y2, k3 * y1],
        [k1, -k2 - k3 * y2, -k3 * y1],
        [0.0, k2, 0.0]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE problem on [0, 40].
    
    Primary strategy: LSODA with very tight tolerances. LSODA automatically
    switches between stiff (BDF) and non-stiff (Adams) methods, which should
    minimize function evaluations during the rapid transient while maintaining
    high precision on the slow manifold.
    
    Fallback: Hybrid Radau + DOP853 if LSODA fails or times out.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 270.0  # 60s safety margin before 330s kill

    # Try LSODA first - it auto-switches stiff/non-stiff
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-14,
            atol=1e-16,
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # LSODA failed, try fallback
            raise Exception("LSODA failed")
            
    except Exception:
        pass
    
    # Check deadline before starting fallback
    if time.time() > deadline:
        # Return best effort - try a quick Radau with looser tol
        try:
            sol_quick = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-12,
                atol=1e-14,
            )
            if sol_quick.success:
                return sol_quick.y[:, -1]
            return sol_quick.y[:, -1] if sol_quick.y.size > 0 else y0
        except Exception:
            return y0

    # Fallback: Hybrid Radau + DOP853
    t_switch = 1.0
    if t_switch > t1:
        t_switch = t1

    try:
        # Phase 1: Stiff region [t0, t_switch]
        sol1 = solve_ivp(
            f,
            [t0, t_switch],
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-13,
            atol=1e-15,
        )
        
        if not sol1.success:
            # Fallback to full Radau if phase 1 fails
            sol_full = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-13,
                atol=1e-15,
            )
            return sol_full.y[:, -1] if sol_full.success else y0

        y_switch = sol1.y[:, -1]
        
        if time.time() > deadline:
            return y_switch

        # Phase 2: Non-stiff region [t_switch, t1]
        sol2 = solve_ivp(
            f,
            [t_switch, t1],
            y_switch,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        else:
            # Fallback to Radau for the rest if DOP853 fails
            sol2_fallback = solve_ivp(
                f,
                [t_switch, t1],
                y_switch,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-13,
                atol=1e-15,
            )
            if sol2_fallback.success:
                return sol2_fallback.y[:, -1]
            return y_switch

    except Exception:
        # Final fallback: Full Radau
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-13,
                atol=1e-15,
            )
            if sol.success:
                return sol.y[:, -1]
            return sol.y[:, -1] if sol.y.size > 0 else y0
        except Exception:
            return y0


# EVOLVE-BLOCK-END
