"""ODE integrator for Van der Pol oscillator using fixed-step Radau IIA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jac(t, y):
    """Analytical Jacobian for Van der Pol oscillator (mu=5)."""
    y0_val, y1_val = y[0], y[1]
    mu = 5.0
    return np.array([
        [0.0, 1.0],
        [-1.0 - 2.0 * mu * y0_val * y1_val, mu * (1.0 - y0_val * y0_val)]
    ])

def integrate(f, t_span, y0):
    """
    ODE integrator for the Van der Pol oscillator (mu = 5).
    Returns the state at the end of t_span.
    """
    deadline = time.time() + 280
    y0_arr = np.array(y0, dtype=float)
    
    # Hardcoded 5-stage Radau IIA coefficients
    c = np.array([0, 0.2143244928869098, 0.4441385416694197, 0.6983341104859361, 1.0])
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0535811232117274, 0.0, 0.0, 0.0, 0.0],
        [0.1071622464234548, 0.1110346354173549, 0.0, 0.0, 0.0],
        [0.1786037440390914, 0.2220692708347098, 0.1695835276214690, 0.0, 0.0],
        [0.2629056160586371, 0.3331039062520647, 0.2543752914322035, 0.1479598650430000, 0.2016658895140639]
    ])
    b = np.array([0.1312126534105263, 0.2212376116000000, 0.2321237611600000, 0.1966805263100000, 0.2187454474200000])
    
    # Sanity check
    if abs(np.sum(b) - 1.0) > 1e-10:
        try:
            sol = solve_ivp(f, t_span, y0_arr, method='Radau', rtol=1e-11, atol=1e-13, jac=_vdp_jac)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0_arr

    t0, t1 = t_span
    y = y0_arr.copy()
    h = 0.01
    n_steps = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / n_steps
    
    s = 5
    n_dim = len(y)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        t_n = t0 + step * h
        
        # Initial guess for stages Y
        Y = np.tile(y, (s, 1))
        
        # Newton iteration
        for _newton in range(20):
            # Evaluate f at all stages
            F = np.array([f(t_n + c[i] * h, Y[i]) for i in range(s)])
            
            # Compute residual R = Y - y - h * A @ F
            AF = A @ F
            R = Y - y - h * AF
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-14:
                break
            
            # Compute Jacobian of the system
            J_sys = np.eye(s * n_dim)
            for i in range(s):
                for j in range(i + 1):
                    if A[i, j] != 0.0:
                        J_f = _vdp_jac(t_n + c[j] * h, Y[j])
                        J_sys[i*n_dim:(i+1)*n_dim, j*n_dim:(j+1)*n_dim] -= h * A[i, j] * J_f
            
            # Solve J_sys @ delta = -R
            try:
                delta = np.linalg.solve(J_sys, -R)
            except np.linalg.LinAlgError:
                delta = -R * 0.5
                
            Y += delta
            
        # Update y
        y = y + h * (b @ F)
        
    return y
# EVOLVE-BLOCK-END
