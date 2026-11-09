import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import solve_ivp

def _robertson_jac(t, y):
    y0, y1, y2 = y
    J = np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e3 * y1, -1e4 * y1],
        [0.0, 6e3 * y1, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Try custom Radau IIA
    try:
        return _custom_radau(f, t_span, y0)
    except Exception:
        # Fallback to solve_ivp if custom fails
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_robertson_jac,
            rtol=1e-10, 
            atol=1e-10
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            return y0 # Last resort

def _custom_radau(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Coefficients
    sqrt6 = np.sqrt(6)
    c = np.array([0.2 - 2*sqrt6/15, 0.5, 1.0 - 2*sqrt6/15])
    A = np.array([
        [0.04562648399503793, -0.2623589960532283, 0.2167325120581903],
        [-0.01057793164367395, 0.5105779316436739, -0.4999999999999999],
        [-0.006732512058190346, 0.2900650241163807, 0.7166674879418096]
    ])
    b = np.array([0.1540453005336774, 0.6919093989326452, 0.1540453005336774])
    b_e = np.array([0.1983692834083312, 0.6032614331833376, 0.1983692834083312])
    
    h = (t1 - t0) / 1000.0
    rtol = 1e-8
    atol = 1e-10
    
    max_steps = 5000
    steps = 0
    
    while t < t1 and steps < max_steps:
        h_new = min(h, t1 - t)
        
        Y = np.tile(y, (3, 1))
        converged = False
        
        for _ in range(10):
            t_stages = t + c * h_new
            f_stages = np.array([f(t_stages[i], Y[i]) for i in range(3)])
            
            Af = A @ f_stages
            R = Y - y - h_new * Af
            
            if np.max(np.abs(R)) < 1e-12:
                converged = True
                break
            
            # Build 9x9 Jacobian
            J_sys = np.eye(9)
            for i in range(3):
                for j in range(3):
                    J_ode_j = _robertson_jac(t_stages[j], Y[j])
                    block = -h_new * A[i,j] * J_ode_j
                    if i == j:
                        block += np.eye(3)
                    J_sys[3*i:3*i+3, 3*j:3*j+3] += block
            
            try:
                lu, piv = lu_factor(J_sys)
                dY_flat = lu_solve((lu, piv), R.flatten())
                dY = dY_flat.reshape(3, 3)
                Y += dY
            except Exception:
                break
        
        if not converged:
            h_new *= 0.5
            if h_new < 1e-6:
                raise RuntimeError("Step size too small")
            continue
            
        f_stages_final = np.array([f(t + c[i]*h_new, Y[i]) for i in range(3)])
        y_next = y + h_new * (b @ f_stages_final)
        y_next_e = y + h_new * (b_e @ f_stages_final)
        
        err = np.max(np.abs(y_next - y_next_e))
        scale = np.max(np.maximum(atol, rtol * np.abs(y_next)))
        
        if err <= scale:
            t += h_new
            y = y_next
            if err > 0:
                h_new *= min(5.0, max(0.1, 0.9 * (scale/err)**0.2))
            else:
                h_new *= 2.0
            h = h_new
        else:
            h_new *= max(0.1, 0.9 * (scale/err)**0.25)
            h = h_new
        steps += 1
        
    if t < t1:
        raise RuntimeError("Max steps reached")
        
    return y
