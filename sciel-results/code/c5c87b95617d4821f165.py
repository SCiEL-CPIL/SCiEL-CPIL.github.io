"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_Om_a(a, w0, wa, Om0=0.3):
    """Compute Omega_m(a) for a flat CPL universe."""
    u = 1.0 + w0 + wa * (1.0 - a)
    if u <= 0.05:
        return None
    denom = Om0 * a**3 + (1.0 - Om0) / u
    if denom <= 0:
        return None
    Om_a = Om0 * a**3 / denom
    if Om_a <= 0 or Om_a >= 1:
        return None
    return Om_a

def _target_gamma_LW(Om_a, u):
    """Linder-Waugh target gamma."""
    return (5.0/7.0) * (Om_a / u) ** 0.38

def _gamma_from_coeffs(c, q, u, wa, a):
    """Evaluate gamma from coefficients."""
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _fit_coefficients():
    """Fit the six growth-index coefficients via least squares to the Linder-Waugh formula."""
    t_start = time.time()
    deadline = 65.0  # Safety margin before 90s limit
    
    # Grid of CPL parameters and scale factors
    w0_vals = np.linspace(-1.2, -0.8, 30)
    wa_vals = np.linspace(-0.5, 0.5, 30)
    a_vals = np.linspace(0.2, 1.0, 60)
    
    Omega_m0 = 0.3
    
    # Build the design matrix and target vector
    rows = []
    targets = []
    
    for w0 in w0_vals:
        if time.time() - t_start > deadline:
            break
        for wa in wa_vals:
            if time.time() - t_start > deadline:
                break
            for a in a_vals:
                Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                if Om_a is None:
                    continue
                
                u = 1.0 + w0 + wa * (1.0 - a)
                q = 1.0 - Om_a
                target_gamma = _target_gamma_LW(Om_a, u)
                
                # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                row = np.array([1.0, q, q**2, u, wa * a, u * q])
                rows.append(row)
                targets.append(target_gamma)
                
                if time.time() - t_start > deadline:
                    break
    
    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(rows)
    b = np.array(targets)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # Clip to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Verify gamma stays in [0.1, 1.2] on the training grid
    for w0 in w0_vals:
        if time.time() - t_start > deadline:
            break
        for wa in wa_vals:
            if time.time() - t_start > deadline:
                break
            for a in a_vals:
                Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                if Om_a is None:
                    continue
                u = 1.0 + w0 + wa * (1.0 - a)
                q = 1.0 - Om_a
                gamma = _gamma_from_coeffs(coeffs, q, u, wa, a)
                if gamma < 0.1 or gamma > 1.2:
                    return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return coeffs.tolist()

def construct():
    """Construct the six growth-index coefficients."""
    t_start = time.time()
    
    try:
        coeffs = _fit_coefficients()
    except Exception:
        coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Ensure finite and in range
    for i in range(6):
        if not np.isfinite(coeffs[i]) or coeffs[i] < -10 or coeffs[i] > 10:
            coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            break
    
    return coeffs
# EVOLVE-BLOCK-END
