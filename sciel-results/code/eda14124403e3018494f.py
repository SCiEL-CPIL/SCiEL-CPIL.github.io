"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Derive six coefficients for the growth-index ansatz:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    
    We fit to the Linder approximation:
    gamma_exact = (5/7) * (Om(a) / (1 + w0 + wa*(1-a)))^0.38
    
    We perform a weighted least-squares fit over a denser grid of cosmological
    parameters to minimize the relative error in f, which is proportional
    to |delta_gamma * ln(Om(a))|.
    """
    import numpy as np
    
    # Denser grid of cosmological parameters
    Om0_vals = np.linspace(0.2, 0.35, 20)
    w0_vals = np.linspace(-1.0, 0.0, 15)
    wa_vals = np.linspace(-0.5, 0.5, 15)
    
    # Denser scale factor grid
    a_vals = np.linspace(0.1, 1.0, 100)
    
    # Build design matrix and target
    A_rows = []
    y_vals = []
    weights = []
    
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    # Compute Om(a) for flat CPL universe
                    # E^2(a) = Om0 * a^3 + (1 - Om0) * a^2 * (1 + wa)^(2*wa) * exp(3*wa*(1-a))
                    if abs(wa) < 1e-10:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
                    
                    if E2 <= 0:
                        continue
                    
                    Om_a = Om0 * a**3 / E2
                    if Om_a <= 0 or Om_a >= 1:
                        continue
                    
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a)
                    
                    if u <= 0:
                        continue
                    
                    # Target gamma from Linder formula
                    gamma_target = (5.0 / 7.0) * (Om_a / u)**0.38
                    
                    if gamma_target < 0.1 or gamma_target > 1.2:
                        continue
                    
                    # Design vector: [1, q, q^2, u, wa*a, u*q]
                    row = np.array([1.0, q, q**2, u, wa * a, u * q])
                    A_rows.append(row)
                    y_vals.append(gamma_target)
                    
                    # Weight by sensitivity of f to gamma: |ln(Om_a)|
                    # This prioritizes minimizing relative error in f
                    w = abs(np.log(Om_a)) if Om_a > 0 else 1.0
                    weights.append(w)
    
    if len(A_rows) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(A_rows)
    y = np.array(y_vals)
    w = np.array(weights)
    
    # Solve weighted least squares: minimize sum(w_i * (A_i @ c - y_i)^2)
    # Equivalent to lstsq(sqrt(w)*A, sqrt(w)*y)
    sqrt_w = np.sqrt(w)
    A_weighted = A * sqrt_w[:, np.newaxis]
    y_weighted = y * sqrt_w
    
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
    
    # Clip coefficients to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Verify gamma stays in [0.1, 1.2] on a dense validation grid
    c0, c1, c2, c3, c4, c5 = coeffs
    
    valid = True
    # Use a separate, dense validation grid to ensure robustness
    Om0_val = np.linspace(0.2, 0.35, 25)
    w0_val = np.linspace(-1.0, 0.0, 20)
    wa_val = np.linspace(-0.5, 0.5, 20)
    a_val = np.linspace(0.1, 1.0, 150)
    
    for Om0 in Om0_val:
        for w0 in w0_val:
            for wa in wa_val:
                for a in a_val:
                    if abs(wa) < 1e-10:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
                    
                    if E2 <= 0:
                        continue
                    
                    Om_a = Om0 * a**3 / E2
                    if Om_a <= 0 or Om_a >= 1:
                        continue
                    
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a)
                    
                    if u <= 0:
                        continue
                    
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                    
                    if gamma < 0.1 or gamma > 1.2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use constant 5/7
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5)]
# EVOLVE-BLOCK-END
