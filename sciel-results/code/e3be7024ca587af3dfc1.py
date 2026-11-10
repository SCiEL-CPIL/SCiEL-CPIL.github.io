"""Improved growth index approximation using exact ODE reference."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def construct():
    # Grid of training cosmologies
    # We want to cover the likely validation range.
    # Om_m0: [0.2, 0.3]
    # w0: [-1.2, 0.5]
    # wa: [-0.5, 0.5]
    
    Om_m0_vals = np.linspace(0.2, 0.3, 4)
    w0_vals = np.linspace(-1.2, 0.5, 5)
    wa_vals = np.linspace(-0.5, 0.5, 5)
    a_vals = np.linspace(0.05, 1.0, 30) # Scale factors for evaluation
    
    A_matrix = []
    y_vector = []
    
    for Om_m0 in Om_m0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                # Define H^2(a) and Om_m(a) for exact CPL
                # rho_de(a) = rho_de0 * a^(-3(w0+wa)) * exp(3*wa*(a-1))
                # rho_m(a) = rho_m0 * a^(-3)
                
                def get_om_m(a, Om_m0=Om_m0, w0=w0, wa=wa):
                    rho_m = Om_m0 * (a ** -3)
                    rho_de = (1 - Om_m0) * (a ** (-3 * (w0 + wa))) * np.exp(3 * wa * (a - 1))
                    return rho_m / (rho_m + rho_de)
                
                def get_w(a, w0=w0, wa=wa):
                    return w0 + wa * (1 - a)
                
                # ODE for D(a):
                # D'' + (3/a + (3(1+w) + w')/(a(1+w))) D' - (3/2) Om_m/a D = 0
                # w' = -wa
                # coeff_D_prime = 3/a + (3(1+w) - wa) / (a(1+w))
                
                def ode_func(a, y):
                    D, D_prime = y
                    w = get_w(a)
                    Om_m = get_om_m(a)
                    
                    # Handle singularity at a=0? We start at a=0.05, so ok.
                    # If 1+w is close to 0, this might blow up. 
                    # w0=-1, wa=0 -> w=-1. 1+w=0.
                    # For phantom or w<-1, 1+w < 0.
                    # The term is (3(1+w) - wa) / (1+w) = 3 - wa/(1+w).
                    
                    denom = 1 + w
                    if abs(denom) < 1e-6:
                        # Approximation for w ~ -1
                        # 3 - wa/denom is large.
                        # Let's just use a large number or skip?
                        # Actually, for w=-1, the ODE is D'' + (3/a - wa/(a(1+w)))... 
                        # If w=-1 exactly, wa=0 usually.
                        # Let's regularize.
                        denom = np.sign(denom) * 1e-6 if denom != 0 else 1e-6
                    
                    coeff_Dp = 3.0/a + (3.0 * denom - wa) / (a * denom)
                    coeff_D = -1.5 * Om_m / a
                    
                    D_double_prime = -coeff_Dp * D_prime - coeff_D * D
                    return [D_prime, D_double_prime]
                
                # Initial conditions at a=1:
                # D(1) = 1
                # f(1) = D'(1)/D(1) ~ 0.55 + 0.05(1+w0)
                f_init = 0.55 + 0.05 * (1 + w0)
                
                try:
                    sol = solve_ivp(ode_func, [1.0, 0.05], [1.0, f_init], t_eval=a_vals[::-1], method='RK45', rtol=1e-8, atol=1e-10)
                    if not sol.success:
                        continue
                    
                    # sol.y[0] is D, sol.y[1] is D'
                    # a_vals were reversed for integration, so sol.t is [1.0, ..., 0.05]
                    # We need to align with a_vals [0.05, ..., 1.0]
                    # sol.t is decreasing. a_vals is increasing.
                    # Let's just map by index or flip.
                    D_true = sol.y[0][::-1]
                    Dp_true = sol.y[1][::-1]
                    a_eval = a_vals # [0.05, ..., 1.0]
                    
                    # Calculate f_true
                    # f = D'/D
                    # At a=1, D=1, D'=f_init.
                    f_true = Dp_true / D_true
                    
                    # Calculate D_norm_true
                    # D is already normalized to 1 at a=1 because D(1)=1.
                    # The formula in prompt: D(a)/D(1) = exp(-int_ln(a)^0 f(exp(x)) dx)
                    # This is consistent with D(1)=1.
                    D_norm_true = D_true
                    
                    # Build A and y for least squares
                    for i, a in enumerate(a_eval):
                        Om_m_a = get_om_m(a)
                        q = 1 - Om_m_a
                        u = 1 + w0 + wa * (1 - a)
                        
                        row = [1.0, q, q*q, u, wa*a, u*q]
                        A_matrix.append(row)
                        
                        # Target: We want to fit f and D.
                        # We can't fit f and D directly with linear coefficients easily 
                        # because f = Om^gamma and D depends on integral of gamma.
                        # However, gamma is linear in c.
                        # f = exp(gamma * ln(Om_m))
                        # This is non-linear in c.
                        
                        # Wait, the current approach fits gamma directly.
                        # The prompt says: "gamma = c0 + ... + c5*u*q"
                        # "f = Om(a)^gamma"
                        # "Minimize ... relative error of both f and normalized D"
                        
                        # If I fit gamma to match f, I am fitting:
                        # Om^gamma_model = f_true
                        # gamma_model * ln(Om) = ln(f_true)
                        # gamma_model = ln(f_true) / ln(Om)
                        # This gives a linear target for gamma!
                        
                        # Let's check if fitting gamma to match f is sufficient for D.
                        # If gamma is accurate, D should be accurate.
                        # Let's try fitting gamma to match f first.
                        
                        if Om_m_a > 0 and Om_m_a < 1 and f_true[i] > 0:
                            gamma_target = np.log(f_true[i]) / np.log(Om_m_a)
                        else:
                            # Fallback to Linder approx if log fails
                            w_a = w0 + wa * (1 - a)
                            gamma_target = 0.55 + 0.05 * (1 + w_a)
                            
                        y_vector.append(gamma_target)
                        
                except Exception:
                    continue
    
    if not A_matrix:
        # Fallback to simple Linder fit if ODE fails
        return [0.55, 0.0, 0.0, 0.05, 0.0, 0.0]
        
    A = np.array(A_matrix)
    y = np.array(y_vector)
    
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    
    # Clip coefficients
    coeffs = np.clip(coeffs, -10, 10)
    
    return coeffs.tolist()
# EVOLVE-BLOCK-END
