"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.integrate import quad

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Initial best: constant gamma
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_error = float('inf')
    
    # Helper to compute error for given coefficients
    def compute_error(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Check gamma bounds
        # Test a range of cosmologies and scale factors
        test_cosmologies = [
            (0.3, 0.0, 0.0),   # LambdaCDM
            (0.3, 0.0, -0.5),  # CPL with wa=-0.5
            (0.3, -0.5, 0.0),  # CPL with w0=-0.5
            (0.3, -1.0, 0.0),  # CPL with w0=-1.0
            (0.3, -1.0, 0.5),  # CPL with w0=-1.0, wa=0.5
            (0.2, -1.0, 0.0),  # Lower Om
            (0.4, -1.0, 0.0),  # Higher Om
        ]
        
        a_values = np.linspace(0.1, 1.0, 50)
        
        max_rel_err = 0.0
        
        for Om0, w0, wa in test_cosmologies:
            # Reference: solve the ODE for f and D
            # f = d ln D / d ln a
            # f^2 + f(3 - 3*Om(a) - 3/2*(1+3w(a))*Om_DE(a)/Om(a)) + 3*Om(a) = 0
            # where Om_DE(a) = Om_DE0 * a^(-3(1+w(a)))
            
            Om_DE0 = 1.0 - Om0
            
            def Om_of_a(a):
                Om_DE = Om_DE0 * a**(-3.0 * (1.0 + w0 + wa * (1.0 - a)))
                return Om_DE / (Om_DE + Om0 * a**3)
            
            def w_of_a(a):
                return w0 + wa * (1.0 - a)
            
            # Solve ODE for f: df/dlna = -f^2 - f*(3 - 3*Om - 3/2*(1+3w)*Om_DE/Om) + 3*Om
            # Actually the standard growth equation is:
            # f'' + (3 + 2*H'/H - 3*Om) f' + (1 - Om) f = 0 (in terms of ln a)
            # But let's use the simpler form: f = Om^gamma as approximation
            
            # For reference, use the standard approximation or solve numerically
            # Let's solve the second-order ODE for D:
            # D'' + (3 + H'/H) D' - (3/2) Om(a) D = 0 where ' is d/dlna
            
            def H_prime_over_H(a):
                # H'/H = d ln H / d ln a = (1/2) * d ln H^2 / d ln a
                # H^2/H0^2 = Om a^-3 + Om_DE a^(-3(1+w))
                # For CPL: Om_DE(a) = Om_DE0 * a^(-3(1+w0+wa(1-a)))
                # Let's compute numerically
                da = 1e-6
                a_plus = min(a + da, 1.0)
                a_minus = max(a - da, 0.01)
                
                H2_plus = Om0 * a_plus**(-3) + Om_DE0 * a_plus**(-3*(1+w0+wa*(1-a_plus)))
                H2_minus = Om0 * a_minus**(-3) + Om_DE0 * a_minus**(-3*(1+w0+wa*(1-a_minus)))
                
                H_plus = np.sqrt(H2_plus)
                H_minus = np.sqrt(H2_minus)
                
                return (H_plus - H_minus) / (a_plus - a_minus) / ((H_plus + H_minus) / 2)
            
            # Solve ODE: D'' + (3 + H'/H) D' - (3/2) Om(a) D = 0
            # with D(1) = 1, D'(1) = f(1) = Om(1)^gamma ≈ 1^gamma = 1
            # Actually D'(1) should be f(1) = Om(1)^gamma. For our ansatz, at a=1, q=1-Om(1)=0, u=1+w0+wa*0=1+w0
            # gamma(1) = c0 + c1*0 + c2*0 + c3*(1+w0) + c4*wa*1 + c5*(1+w0)*0 = c0 + c3*(1+w0) + c4*wa
            
            def gamma_of_a(a, coeffs):
                Om = Om_of_a(a)
                q = 1.0 - Om
                u = 1.0 + w0 + wa * (1.0 - a)
                return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            # Check gamma bounds
            for a in a_values:
                g = gamma_of_a(a, coeffs)
                if g < 0.1 or g > 1.2:
                    return float('inf')
            
            # Integrate ODE from a=1 to a=0.1
            def ode_system(x, y):
                a = np.exp(x)
                Dy, Dyy = y
                HpH = H_prime_over_H(a)
                Om = Om_of_a(a)
                return [Dy, -(3.0 + HpH) * Dy + 1.5 * Om * Dy]  # Wait, sign
                # D'' + (3 + H'/H) D' - (3/2) Om D = 0
                # So D'' = -(3 + H'/H) D' + (3/2) Om D
                # But y = [D, D'], so D'' is the derivative of D'
            
            # Let me redo: let y0 = D, y1 = D'
            # y0' = y1
            # y1' = -(3 + H'/H) * y1 + (3/2) * Om * y0
            
            from scipy.integrate import solve_ivp
            
            x_start = 0.0  # ln(1) = 0
            x_end = np.log(0.1)
            
            # Initial conditions at a=1
            D1 = 1.0
            f1 = gamma_of_a(1.0, coeffs)  # f(1) = Om(1)^gamma(1) = 1^gamma = 1
            # Actually Om(1) = 1, so f(1) = 1^gamma = 1
            # But wait, the ansatz gives f = Om^gamma, so f(1) = 1
            # However, the true f(1) might not be exactly 1. Let's use the ODE solution.
            # For the reference, we need to solve the ODE properly.
            # Let's set D(1)=1 and D'(1)=1 as a reasonable initial condition.
            
            y0 = [1.0, 1.0]
            
            try:
                sol = solve_ivp(
                    lambda x, y: [y[1], -(3.0 + H_prime_over_H(np.exp(x))) * y[1] + 1.5 * Om_of_a(np.exp(x)) * y[0]],
                    [x_start, x_end], y0, method='RK45', rtol=1e-8, atol=1e-10, dense_output=True
                )
                if not sol.success:
                    return float('inf')
                
                # Evaluate at test points
                for a in a_values:
                    x = np.log(a)
                    y = sol.sol(x)
                    D_ref = y[0]
                    f_ref = y[1]  # D' = f
                    
                    # Our approximation
                    Om = Om_of_a(a)
                    q = 1.0 - Om
                    u = 1.0 + w0 + wa * (1.0 - a)
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                    f_approx = Om**gamma
                    
                    # Normalized D: D(a)/D(1) = D(a) since D(1)=1
                    D_approx = D_ref * np.exp(-quad(lambda xx: f_approx_from_x(xx, coeffs, Om0, w0, wa), x, 0.0)[0])
                    # Wait, D(a)/D(1) = exp(-int_a^1 f dx) where x=ln a
                    # So D(a) = D(1) * exp(-int_a^1 f(x) dx) = exp(-int_0^x f(xx) dx) if we go from 1 to a
                    # Actually D(a)/D(1) = exp(int_1^a f(x) dx) = exp(int_0^x f(xx) dx)
                    # So D(a) = exp(int_0^x f(xx) dx)
                    
                    def f_from_x(xx):
                        aa = np.exp(xx)
                        Om_aa = Om_of_a(aa)
                        q_aa = 1.0 - Om_aa
                        u_aa = 1.0 + w0 + wa * (1.0 - aa)
                        gamma_aa = c0 + c1*q_aa + c2*q_aa**2 + c3*u_aa + c4*wa*aa + c5*u_aa*q_aa
                        return Om_aa**gamma_aa
                    
                    D_approx = np.exp(quad(f_from_x, 0.0, x, limit=50)[0])
                    
                    # Relative errors
                    if abs(f_ref) > 1e-10:
                        rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                        max_rel_err = max(max_rel_err, rel_err_f)
                    
                    if abs(D_ref) > 1e-10:
                        rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                        max_rel_err = max(max_rel_err, rel_err_D)
                        
            except Exception:
                return float('inf')
        
        return max_rel_err
    
    # Grid search over coefficient space
    # c0 around 0.55, c1-c5 small
    c0_range = np.linspace(0.45, 0.65, 8)
    c1_range = np.linspace(-0.1, 0.1, 5)
    c2_range = np.linspace(-0.05, 0.05, 4)
    c3_range = np.linspace(-0.1, 0.1, 5)
    c4_range = np.linspace(-0.05, 0.05, 4)
    c5_range = np.linspace(-0.05, 0.05, 4)
    
    for c0 in c0_range:
        for c1 in c1_range:
            for c2 in c2_range:
                for c3 in c3_range:
                    for c4 in c4_range:
                        for c5 in c5_range:
                            if time.time() > deadline:
                                return best_coeffs
                            
                            coeffs = [c0, c1, c2, c3, c4, c5]
                            err = compute_error(coeffs)
                            if err < best_error:
                                best_error = err
                                best_coeffs = coeffs
    
    return best_coeffs
# EVOLVE-BLOCK-END
