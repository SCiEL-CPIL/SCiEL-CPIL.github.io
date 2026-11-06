"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin before 90s kill

    # Safe baseline: constant gamma
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_error = float('inf')

    # Test cosmologies for error evaluation - diverse set
    test_cosmologies = [
        (0.30, 0.0, 0.0),
        (0.30, -0.5, 0.0),
        (0.30, -1.0, 0.5),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.28, -0.8, 0.3),
        (0.32, -1.2, 0.2),
    ]
    a_values = np.linspace(0.1, 1.0, 30)

    def compute_error(coeffs):
        """Compute max relative error of f and D across test cosmologies."""
        c0, c1, c2, c3, c4, c5 = coeffs
        max_rel_err = 0.0

        for Om0, w0, wa in test_cosmologies:
            Om_DE0 = 1.0 - Om0

            def Om_of_a(a):
                if a <= 0:
                    return 1.0
                Om_DE = Om_DE0 * a**(-3.0 * (1.0 + w0 + wa * (1.0 - a)))
                return Om_DE / (Om_DE + Om0 * a**3)

            def gamma_of_a(a):
                Om = Om_of_a(a)
                q = 1.0 - Om
                u = 1.0 + w0 + wa * (1.0 - a)
                return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

            # Check gamma bounds
            for a in a_values:
                g = gamma_of_a(a)
                if g < 0.1 or g > 1.2:
                    return float('inf')

            # Solve ODE for reference: D'' + (3 + H'/H) D' - (3/2) Om D = 0
            # where ' is d/dlna
            def H_prime_over_H(a):
                da = 1e-5
                a_plus = min(a + da, 0.999)
                a_minus = max(a - da, 0.101)
                H2_plus = Om0 * a_plus**(-3) + Om_DE0 * a_plus**(-3 * (1 + w0 + wa * (1 - a_plus)))
                H2_minus = Om0 * a_minus**(-3) + Om_DE0 * a_minus**(-3 * (1 + w0 + wa * (1 - a_minus)))
                H_plus = np.sqrt(max(H2_plus, 1e-10))
                H_minus = np.sqrt(max(H2_minus, 1e-10))
                return (H_plus - H_minus) / (a_plus - a_minus) / ((H_plus + H_minus) / 2)

            def ode_system(x, y):
                a = np.exp(x)
                Dy, Dyy = y
                HpH = H_prime_over_H(a)
                Om = Om_of_a(a)
                return [Dy, -(3.0 + HpH) * Dy + 1.5 * Om * Dy]

            x_start = 0.0
            x_end = np.log(0.1)
            y0 = [1.0, 1.0]

            try:
                sol = solve_ivp(ode_system, [x_start, x_end], y0,
                                method='RK45', rtol=1e-7, atol=1e-9, dense_output=True)
                if not sol.success:
                    return float('inf')

                # Precompute f_approx for D integration
                def f_from_x(xx):
                    aa = np.exp(xx)
                    Om_aa = Om_of_a(aa)
                    q_aa = 1.0 - Om_aa
                    u_aa = 1.0 + w0 + wa * (1.0 - aa)
                    gamma_aa = c0 + c1 * q_aa + c2 * q_aa**2 + c3 * u_aa + c4 * wa * aa + c5 * u_aa * q_aa
                    return Om_aa**gamma_aa

                for a in a_values:
                    x = np.log(a)
                    y = sol.sol(x)
                    D_ref = y[0]
                    f_ref = y[1]

                    Om = Om_of_a(a)
                    q = 1.0 - Om
                    u = 1.0 + w0 + wa * (1.0 - a)
                    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
                    f_approx = Om**gamma

                    # D_approx = exp(int_0^x f(xx) dx)
                    D_approx = np.exp(quad(f_from_x, 0.0, x, limit=30)[0])

                    if abs(f_ref) > 1e-10:
                        rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                        max_rel_err = max(max_rel_err, rel_err_f)

                    if abs(D_ref) > 1e-10:
                        rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                        max_rel_err = max(max_rel_err, rel_err_D)

            except Exception:
                return float('inf')

        return max_rel_err

    # First, try a quick local optimization from the seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    err0 = compute_error(x0)
    if err0 < best_error:
        best_error = err0
        best_coeffs = list(x0)

    # Use differential evolution for global search
    if time.time() < deadline - 20.0:
        try:
            bounds = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)]

            def objective(x):
                if time.time() > deadline:
                    return 1e6
                if any(c < -10 or c > 10 for c in x):
                    return 1e6
                err = compute_error(x)
                return err if err < float('inf') else 1e6

            result = differential_evolution(
                objective, bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False
            )

            if result.fun < best_error:
                best_error = result.fun
                best_coeffs = list(result.x)
                # Polish with Nelder-Mead
                from scipy.optimize import minimize
                res2 = minimize(objective, result.x, method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
                if res2.fun < best_error:
                    best_error = res2.fun
                    best_coeffs = list(res2.x)

        except Exception:
            pass

    # Ensure coefficients are within [-10, 10]
    best_coeffs = [max(-10.0, min(10.0, c)) for c in best_coeffs]

    return best_coeffs
# EVOLVE-BLOCK-END
