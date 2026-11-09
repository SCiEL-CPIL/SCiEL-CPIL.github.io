import math
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # ---------- Vectorized reference growth ODE (in x = ln a) ----------
    # y = D, y' = dD/dx.  f = y'/y.
    # d(y')/dx = -[3/2 + 1.5(1+w) + d(1+w)/dx] y' + (3/2) Om y
    # d(1+w)/dx = wa * d(1-a)/dx = wa * a  (since d(1-a)/dx = -a)

    def h2_of_a_vec(a, Om, w0, wa):
        # H^2/H0^2 = Om a^-3 + (1-Om) exp(3*wa*(1-a))
        return Om * a ** (-3.0) + (1.0 - Om) * np.exp(3.0 * wa * (1.0 - a))

    def om_m_of_a_vec(a, Om, w0, wa):
        h2 = h2_of_a_vec(a, Om, w0, wa)
        return Om * a ** (-3.0) / h2

    def growth_rhs_vec(x, y, yp, Om, w0, wa):
        """Vectorized RHS for RK4 step. x, y, yp can be arrays."""
        a = np.exp(x)
        w = w0 + wa * (1.0 - a)
        dw_dx = wa * a
        Om_a = om_m_of_a_vec(a, Om, w0, wa)
        dyp_dx = -(1.5 + 1.5 * (1.0 + w) + dw_dx) * yp + 1.5 * Om_a * y
        return y, yp, dyp_dx

    def integrate_growth(Om, w0, wa, xs):
        """RK4 integration of growth ODE. xs sorted ascending."""
        n = len(xs)
        Y = np.empty((n, 2))
        Y[0, 0] = 1.0
        Y[0, 1] = 0.5  # initial slope ~ Om(a0)^0.55, but we'll use 0.5 as seed
        
        for i in range(1, n):
            x0 = xs[i - 1]
            x1 = xs[i]
            h = x1 - x0
            y0, yp0 = Y[i - 1, 0], Y[i - 1, 1]
            
            # RK4
            k1y, k1yp, k1dyp = growth_rhs_vec(x0, y0, yp0, Om, w0, wa)
            k2y, k2yp, k2dyp = growth_rhs_vec(x0 + 0.5 * h, y0 + 0.5 * h * k1y, yp0 + 0.5 * h * k1yp, Om, w0, wa)
            k3y, k3yp, k3dyp = growth_rhs_vec(x0 + 0.5 * h, y0 + 0.5 * h * k2y, yp0 + 0.5 * h * k2yp, Om, w0, wa)
            k4y, k4yp, k4dyp = growth_rhs_vec(x0 + h, y0 + h * k3y, yp0 + h * k3yp, Om, w0, wa)
            
            Y[i, 0] = y0 + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
            Y[i, 1] = yp0 + (h / 6.0) * (k1yp + 2 * k2yp + 2 * k3yp + k4yp)
        
        f = Y[:, 1] / Y[:, 0]
        D = Y[:, 0] / Y[-1, 0]
        return f, D

    # ---------- Model ansatz (vectorized) ----------
    def model_f_vec(c, a, Om, w0, wa):
        c0, c1, c2, c3, c4, c5 = c
        Om_a = om_m_of_a_vec(a, Om, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
        return Om_a ** gamma

    def model_D_vec(c, a_grid, Om, w0, wa):
        """D/D(1) = exp(-int_{ln a}^{0} f(x) dx)"""
        xs = np.log(a_grid)
        fs = model_f_vec(c, a_grid, Om, w0, wa)
        
        # Cumulative integral from end (x=0) backwards
        dx = np.diff(xs)
        trap = 0.5 * (fs[:-1] + fs[1:]) * dx
        cum = np.zeros(len(xs))
        cum[-2] = trap[-1]
        if len(xs) > 2:
            cum[:-2] = np.cumsum(trap[:-1][::-1])[::-1]
            cum[:-2] += cum[-2]
        
        D = np.exp(-cum)
        return D

    # ---------- Analytical Taylor expansion initialization ----------
    # In the matter-dominated limit (q->1, u->1), the exact growth index
    # has known Taylor coefficients. We match the first-order derivatives
    # with respect to q and u to get a physically consistent starting point.
    #
    # From GR growth ODE analysis, near q=1, u=1:
    #   gamma(q,u) ≈ gamma_0 + gamma_q*(q-1) + gamma_u*(u-1) + ...
    # where gamma_0 ≈ 0.55 (Edwards et al. 2007),
    # gamma_q ≈ -0.1 (sensitivity to dark energy density),
    # gamma_u ≈ -0.05 (sensitivity to dark energy equation of state).
    #
    # Our ansatz: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # At q=1, u=1: gamma = c0 + c1 + c2 + c3 + c4*wa + c5
    # ∂gamma/∂q at (1,1) = c1 + 2*c2 + c5
    # ∂gamma/∂u at (1,1) = c3 + c5
    #
    # We set c2=c4=0 initially, and solve:
    #   c0 + c1 + c3 + c5 = 0.55
    #   c1 + c5 = -0.1
    #   c3 + c5 = -0.05
    # Choose c5 = 0.02, then c1 = -0.12, c3 = -0.07, c0 = 0.55 + 0.12 + 0.07 - 0.02 = 0.72
    # But c0 should be near 0.55. Let's try c5 = -0.05:
    #   c1 = -0.05, c3 = 0.0, c0 = 0.55 + 0.05 - 0.0 = 0.60
    # Better: c5 = 0.0, c1 = -0.1, c3 = -0.05, c0 = 0.55 + 0.1 + 0.05 = 0.70
    # Hmm, c0 is too large. Let's use a different approach.
    #
    # Actually, the standard parametrization is gamma ≈ 0.55 - 0.1*q + 0.05*(1-u)
    # So: c0 = 0.55, c1 = -0.1, c3 = -0.05, c2=c4=c5=0
    # This gives gamma = 0.55 - 0.1*q - 0.05*u
    # At q=1,u=1: gamma = 0.55 - 0.1 - 0.05 = 0.40 (too low)
    #
    # Correct: gamma ≈ 0.55 + 0.1*(1-q) - 0.05*(u-1)
    # = 0.55 + 0.1 - 0.1*q - 0.05*u + 0.05
    # = 0.70 - 0.1*q - 0.05*u
    # So c0=0.70, c1=-0.1, c3=-0.05
    # But this gives gamma(1,1) = 0.70-0.1-0.05 = 0.55 ✓
    # ∂gamma/∂q = -0.1 ✓
    # ∂gamma/∂u = -0.05 ✓
    
    c_taylor = np.array([0.70, -0.1, 0.0, -0.05, 0.0, 0.0])
    
    # ---------- Cosmology grids ----------
    # Fitting cosmologies (moderate set for speed)
    fit_cosmos = []
    for Om in [0.22, 0.27, 0.32, 0.38]:
        for w0 in [-1.2, -1.0, -0.7, -0.3, 0.0]:
            for wa in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                fit_cosmos.append((Om, w0, wa))

    # a grid for integration (denser for accuracy)
    a_int = np.linspace(0.05, 1.0, 300)

    # Precompute reference data for fitting cosmologies
    ref_data = {}
    for (Om, w0, wa) in fit_cosmos:
        if time.time() > DEADLINE:
            break
        try:
            f_ref, D_ref = integrate_growth(Om, w0, wa, np.log(a_int))
            ref_data[(Om, w0, wa)] = (f_ref, D_ref)
        except Exception:
            pass

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---------- Objective functions ----------
    def objective_smooth(c, beta=100.0):
        """Smooth max via log-sum-exp."""
        c = np.asarray(c)
        all_errs = []
        for key, (f_ref, D_ref) in ref_data.items():
            Om, w0, wa = key
            f_mod = model_f_vec(c, a_int, Om, w0, wa)
            D_mod = model_D_vec(c, a_int, Om, w0, wa)
            err_f = np.abs(f_mod - f_ref) / np.maximum(np.abs(f_ref), 1e-8)
            err_D = np.abs(D_mod - D_ref) / np.maximum(np.abs(D_ref), 1e-8)
            all_errs.append(err_f)
            all_errs.append(err_D)
        
        if not all_errs:
            return 0.0
        
        all_errs = np.concatenate(all_errs)
        m = np.max(all_errs)
        if m < 1e-12:
            return 0.0
        scaled = all_errs / m
        log_sum_exp = m + np.log(np.sum(np.exp(beta * (scaled - 1.0)))) / beta
        return float(log_sum_exp)

    def objective_max(c):
        """Exact max objective."""
        c = np.asarray(c)
        max_err = 0.0
        for key, (f_ref, D_ref) in ref_data.items():
            Om, w0, wa = key
            f_mod = model_f_vec(c, a_int, Om, w0, wa)
            D_mod = model_D_vec(c, a_int, Om, w0, wa)
            err_f = np.abs(f_mod - f_ref) / np.maximum(np.abs(f_ref), 1e-8)
            err_D = np.abs(D_mod - D_ref) / np.maximum(np.abs(D_ref), 1e-8)
            max_err = max(max_err, float(np.max(err_f)), float(np.max(err_D)))
        return max_err

    # ---------- Optimization ----------
    # Start from Taylor expansion point
    best_c = c_taylor.copy()
    best_val = objective_max(best_c)
    
    # Also try the simple seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    val_seed = objective_max(x0)
    if val_seed < best_val:
        best_val = val_seed
        best_c = x0.copy()

    # Diverse starting points including Taylor-based variations
    starts = [
        c_taylor,
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.70, -0.1, 0.0, -0.05, 0.0, 0.0],
        [0.65, -0.08, 0.0, -0.04, 0.0, 0.0],
        [0.75, -0.12, 0.0, -0.06, 0.0, 0.0],
        [0.70, -0.1, 0.02, -0.05, 0.0, 0.0],
        [0.70, -0.1, -0.02, -0.05, 0.0, 0.0],
        [0.70, -0.1, 0.0, -0.05, 0.01, 0.0],
        [0.70, -0.1, 0.0, -0.05, -0.01, 0.0],
        [0.70, -0.1, 0.0, -0.05, 0.0, 0.01],
        [0.70, -0.1, 0.0, -0.05, 0.0, -0.01],
        [0.68, -0.09, 0.01, -0.045, 0.005, 0.005],
        [0.72, -0.11, -0.01, -0.055, -0.005, -0.005],
    ]

    # Stage 1: Broad search with smooth objective
    for s in starts:
        if time.time() > DEADLINE - 20.0:
            break
        s = np.asarray(s)
        try:
            res = minimize(objective_smooth, s, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    best_c = np.clip(best_c, -10.0, 10.0)

    # ---------- Expand cosmology set and refine ----------
    if time.time() < DEADLINE - 15.0:
        extra_cosmos = []
        for Om in [0.20, 0.25, 0.30, 0.35, 0.40]:
            for w0 in [-1.5, -1.0, -0.5, 0.0]:
                for wa in [-0.5, 0.0, 0.5]:
                    if (Om, w0, wa) not in ref_data:
                        extra_cosmos.append((Om, w0, wa))
        
        for (Om, w0, wa) in extra_cosmos:
            if time.time() > DEADLINE - 8.0:
                break
            try:
                f_ref, D_ref = integrate_growth(Om, w0, wa, np.log(a_int))
                ref_data[(Om, w0, wa)] = (f_ref, D_ref)
            except Exception:
                pass
        
        # Stage 2: Refine with smooth objective on expanded set
        if time.time() < DEADLINE - 8.0:
            try:
                res = minimize(objective_smooth, best_c, method='Nelder-Mead',
                               options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x.copy()
            except Exception:
                pass
        
        # Stage 3: High-precision polish with exact max
        if time.time() < DEADLINE - 5.0:
            try:
                res = minimize(objective_max, best_c, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x.copy()
            except Exception:
                pass
        
        best_c = np.clip(best_c, -10.0, 10.0)

    # ---------- Gamma constraint check ----------
    try:
        min_g = 1.2
        max_g = 0.1
        for Om in [0.2, 0.4]:
            for w0 in [-1.5, 0.0]:
                for wa in [-0.5, 0.5]:
                    a_test = np.linspace(0.1, 1.0, 30)
                    Om_a = om_m_of_a_vec(a_test, Om, w0, wa)
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a_test)
                    g = best_c[0] + best_c[1]*q + best_c[2]*q*q + best_c[3]*u + best_c[4]*wa*a_test + best_c[5]*u*q
                    min_g = min(min_g, float(np.min(g)))
                    max_g = max(max_g, float(np.max(g)))
        if min_g < 0.1 or max_g > 1.2:
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    except Exception:
        pass

    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
