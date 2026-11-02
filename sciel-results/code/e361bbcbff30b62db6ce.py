"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import erf, sqrt, pi
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite

# H_n(0) values for even n
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1680.0
_H12_0 = -145152.0

def _poly_coeffs(a0, a4, a8):
    """
    Return coefficients of the degree-12 polynomial P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    in descending power order: [c12, c10, c8, c6, c4, c2, c0]
    P(x) = c12*x^12 + c10*x^10 + c8*x^8 + c6*x^6 + c4*x^4 + c2*x^2 + c0
    """
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 145152.0
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 42x^10 + 630x^8 - 3780x^6 + 7560x^4 - 5040x^2 + 945
    c12 = a12
    c10 = -42.0 * a12
    c8 = a8 + 630.0 * a12
    c6 = -28.0 * a8 - 3780.0 * a12
    c4 = a4 + 210.0 * a8 + 7560.0 * a12
    c2 = -6.0 * a4 - 420.0 * a8 - 5040.0 * a12
    c0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    return [c12, c10, c8, c6, c4, c2, c0]

def _poly_eval(x, coeffs):
    """Evaluate polynomial P(x) at x (scalar or array) using Horner's method for even powers."""
    c12, c10, c8, c6, c4, c2, c0 = coeffs
    x2 = x * x
    return ((c12 * x2 + c10) * x2 + c8) * x2 * x2 + ((c6 * x2 + c4) * x2 + c2) * x2 + c0

def _compute_bound_analytic(a0, a4, a8):
    """
    Compute the C4 uncertainty bound analytically.
    Since g = ĝ (Fourier transform is identical for even-degree Hermite-Gaussians with degrees ≡ 0 mod 4),
    the bound is ||g||_1 / min|g|.
    """
    coeffs = _poly_coeffs(a0, a4, a8)
    c12, c10, c8, c6, c4, c2, c0 = coeffs
    
    # Find all real zeros of P(x)
    full_coeffs = np.array([c12, 0, c10, 0, c8, 0, c6, 0, c4, 0, c2, 0, c0])
    roots = np.roots(full_coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    zeros = np.sort(real_roots)
    
    # Check if P(x) has any real zeros (if so, bound is infinite)
    if len(zeros) > 0:
        for z in zeros:
            val = _poly_eval(z, coeffs)
            if abs(val) < 1e-10:
                return np.inf
    
    # P(x) has constant sign. Find the minimum of |P(x)|exp(-x^2/2).
    # Critical points: P'(x) - x*P(x) = 0
    # Q(x) = -c12*x^13 + (12*c12-c10)*x^11 + (10*c10-c8)*x^9 + (8*c8-c6)*x^7 + (6*c6-c4)*x^5 + (4*c4-c2)*x^3 + (2*c2-c0)*x
    # Q(x) = x * R(x^2) where R is degree 6 in x^2
    R_coeffs = np.array([-c12, 0, 12*c12-c10, 0, 10*c10-c8, 0, 8*c8-c6, 0, 6*c6-c4, 0, 4*c4-c2, 0, 2*c2-c0])
    R_roots = np.roots(R_coeffs)
    R_roots_real = R_roots[np.abs(R_roots.imag) < 1e-10].real
    R_roots_pos = R_roots_real[R_roots_real > 1e-14]
    
    # Critical points: x = 0, x = ±sqrt(y) for y in R_roots_pos
    critical_points = [0.0]
    for y in R_roots_pos:
        y_sqrt = np.sqrt(y)
        critical_points.append(y_sqrt)
        critical_points.append(-y_sqrt)
    
    # Vectorize evaluation at critical points
    x_cps = np.array(critical_points)
    x2 = x_cps * x_cps
    P_vals = ((c12 * x2 + c10) * x2 + c8) * x2 * x2 + ((c6 * x2 + c4) * x2 + c2) * x2 + c0
    g_vals = np.abs(P_vals) * np.exp(-x2 / 2.0)
    
    min_val = np.min(g_vals)
    
    if min_val < 1e-15:
        return np.inf
    
    # Compute ||g||_1 = integral |P(x)| exp(-x^2/2) dx over R
    # Since P has constant sign, this is just |integral P(x) exp(-x^2/2) dx|
    # Use antiderivative formula
    cutoff = 20.0
    
    def _I0(x):
        return sqrt(pi/2) * erf(x / sqrt(2))
    
    def _I2(x):
        return -x * np.exp(-x**2/2) + _I0(x)
    
    def _I4(x):
        return -x**3 * np.exp(-x**2/2) + 3 * _I2(x)
    
    def _I6(x):
        return -x**5 * np.exp(-x**2/2) + 5 * _I4(x)
    
    def _I8(x):
        return -x**7 * np.exp(-x**2/2) + 7 * _I6(x)
    
    def _I10(x):
        return -x**9 * np.exp(-x**2/2) + 9 * _I8(x)
    
    def _I12(x):
        return -x**11 * np.exp(-x**2/2) + 11 * _I10(x)
    
    def F(x):
        val = c0 * _I0(x) + c2 * _I2(x) + c4 * _I4(x) + c6 * _I6(x) + c8 * _I8(x) + c10 * _I10(x) + c12 * _I12(x)
        return val
    
    l1_norm = abs(F(cutoff) - F(-cutoff))
    
    return l1_norm / min_val


def _cmaes_minimize(objective, x0, sigma0, bounds, max_iter=500, seed=42):
    """
    Simple CMA-ES implementation for 3D optimization.
    """
    rng = np.random.default_rng(seed)
    n = len(x0)
    
    # CMA-ES parameters
    mu = max(1, n // 2)
    mu_eff = 1.0 / sum(1.0 / (i + 1) for i in range(mu))
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = (mu_eff / mu_eff) * (2 / ((n + 2) ** 2 + mu_eff) - c_1)
    
    # Initialize
    x = np.array(x0, dtype=float)
    sigma = sigma0
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    chi_n = sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
    
    best_x = x.copy()
    best_f = objective(x)
    
    for iteration in range(max_iter):
        # Generate population
        population = []
        for i in range(mu):
            z = rng.standard_normal(n)
            # Approximate C^(1/2) using eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            xi = x + sigma * (C_sqrt @ z)
            population.append(xi)
        
        # Evaluate
        fitness = [(objective(xi), xi) for xi in population]
        fitness.sort(key=lambda t: t[0])
        
        # Update best
        if fitness[0][0] < best_f:
            best_f = fitness[0][0]
            best_x = fitness[0][1].copy()
        
        # Selection weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Update mean
        x_new = np.sum([weights[i] * fitness[i][1] for i in range(mu)], axis=0)
        
        # Update evolution paths
        sigma_old = sigma
        ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma_old * (C @ (x_new - x))
        if np.linalg.norm(ps) / chi_n < (1.4 + c_c) * sqrt(1 - (1 - c_sigma)**((iteration + 1) * c_sigma)):
            pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma_old
        
        # Update covariance matrix
        delta_sig = (np.linalg.norm(ps) / chi_n - 1) * c_sigma * d_sigma
        hsig = 1 if np.linalg.norm(ps) / chi_n < (1.4 + c_c) * sqrt(1 - (1 - c_sigma)**((iteration + 1) * c_sigma)) else 0
        C = (1 - c_1 - c_mu + hsig * c_1 * (1 - c_c)) * C
        C += c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * C)
        C += c_mu * np.sum([weights[i] * (fitness[i][1] - x) @ (fitness[i][1] - x).T for i in range(mu)], axis=0)
        
        # Ensure C is positive definite
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
        x = x_new
        
        # Clamp to bounds
        for i in range(n):
            lo, hi = bounds[i]
            x[i] = np.clip(x[i], lo, hi)
        
        if sigma < 1e-15:
            break
    
    return best_x, best_f


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the C4 uncertainty bound.
    """
    start_time = time.time()
    deadline = 210.0 - 15.0  # 15 second safety margin
    
    def objective(params):
        a0, a4, a8 = params
        return _compute_bound_analytic(a0, a4, a8)
    
    # Start with the known prior
    best_a0, best_a4, best_a8 = -113/100.0, 1/25.0, 1/3240.0
    best_bound = _compute_bound_analytic(best_a0, best_a4, best_a8)
    
    # Phase 1: Coarse grid search to identify promising regions
    n_coarse = 8
    a0_range = np.linspace(-3, 3, n_coarse)
    a4_range = np.linspace(-1.0, 1.0, n_coarse)
    a8_range = np.linspace(-0.1, 0.1, n_coarse)
    
    grid_best = []
    for a0 in a0_range:
        if time.time() > deadline:
            break
        for a4 in a4_range:
            if time.time() > deadline:
                break
            for a8 in a8_range:
                if time.time() > deadline:
                    break
                bound = _compute_bound_analytic(a0, a4, a8)
                if bound < best_bound:
                    best_bound = bound
                    best_a0, best_a4, best_a8 = a0, a4, a8
                if bound < 200.0:
                    grid_best.append((a0, a4, a8))
    
    # Phase 2: Multi-start CMA-ES with diverse initializations
    if time.time() < deadline - 60:
        bounds = [(-3.0, 3.0), (-1.0, 1.0), (-0.1, 0.1)]
        
        # Create diverse starting points
        start_points = [(best_a0, best_a4, best_a8)]
        
        # Add grid points with good bounds
        grid_best_sorted = sorted(grid_best, key=lambda p: _compute_bound_analytic(*p))
        for pt in grid_best_sorted[:10]:
            start_points.append(pt)
        
        # Add random points
        rng = np.random.default_rng(123)
        for _ in range(15):
            pt = (
                rng.uniform(-3, 3),
                rng.uniform(-1, 1),
                rng.uniform(-0.1, 0.1)
            )
            start_points.append(pt)
        
        # Run CMA-ES from multiple starts
        for idx, (s0, s4, s8) in enumerate(start_points):
            if time.time() > deadline - 30:
                break
            
            # Use different seeds for diversity
            seed = 42 + idx * 7
            sigma0 = 0.5 if idx == 0 else 0.3
            
            try:
                x_opt, f_opt = _cmaes_minimize(
                    objective, [s0, s4, s8], sigma0, bounds,
                    max_iter=300, seed=seed
                )
                if f_opt < best_bound:
                    best_bound = f_opt
                    best_a0, best_a4, best_a8 = x_opt
            except Exception:
                pass
    
    # Phase 3: Differential evolution for additional global exploration
    if time.time() < deadline - 40:
        try:
            bounds = [(-3.0, 3.0), (-1.0, 1.0), (-0.1, 0.1)]
            result = differential_evolution(
                objective, bounds, maxiter=200, tol=1e-12,
                seed=42, polish=True, timeout=int(deadline - time.time() - 25),
                popsize=15
            )
            if result.fun < best_bound:
                best_a0, best_a4, best_a8 = result.x
                best_bound = result.fun
        except Exception:
            pass
    
    # Phase 4: Local refinement with Nelder-Mead
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective, [best_a0, best_a4, best_a8],
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
            )
            if result.fun < best_bound:
                best_a0, best_a4, best_a8 = result.x
                best_bound = result.fun
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish with Powell
    if time.time() < deadline - 10:
        try:
            result = minimize(
                objective, [best_a0, best_a4, best_a8],
                method='Powell',
                options={'maxiter': 30000, 'xtol': 1e-18, 'ftol': 1e-18}
            )
            if result.fun < best_bound:
                best_a0, best_a4, best_a8 = result.x
                best_bound = result.fun
        except Exception:
            pass
    
    return np.array([best_a0, best_a4, best_a8])
# EVOLVE-BLOCK-END
