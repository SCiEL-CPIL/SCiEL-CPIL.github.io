"""
Strategy: Global Stochastic Search (Differential Evolution) + Analytic Gradient Refinement
The uncertainty bound for the Hermite-Gaussian test function is a non-convex function of the
coefficients. Gradient-based methods from the seed often get stuck in local minima.
We use Differential Evolution (DE) to perform a global search over a reasonable parameter space,
and then refine the best candidate with L-BFGS-B. This hybrid approach is more robust to
the complex landscape defined by the high-degree polynomial moments.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    
    g(x) = P(x) e^{-x^2/2}
    P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    
    The uncertainty product is A(g) * A(hat{g}) = sqrt(<x^2> <xi^2>)
    where <x^2> is the variance in position space and <xi^2> is the variance in frequency space.
    """
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330
    
    # Constraint: g(0) = P(0) = 0
    # P(0) = c0*1 + c4*12 + c8*168 + c12*1330 = 0
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    a0 = c0 + 12*c4 + 168*c8 + 1330*c12
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Q(x) = P(x)^2 = \sum q_k x^{2k}
    q = np.convolve(p, p)
    
    # Integrals I_k = \int x^{2k} e^{-x^2} dx = sqrt(pi) 2^k k!
    factors = np.array([math.sqrt(math.pi) * (2**k) * math.factorial(k) for k in range(len(q))])
    I = q * factors
    
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10
        
    # <x^2> = I[1] / I[0]
    x2 = I[1] / I[0]
    
    # Compute <xi^2> using Parseval: <xi^2> = (1/4pi^2) \int |g''(x)|^2 dx / norm_sq
    # g''(x) = R(x) e^{-x^2/2} where R(x) = P'' - 2xP' + (x^2-1)P
    r = np.zeros(8)
    for j in range(8):
        val = 0.0
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        if j >= 1 and j <= 6:
            val += -4*j*p[j]
        if j >= 1 and j <= 7:
            val += p[j-1]
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    s = np.convolve(r, r)
    factors_R = np.array([math.sqrt(math.pi) * (2**k) * math.factorial(k) for k in range(len(s))])
    I_R = s * factors_R
    norm_R_sq = I_R[0]
    
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    product = math.sqrt(x2 * xi2)
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # 180s limit, use 175s for safety
    
    # Seed coefficients
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    # 1. Global Search using Differential Evolution
    # Define bounds for the coefficients. The seed is small, but the optimal might be different.
    # We use a relatively wide range to allow DE to explore.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    try:
        # DE is a population-based global optimizer.
        # maxiter=50, popsize=10 -> 500 evaluations. Each eval is fast (~ms).
        de_res = scipy.optimize.differential_evolution(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-8,
            polish=False # We will polish manually
        )
        
        if de_res.success and de_res.fun < best_val:
            best_val = de_res.fun
            best_coeffs = de_res.x
    except Exception:
        pass
    
    # 2. Refine with L-BFGS-B from the best DE solution
    if time.time() < deadline - 10.0:
        try:
            lbfgs_res = scipy.optimize.minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if lbfgs_res.success and lbfgs_res.fun < best_val:
                best_val = lbfgs_res.fun
                best_coeffs = lbfgs_res.x
        except Exception:
            pass
            
    # 3. If time permits, try a few random restarts with L-BFGS-B to escape local minima
    if time.time() < deadline - 10.0:
        np.random.seed(42)
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            # Random restarts in a reasonable range
            r_coeffs = np.random.normal(0, 1, 3)
            try:
                res_r = scipy.optimize.minimize(
                    lambda x: _compute_bound(x[0], x[1], x[2]),
                    r_coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_r.success and res_r.fun < best_val:
                    best_val = res_r.fun
                    best_coeffs = res_r.x
            except Exception:
                continue

    return best_coeffs
# EVOLVE-BLOCK-END
