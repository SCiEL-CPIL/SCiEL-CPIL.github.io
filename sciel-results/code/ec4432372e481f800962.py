"""
Strategy: Spectral Chebyshev Approximation for Universal Quadrature.

This approach fundamentally rethinks the problem by leveraging the smoothness
of the luminosity distance integrand. Instead of treating the quadrature rule
as a black-box optimization problem over 16 parameters (8 nodes, 8 weights),
we model the integrand $f(x) = 1/E(zx)$ as a function of $x \in [0,1]$ that
can be well-approximated by a truncated Chebyshev series.

Key Structural Differences:
1. **Spectral Representation**: We assume $f(x; z, w_0, w_a) \approx \sum_{k=0}^{K} c_k(z, w_0, w_a) T_k(2x-1)$.
   The coefficients $c_k$ depend on the cosmology and redshift.
2. **Exact Integration of Approximation**: We seek nodes $\{x_i\}$ and weights $\{w_i\}$ such that
   $\sum_i w_i f(x_i) \approx \sum_k c_k \int_0^1 T_k(2x-1) dx$ for all $k \le K$.
   This is equivalent to finding a quadrature rule that is exact for the first $K+1$ Chebyshev polynomials.
3. **Global Optimality**: By matching the spectral moments of the integrand family (averaged over a
   broad parameter space), we design a rule that is "spectrally optimal" for the class of functions,
   rather than just fitting a discrete grid of test cases. This should generalize better to validation
   cosmologies.
4. **Efficient Optimization**: We optimize the nodes and weights to minimize the error against
   the spectral moments, which is a smoother and more structured objective than the raw worst-case
   error on a grid. We use a two-stage approach: first, find nodes/weights that exactly integrate
   the Chebyshev basis up to a certain order, then polish with a local minimax search on a
   diverse set of cosmologies.

This avoids the local minima traps of black-box search on discrete grids by using the
analytic structure of the integrand family.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

# Deadline guard
START_TIME = time.time()
DEADLINE = 85  # seconds, leave 5s margin

def E_z(z, w0, wa):
    """Calculate E(z) = H(z)/H0 for flat universe with CPL w(z) = w0 + wa*(1 - 1/(1+z))"""
    if z < 0:
        return 1.0
    # E(z)^2 = (1+z)^(-3*w0) * (1 - wa * z / (1+z))^(1.5*wa)
    term1 = (1 + z) ** (-3 * w0)
    base = 1 - wa * z / (1 + z)
    if base < 0:
        base = 1e-12
    term2 = base ** (1.5 * wa)
    return np.sqrt(term1 * term2)

def true_integral_high_res(z, w0, wa):
    """High-precision reference integral using Gauss-Legendre 100 points."""
    if z == 0:
        return 1.0
    x_gl, w_gl = np.polynomial.legendre.leggauss(100)
    x_map = 0.5 * (x_gl + 1)
    w_map = w_gl / 2.0
    z_vals = z * x_map
    E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals])
    return np.sum(w_map / E_vals)

def quadrature_approx(nodes, weights, z, w0, wa):
    """Calculate the quadrature approximation: sum w_i / E(z * x_i)"""
    if z == 0:
        return 1.0
    z_vals = z * nodes
    E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals])
    return np.sum(weights / E_vals)

def chebyshev_T(n, x):
    """Chebyshev polynomial of the first kind of degree n."""
    return np.cos(n * np.arccos(np.clip(x, -1, 1)))

def compute_chebyshev_moments(num_moments=8, num_samples=20):
    """
    Compute the 'average' Chebyshev moments of the integrand family.
    We sample w0, wa, z from representative ranges and compute the moments 
    of f(x) = 1/E(z*x) with respect to the Chebyshev basis on [0,1].
    """
    w0_vals = np.linspace(-1.5, -0.5, 8)
    wa_vals = np.linspace(-1.0, 1.0, 8)
    z_vals = np.logspace(-1, np.log10(5), 8)
    
    moments = np.zeros(num_moments)
    count = 0
    
    x_gl, w_gl = np.polynomial.legendre.leggauss(200)
    x_map = 0.5 * (x_gl + 1)
    w_map = w_gl / 2.0
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                z_vals_int = z * x_map
                E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals_int])
                f_vals = 1.0 / E_vals
                
                t_vals = 2 * x_map - 1
                for k in range(num_moments):
                    T_k_vals = chebyshev_T(k, t_vals)
                    moment_k = np.sum(w_map * f_vals * T_k_vals)
                    moments[k] += moment_k
                count += 1
    
    moments /= count
    return moments

def minimax_objective(params):
    """
    Objective: Minimize the worst-case relative error over a stratified set.
    params: 16-vector [x1..x8, w1..w8]
    """
    if time.time() > DEADLINE - 2:
        return 1e9

    x = params[:8]
    w = params[8:]
    
    if np.any(x < 0) or np.any(x > 1):
        return 1e9
    if np.any(np.diff(x) <= 0):
        return 1e9
    if np.any(w <= 0):
        return 1e9
        
    w_sum = np.sum(w)
    if w_sum < 1e-6:
        return 1e9
    w_norm = w / w_sum
    
    max_rel_err = 0.0
    
    COSMOLOGIES = [
        (-1.0, 0.0), (-1.0, 0.5), (-1.0, -0.5), (-0.5, 0.0), 
        (-1.5, 0.0), (-0.9, 0.3), (-1.1, 0.2), (-0.8, -0.2), 
        (-1.2, 0.5), (-0.7, 0.4), (-1.0, 1.0), (-1.0, -1.0)
    ]
    Z_VALS = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    for z in Z_VALS:
        for w0, wa in COSMOLOGIES:
            true_val = true_integral_high_res(z, w0, wa)
            if true_val < 1e-10:
                continue
            approx_val = quadrature_approx(x, w_norm, z, w0, wa)
            rel_err = abs(approx_val - true_val) / true_val
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                
    return max_rel_err

def construct():
    # Step 1: Compute average Chebyshev moments
    if time.time() > DEADLINE - 30:
        x_init, w_init = np.polynomial.legendre.leggauss(8)
        x_init = 0.5 * (x_init + 1)
        w_init = w_init / 2.0
        return {"nodes": x_init, "weights": w_init}
    
    moments = compute_chebyshev_moments(num_moments=8)
    
    # Step 2: Solve for nodes and weights using moment matching
    x_init, w_init = np.polynomial.legendre.leggauss(8)
    x_init = 0.5 * (x_init + 1)
    w_init = w_init / 2.0
    
    def moment_residuals(params):
        x = params[:8]
        w = params[8:]
        
        x = np.clip(x, 1e-6, 1 - 1e-6)
        x = np.sort(x)
        
        w_sum = np.sum(w)
        if w_sum < 1e-6:
            w = np.full(8, 1.0/8.0)
        else:
            w = w / w_sum
        
        t_vals = 2 * x - 1
        residuals = np.zeros(8)
        for k in range(8):
            T_k_vals = chebyshev_T(k, t_vals)
            residuals[k] = np.sum(w * T_k_vals) - moments[k]
            
        return residuals
    
    try:
        if time.time() < DEADLINE - 20:
            res = minimize(
                lambda p: np.sum(moment_residuals(p)**2),
                np.concatenate([x_init, w_init]),
                method='L-BFGS-B',
                bounds=[(1e-6, 1-1e-6)]*8 + [(1e-6, 1.0)]*8,
                options={'maxiter': 100}
            )
            if res.success:
                x_init = res.x[:8]
                w_init = res.x[8:]
    except Exception:
        pass
    
    x_init = np.sort(x_init)
    x_init = np.clip(x_init, 1e-8, 1 - 1e-8)
    for i in range(1, 8):
        if x_init[i] <= x_init[i-1]:
            x_init[i] = x_init[i-1] + 1e-8
    x_init = np.clip(x_init, 1e-8, 1 - 1e-8)
    
    w_init = np.abs(w_init)
    w_sum = np.sum(w_init)
    if w_sum < 1e-12:
        w_init = np.full(8, 1.0/8.0)
    else:
        w_init = w_init / w_sum
        
    best_params = np.concatenate([x_init, w_init])
    best_score = minimax_objective(best_params)
    
    # Step 3: Local minimax polish
    try:
        if time.time() < DEADLINE - 5:
            res = minimize(
                minimax_objective,
                best_params,
                method='Powell',
                bounds=[(1e-6, 1-1e-6)]*8 + [(1e-6, 1.0)]*8,
                options={'maxiter': 50, 'maxfev': 200}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass
    
    x = best_params[:8]
    w = best_params[8:]
    
    x = np.sort(x)
    x = np.clip(x, 1e-8, 1 - 1e-8)
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-8
    x = np.clip(x, 1e-8, 1 - 1e-8)
    
    w = np.abs(w)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        w = np.full(8, 1.0/8.0)
    else:
        w = w / w_sum
        
    return {"nodes": x, "weights": w}
