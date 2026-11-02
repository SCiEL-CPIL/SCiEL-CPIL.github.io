"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import fftconvolve

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    def compute_value(f):
        """Compute normalized max autoconvolution. Scale-invariant so we normalize sum to 1."""
        s = np.sum(f)
        if s == 0:
            return np.inf
        f_norm = f / s
        conv = fftconvolve(f_norm, f_norm, mode='full')
        return np.max(conv)
    
    def compute_value_with_grad(x):
        """x is log(f) parameters. Returns value and gradient w.r.t. x."""
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return np.inf, None
        f = np.exp(x)
        s = np.sum(f)
        f_norm = f / s
        conv = fftconvolve(f_norm, f_norm, mode='full')
        val = np.max(conv)
        
        # Gradient: d/dx_i of max(conv) where conv_k = sum_j f_norm[j]*f_norm[j+k-N+1]... 
        # Actually conv has length 2N-1, conv[m] = sum_j f_norm[j] * f_norm[m-j] for valid m
        # Let's use the fact that max is achieved at some index m*
        m_star = np.argmax(conv)
        
        # conv[m] = sum_j f_norm[j] * f_norm[m-j] (with appropriate bounds)
        # d(conv[m])/d(f_norm[i]) = f_norm[m-i] + f_norm[i-m] (by symmetry of convolution)
        # d(f_norm[i])/dx_j = delta_ij * f[i]/s - f[i]*f[j]/s^2
        
        # For efficiency, compute gradient numerically if analytic is too complex
        # But let's do analytic:
        # Let g = f_norm, then conv[m] = sum_i g[i] * g[m-i] (circular-like but linear conv)
        # d(conv[m])/dg[i] = g[m-i] + g[i-m] where terms are 0 if index out of [0, N-1]
        
        # Build the gradient of conv[m] w.r.t. g
        grad_conv_wrt_g = np.zeros(N)
        # For linear convolution: conv[m] = sum_{i=0}^{N-1} g[i] * g[m-i] where g[k]=0 if k<0 or k>=N
        # d(conv[m])/dg[i] = g[m-i] + g[i-m]
        for i in range(N):
            j1 = m_star - i
            j2 = i - m_star
            term1 = g_val[j1] if 0 <= j1 < N else 0.0
            term2 = g_val[j2] if 0 <= j2 < N else 0.0
            grad_conv_wrt_g[i] = term1 + term2
        
        # Now d(conv[m])/dx_j = sum_i (d(conv[m])/dg[i]) * (dg[i]/dx_j)
        # dg[i]/dx_j = (delta_ij * s - f[i]) * f[j] / s^2 = (delta_ij - g[i]) * f[j] / s
        # = (delta_ij - g[i]) * g[i] * s / s... wait
        # f = exp(x), g = f/s
        # dg[i]/dx_j = d(f[i]/s)/dx_j = (delta_ij * f[i] * s - f[i] * f[j]) / s^2
        # = f[i] * (delta_ij * s - f[j]) / s^2
        # = g[i] * (delta_ij * s - f[j]) / s
        # = g[i] * (delta_ij - g[j])
        
        # So d(conv[m])/dx_j = sum_i grad_conv_wrt_g[i] * g[i] * (delta_ij - g[j])
        # = g[j] * (sum_i grad_conv_wrt_g[i]*g[i] - sum_i grad_conv_wrt_g[i]*g[i]*g[j])
        # = g[j] * (sum_i grad_conv_wrt_g[i]*g[i] - g[j] * sum_i grad_conv_wrt_g[i]*g[i])
        # = g[j] * (1 - g[j]) * sum_i grad_conv_wrt_g[i]*g[i]
        
        # Wait, that's not right. Let me redo:
        # d(conv[m])/dx_j = sum_i grad_conv_wrt_g[i] * dg[i]/dx_j
        # dg[i]/dx_j = g[i]*(delta_ij - g[j])  ... let me verify
        # f[i] = exp(x_i), s = sum f
        # g[i] = f[i]/s
        # dg[i]/dx_j = (delta_ij * f[i] * s - f[i] * f[j]) / s^2
        # = f[i]/s^2 * (delta_ij * s - f[j])
        # = g[i]/s * (delta_ij * s - f[j])
        # = g[i] * (delta_ij - f[j]/s)
        # = g[i] * (delta_ij - g[j])
        # Yes!
        
        # So d(conv[m])/dx_j = sum_i grad_conv_wrt_g[i] * g[i] * (delta_ij - g[j])
        # = sum_i grad_conv_wrt_g[i]*g[i] * delta_ij - g[j] * sum_i grad_conv_wrt_g[i]*g[i]
        # = grad_conv_wrt_g[j]*g[j] - g[j]*S where S = sum_i grad_conv_wrt_g[i]*g[i]
        
        S = np.dot(grad_conv_wrt_g, g)
        grad_x = g * (grad_conv_wrt_g - S)
        
        return val, grad_x
    
    # Parametric families for good initial guesses
    def make_gaussian(sigma):
        x = np.linspace(-1/4, 1/4, N)
        return np.exp(-x**2 / (2*sigma**2))
    
    def make_raised_cosine(w):
        x = np.linspace(-1/4, 1/4, N)
        f = np.zeros(N)
        mask = np.abs(x) <= w/2
        f[mask] = 0.5 * (1 + np.cos(2*np.pi*x[mask]/w))
        return f
    
    def make_super_gaussian(alpha, sigma):
        x = np.linspace(-1/4, 1/4, N)
        return np.exp(-np.abs(x)**alpha / sigma**alpha)
    
    def make_plateau(w, edge):
        x = np.linspace(-1/4, 1/4, N)
        f = np.zeros(N)
        mask = np.abs(x) <= w/2
        f[mask] = 1.0
        # Smooth edges
        for i in range(N):
            if -w/2 - edge < x[i] < -w/2:
                f[i] = 0.5 * (1 + np.cos(np.pi * (x[i] + w/2) / edge))
            elif w/2 < x[i] < w/2 + edge:
                f[i] = 0.5 * (1 - np.cos(np.pi * (x[i] - w/2) / edge))
        return f
    
    best_f = np.ones(N)
    best_value = compute_value(best_f)
    
    # Generate diverse initial points
    initial_points = []
    
    # Flat
    initial_points.append(np.log(np.ones(N) + 1e-10))
    
    # Gaussians with various widths
    for sigma in [0.02, 0.03, 0.05, 0.07, 0.1]:
        f = make_gaussian(sigma)
        f = np.maximum(f, 1e-10)
        initial_points.append(np.log(f))
    
    # Raised cosines
    for w in [0.1, 0.15, 0.2, 0.25, 0.3]:
        f = make_raised_cosine(w)
        f = np.maximum(f, 1e-10)
        initial_points.append(np.log(f))
    
    # Super-Gaussians
    for alpha in [1.5, 2, 3, 4]:
        for sigma in [0.04, 0.06, 0.08]:
            f = make_super_gaussian(alpha, sigma)
            f = np.maximum(f, 1e-10)
            initial_points.append(np.log(f))
    
    # Plateaus with smooth edges
    for w in [0.1, 0.15, 0.2]:
        for edge in [0.02, 0.04, 0.06]:
            f = make_plateau(w, edge)
            f = np.maximum(f, 1e-10)
            initial_points.append(np.log(f))
    
    # Random smooth functions
    for _ in range(5):
        freqs = np.random.randint(1, 5, size=3)
        amps = np.random.rand(3)
        phases = np.random.rand(3) * 2 * np.pi
        x = np.linspace(-1/4, 1/4, N)
        f = 1.0
        for k in range(3):
            f += amps[k] * np.cos(2*np.pi*freqs[k]*x + phases[k])
        f = np.maximum(f, 1e-10)
        initial_points.append(np.log(f))
    
    # Optimize each initial point
    for idx, x0 in enumerate(initial_points):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        try:
            # First evaluate to get baseline
            f0 = np.exp(x0)
            val0 = compute_value(f0)
            
            # Use L-BFGS-B with analytic gradient
            result = minimize(
                lambda x: compute_value_with_grad(x)[0],
                x0,
                jac=lambda x: compute_value_with_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.x is not None:
                f_opt = np.exp(result.x)
                val_opt = compute_value(f_opt)
                
                if val_opt < best_value:
                    best_value = val_opt
                    best_f = f_opt.copy()
                    
        except Exception:
            continue
    
    # Additional local refinement from best found
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        x_best = np.log(np.maximum(best_f, 1e-10))
        try:
            result = minimize(
                lambda x: compute_value_with_grad(x)[0],
                x_best,
                jac=lambda x: compute_value_with_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.x is not None:
                f_opt = np.exp(result.x)
                val_opt = compute_value(f_opt)
                if val_opt < best_value:
                    best_value = val_opt
                    best_f = f_opt.copy()
        except Exception:
            pass
    
    # Final: try small perturbations around best
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        for _ in range(3):
            noise = np.random.randn(N) * 0.05
            x_pert = np.log(np.maximum(best_f, 1e-10)) + noise
            try:
                result = minimize(
                    lambda x: compute_value_with_grad(x)[0],
                    x_pert,
                    jac=lambda x: compute_value_with_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.x is not None:
                    f_opt = np.exp(result.x)
                    val_opt = compute_value(f_opt)
                    if val_opt < best_value:
                        best_value = val_opt
                        best_f = f_opt.copy()
            except Exception:
                continue
    
    result_heights = np.maximum(best_f, 0)
    return [float(h) for h in result_heights]
# EVOLVE-BLOCK-END
