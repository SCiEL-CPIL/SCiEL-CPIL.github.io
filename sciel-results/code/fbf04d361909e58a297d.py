import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fft_autocorr(f):
    """Compute linear autoconvolution using FFT."""
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * F, size)
    return C[:2*n-1]

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def compute_sensitivity_fft(h, peak_mask):
    """Compute sensitivity of autoconvolution peaks to each bin using FFT."""
    n = len(h)
    size = 2 * n
    P = np.fft.rfft(peak_mask.astype(np.float64), size)
    H = np.fft.rfft(h, size)
    corr = np.fft.irfft(P * np.conj(H), size)
    return corr[:n]

def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        sensitivity = compute_sensitivity_fft(current_h, peak_mask)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final polish pass with adaptive step sizes and tighter convergence."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    threshold_frac = 0.98
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time and no_improve_count < max_no_improve:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        sensitivity = compute_sensitivity_fft(current_h, peak_mask)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        effective_alpha = alpha * (1.0 + 0.5 * np.exp(-no_improve_count / 10.0))
        
        h_new = current_h * (1.0 - effective_alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def softmax_objective_and_grad(z, beta=50.0):
    """
    Objective: smooth_max(autocorr(f)) / (sum(f))^2 where f = softmax(z) * N
    Gradient computed analytically using chain rule and FFT.
    """
    # f = softmax(z) * N so that sum(f) = N
    f = softmax(z) * N
    s = N  # sum(f) is always N
    
    conv = fft_autocorr(f)
    max_c = np.max(conv)
    
    # Smooth max: J = smooth_max(conv) / s^2
    # For large beta, smooth_max ≈ max
    # Let's use log-sum-exp form: M = log(sum(exp(beta*conv))) / beta + max(conv)
    # Actually, let's just compute the gradient numerically if analytic is complex
    # But let's try analytic:
    # Let u = conv, M = max(u) (for now, we'll use smooth version)
    # w_m = exp(beta * u_m) / sum(exp(beta * u_k))
    # dM/du_m = w_m
    # du_m/df_j = f_{m-j} + f_j (for valid indices)
    # df_j/dz_k = f_j (delta_jk - f_k)  [softmax gradient]
    
    # For simplicity and speed, let's use a moderate beta and compute gradient
    # via the smooth max formula
    
    shifted = beta * conv - beta * np.max(conv)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    w = exp_shifted / sum_exp  # weights for smooth max
    
    # d(smooth_max)/du_m = w_m
    # du_m/df_j = f_{m-j} + f_j (autoconvolution gradient)
    # We need A_j = sum_m w_m * (f_{m-j} + f_j)
    # A_j = sum_m w_m f_{m-j} + f_j * sum_m w_m
    # sum_m w_m = 1
    # A_j = (corr(w, f))_j + f_j
    
    # corr(w, f)_j = sum_m w_m f_{m-j}
    # This is cross-correlation of w and f
    
    n = N
    size = 2 * n
    W = np.fft.rfft(w, size)
    F = np.fft.rfft(f, size)
    # corr(w, f)_j = sum_m w_m f_{m-j}
    # In FFT: ifft(fft(w) * conj(fft(f))) gives cross-correlation
    # Let's verify: (w * f)(j) = sum_k w(k) f(j-k)
    # We want sum_m w_m f_{m-j} = sum_k w_k f_{k-j}
    # Let l = k-j, k = j+l: sum_l w_{j+l} f_l
    # This is (w_shifted * f)(0) where w_shifted(l) = w_{j+l}
    # Alternatively: cross-correlation R[j] = sum_k w_k f_{k-j}
    # R = ifft(conj(fft(w)) * fft(f)) or ifft(fft(w) * conj(fft(f)))?
    # Standard: (f1 * f2)(n) = sum_k f1(k) f2(n-k)
    # Cross-corr R12[n] = sum_k f1(k+n) f2(k) = (f1_rev * f2)(n) where f1_rev(k) = f1(-k)
    # In FFT: R12 = ifft(conj(FFT(f1)) * FFT(f2))
    # So: R[j] = sum_k w(k+j) f(k) = ifft(conj(FFT(w)) * FFT(f))[j]
    
    R = np.fft.irfft(np.conj(W) * F, size)
    A = R[:n] + f  # A_j = corr_j + f_j
    
    # d(smooth_max)/df_j = A_j
    # But we also have the 1/s^2 factor. Since s = N is constant, dJ/df_j = A_j / N^2
    # Wait, J = smooth_max(conv) / (sum f)^2. sum f = N is constant.
    # So dJ/df_j = (1/N^2) * d(smooth_max)/df_j = A_j / N^2
    
    # Now dJ/dz_k = sum_j (dJ/df_j) * (df_j/dz_k)
    # df_j/dz_k = f_j (delta_jk - f_k)
    # dJ/dz_k = sum_j (A_j / N^2) * f_j (delta_jk - f_k)
    # = (1/N^2) [ A_k f_k - f_k sum_j A_j f_j ]
    
    B = np.dot(A, f)  # sum_j A_j f_j
    grad_z = (A * f - f * B) / (N * N)
    
    # Also need to account for the smooth_max approximation error
    # For large beta, this is accurate
    
    val = smooth_max(conv, beta) / (N * N)
    return val, grad_z

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric search with cosine bumps (quick warm start)
    for n_c in [2, 3, 4]:
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 35:
                break
            
            for restart in range(3):
                if time.time() - start_time > TIME_LIMIT - 30:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Phase 2: L-BFGS-B optimization in softmax parameterization
    # This allows full exploration of the nonnegative simplex
    if time.time() - start_time < TIME_LIMIT - 20:
        # Initialize z from best_f
        f_init = best_f.copy()
        s_init = np.sum(f_init)
        if s_init > 0:
            f_init = f_init / s_init  # normalize to sum=1
            # z = log(f) + constant (arbitrary)
            z0 = np.log(f_init + 1e-10)
        else:
            z0 = np.zeros(N)
        
        # Run L-BFGS-B with increasing beta
        for beta in [20.0, 50.0, 100.0]:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            try:
                result = minimize(
                    softmax_objective_and_grad,
                    z0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,  # We return (val, grad)
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                            'maxls': 50, 'maxfun': 1000}
                )
                
                if result.success or result.fun < best_score:
                    f_new = softmax(result.x) * N
                    score_new = compute_score(f_new)
                    if score_new < best_score:
                        best_f = f_new
                        best_score = score_new
                        z0 = result.x  # continue from here
            except Exception:
                pass
    
    # Phase 3: Local refinement (sensitivity-based)
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 4: Final polish
    if time.time() - start_time < TIME_LIMIT - 3:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 1
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
