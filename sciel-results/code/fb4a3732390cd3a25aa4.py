from scipy.optimize import minimize
import numpy as np
import time
import random

N = 600
TIME_LIMIT = 135

# Precompute grid
# The intervals are equal on [-1/4, 1/4].
# There are N intervals.
x_edges = np.linspace(-0.25, 0.25, N + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

# FFT size for fast convolution
# Convolution of two length-N sequences results in length 2N-1.
# We need FFT size >= 2N-1. Next power of 2 for 1199 is 2048.
FFT_SIZE = 2048

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f)^2)
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    # Use FFT for fast convolution
    F1 = np.fft.rfft(f, n=FFT_SIZE)
    F2 = np.fft.rfft(f, n=FFT_SIZE)
    conv = np.fft.irfft(F1 * F2, n=FFT_SIZE)
    # The linear convolution is in the first 2N-1 elements, 
    # but due to circular convolution wrapping, we must be careful.
    # Actually, for linear convolution of length N and N, result length is 2N-1.
    # If FFT_SIZE >= 2N-1, then irfft gives the linear convolution padded to FFT_SIZE.
    # The values are at indices 0 to 2N-2.
    # However, the "center" of the autoconvolution (peak) is usually near index N-1 (if centered at 0).
    # Let's just take the max of the whole array to be safe, though the peak should be central.
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
        
    max_conv = np.max(conv)
    return max_conv / (s * s)

def freq_to_time(freq_mag, phase):
    """
    Reconstruct time-domain signal from frequency magnitude and phase.
    We assume real-valued signal, so we use rfft/irfft structure.
    freq_mag: real array of length FFT_SIZE//2 + 1
    phase: real array of length FFT_SIZE//2 + 1
    """
    freq_complex = freq_mag * np.exp(1j * phase)
    # Enforce Hermitian symmetry for real signal
    # freq_complex[k] = conj(freq_complex[FFT_SIZE - k])
    # rfft handles this implicitly if we pass the correct half-spectrum, 
    # but irfft expects the half-spectrum directly.
    time_sig = np.fft.irfft(freq_complex, n=FFT_SIZE)
    # Extract the first N points corresponding to our grid
    return time_sig[:N]

def time_to_freq(time_sig):
    """
    Get frequency magnitude and phase from time-domain signal.
    """
    F = np.fft.rfft(time_sig, n=FFT_SIZE)
    mag = np.abs(F)
    phase = np.angle(F)
    return mag, phase

def construct():
    start_time = time.time()
    
    # Initial guess: Gaussian centered at 0
    # f(x) = exp(-x^2 / (2*sigma^2))
    sigma = 0.1
    f_init = np.exp(-0.5 * (x_centers / sigma) ** 2)
    
    # Normalize
    f_init = f_init / np.sum(f_init) * 600.0 # Scale to have sum ~ 600? No, scale doesn't matter for score.
    
    # Get initial frequency representation
    mag, phase = time_to_freq(f_init)
    
    # We will optimize the magnitude spectrum.
    # Constraint: time domain signal must be non-negative.
    # Objective: minimize max(autoconv) / (sum(f))^2
    # Note: max(autoconv) is related to |F|^2. Specifically, autoconv = ifft(|F|^2).
    # So we are minimizing max(ifft(|F|^2)) subject to ifft(F) >= 0.
    
    # Let's use Projected Gradient Descent on the magnitude spectrum?
    # Or directly on the complex spectrum?
    # If we optimize complex spectrum F, we need to maintain consistency with real time signal.
    # Let's optimize the magnitude spectrum M[k] and keep phase fixed from initialization?
    # That might be too restrictive.
    
    # Alternative: Optimize time-domain signal directly but use FFT to compute gradient of the peak?
    # The peak of autoconv is max_k (ifft(|F|^2))[k].
    # This is a non-smooth function.
    
    # Let's stick to the prompt's suggestion: "projected gradient descent on the frequency domain"
    # This is tricky because the non-negativity constraint is in the time domain.
    # Projection onto non-negative time domain in frequency domain is not simple.
    
    # Let's try a different approach: 
    # 1. Initialize with a good time-domain signal (e.g., Gaussian).
    # 2. Use Nelder-Mead or similar on a reduced set of parameters? 
    #    The current program does this with [h, w, c].
    #    
    # The prompt suggests a *structural* change. 
    # Let's try optimizing the time-domain signal directly using a smooth approximation of the max,
    # and use FFT to compute the gradient efficiently?
    # 
    # Actually, the "spectral representation" hint might imply that we should look at the 
    # eigenvalues of the convolution operator? 
    # 
    # Let's try a simple iterative method:
    # 1. Start with f.
    # 2. Compute conv = ifft(|fft(f)|^2).
    # 3. Find the peak index k_max.
    # 4. The gradient of max(conv) with respect to f is related to the impulse at k_max convolved with f?
    #    d/d f_i max_j (sum_k f_k f_{j-k}) = sum_k f_k delta_{j-k, k_max} + ... 
    #    This is essentially the cross-correlation of f with the delta at the peak location.
    #    
    # Let's implement a gradient descent step on the time-domain signal f,
    # projecting onto the non-negative orthant, and using a line search.
    # We will use the log-sum-exp trick to smooth the max function for better gradients.
    
    def smooth_max(x, beta):
        return (1/beta) * np.log(np.sum(np.exp(beta * x)))
        
    def objective_smooth(f, beta):
        F = np.fft.rfft(f, n=FFT_SIZE)
        conv = np.fft.irfft(F * F, n=FFT_SIZE)
        s = np.sum(f)
        if s < 1e-15: return np.inf
        return smooth_max(conv, beta) / (s*s)
        
    def gradient_smooth(f, beta):
        F = np.fft.rfft(f, n=FFT_SIZE)
        conv = np.fft.irfft(F * F, n=FFT_SIZE)
        s = np.sum(f)
        # Gradient of smooth_max(conv)
        weights = np.exp(beta * conv)
        weights /= np.sum(weights)
        # d smooth_max / d conv = weights
        # d conv / d f = 2 * f (reversed/padded) ? 
        # conv = f * f. d/d f_i (f * f)_j = f_{j-i} + f_{j-i} (if symmetric)
        # Actually, conv = ifft(fft(f)^2).
        # Let's use the property that gradient of ||f*f||_inf is roughly f shifted by the peak location.
        # More accurately, if we use log-sum-exp, the gradient is a weighted sum of the gradients of each conv element.
        # grad_conv_j wrt f_i = f_{j-i} (with appropriate padding/reversal).
        # Since f is real, f*f is symmetric.
        # Let's compute the gradient numerically or analytically?
        # Analytical: 
        # Let C_j = sum_k f_k f_{j-k}
        # d C_j / d f_i = f_{j-i} + f_{j-i} (if i != j-i) ? No.
        # C_j = sum_k f_k f_{j-k}.
        # d C_j / d f_i = f_{j-i} (from first term) + f_{j-i} (from second term, if j-k = i => k = j-i)
        # So d C_j / d f_i = 2 f_{j-i} (if we consider f as symmetric? No, f is just a vector).
        # Wait, f is a vector of length N. The convolution is linear.
        # Let's just use finite differences for the gradient of the smooth max? It's expensive.
        
        # Alternative: Use the fact that the peak is usually at the center.
        # Let's just use a simple heuristic: move f in the direction that reduces the peak.
        # The peak is at index k_max. 
        # To reduce C_{k_max}, we should reduce f_i where f_{k_max-i} is large.
        # This is essentially multiplying f by a shifted version of itself?
        
        # Let's try a different structural change: 
        # Optimize the parameters of a "super-Gaussian" or "plateau" function using a more robust global optimizer.
        # The current program uses Gaussian. Let's try a function that is flat in the middle and decays rapidly.
        # f(x) = (1 + cos(pi x / w))^p for |x| < w, 0 otherwise.
        # This allows tuning the "flatness" via p.
        
        # Let's implement this parametric family and use Differential Evolution or similar.
        
        return None # Placeholder
    
    # Let's restart the strategy: 
    # 1. Define a more flexible parametric family: Super-Gaussian / Cosine Plateau.
    # 2. Use a global optimizer like Differential Evolution from scipy to find the best parameters.
    
    def make_cosine_plateau(params):
        """
        params: [height, width, sharpness]
        f(x) = height * (0.5 * (1 + cos(pi * x / width))) ** sharpness for |x| < width
        """
        h, w, p = params
        if h < 0: h = 0
        if w <= 1e-6: w = 1e-6
        if p < 0.1: p = 0.1
        
        f = np.zeros(N)
        mask = np.abs(x_centers) < w
        x = x_centers[mask]
        f[mask] = h * (0.5 * (1 + np.cos(np.pi * x / w))) ** p
        return f

    def objective_params(params):
        f = make_cosine_plateau(params)
        return compute_score(f)

    def local_refinement(params, time_limit):
        start_time_local = time.time()
        try:
            res = minimize(objective_params, params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            if time.time() - start_time_local > time_limit:
                return params, objective_params(params)
            return res.x, res.fun
        except:
            return params, objective_params(params)

    # Global Optimization: Differential Evolution
    # Bounds: 
    # height: [0, 10]
    # width: [0.01, 0.25]
    # sharpness: [1, 10]
    
    bounds = [(0.1, 10.0), (0.01, 0.25), (1.0, 10.0)]
    
    best_score = np.inf
    best_params = [1.0, 0.15, 2.0]
    
    # Run DE for a short time
    if time.time() - start_time < TIME_LIMIT - 30:
        try:
            from scipy.optimize import differential_evolution
            res = differential_evolution(objective_params, bounds, maxiter=50, popsize=15, 
                                         tol=1e-4, seed=42, polish=True, 
                                         callback=lambda x, convergence: None)
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            print("DE failed:", e)
            
    # Refine with NM
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_params, refined_score = local_refinement(best_params, time_limit=10)
        if refined_score < best_score:
            best_score = refined_score
            best_params = refined_params
            
    # Also try the original Gaussian family to compare
    def make_gaussian(params):
        h, w, c = params
        if h < 0: h = 0
        if w <= 1e-6: w = 1e-6
        f = h * np.exp(-0.5 * ((x_centers - c) / w) ** 2)
        return f

    def objective_gaussian(params):
        f = make_gaussian(params)
        return compute_score(f)

    bounds_gauss = [(0.1, 10.0), (0.01, 0.25), (-0.25, 0.25)]
    
    if time.time() - start_time < TIME_LIMIT - 20:
        try:
            from scipy.optimize import differential_evolution
            res_g = differential_evolution(objective_gaussian, bounds_gauss, maxiter=50, popsize=15, 
                                            tol=1e-4, seed=42, polish=True)
            if res_g.fun < best_score:
                best_score = res_g.fun
                # Convert to cosine plateau params for storage? No, just store the function.
                f_best_gauss = make_gaussian(res_g.x)
                # We need to return the best function.
                # Let's keep track of the best function directly.
                # For now, let's assume the cosine plateau is better or we can just pick the best score.
                # We'll compute the final function at the end.
                # Let's store the best function in a variable.
                # We need to modify the logic to store the function, not just params.
                
                # Let's restart this part to store functions.
                pass
        except Exception as e:
            print("DE Gauss failed:", e)
            
    # Let's restructure to store the best function directly.
    
    best_f = None
    best_val = np.inf
    
    # 1. Cosine Plateau Search
    if time.time() - start_time < TIME_LIMIT - 40:
        try:
            from scipy.optimize import differential_evolution
            res = differential_evolution(objective_params, bounds, maxiter=40, popsize=10, 
                                         tol=1e-4, seed=42, polish=False)
            if res.fun < best_val:
                best_val = res.fun
                best_params_cp = res.x
                best_f = make_cosine_plateau(best_params_cp)
        except:
            pass
            
    # Refine Cosine Plateau
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 10:
        refined_params, refined_score = local_refinement(best_params_cp, time_limit=10)
        if refined_score < best_val:
            best_val = refined_score
            best_f = make_cosine_plateau(refined_params)
            
    # 2. Gaussian Search
    if time.time() - start_time < TIME_LIMIT - 20:
        try:
            from scipy.optimize import differential_evolution
            res_g = differential_evolution(objective_gaussian, bounds_gauss, maxiter=40, popsize=10, 
                                            tol=1e-4, seed=42, polish=False)
            if res_g.fun < best_val:
                best_val = res_g.fun
                best_f = make_gaussian(res_g.x)
        except:
            pass
            
    # Refine Gaussian
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 5:
        # We need the params of the current best_f. 
        # This is getting messy. Let's just assume the last refined one is best or compare scores.
        # To keep it simple, let's just use the best found so far.
        pass
        
    # Fallback to constant if nothing worked
    if best_f is None:
        best_f = np.ones(N)
        
    # Ensure non-negative
    best_f = np.maximum(best_f, 0.0)
    
    # Final check
    final_score = compute_score(best_f)
    if final_score < best_val:
        best_val = final_score
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
