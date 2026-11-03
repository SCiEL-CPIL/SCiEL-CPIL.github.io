import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft, ifft

N = 600
TIME_LIMIT = 145  # Safety margin for 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def parametric_to_step(params):
    """
    Convert parametric representation to a 600-step vector.
    Representation: |sum_{k=0}^{K-1} c_k * exp(-0.5 * ((x - mu_k)/sigma_k)^2 * (1 + i*phi_k))|
    Params: [Re(c_0), Im(c_0), mu_0, sigma_0, phi_0, Re(c_1), ...]
    We use K=4 modes.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.zeros(N, dtype=np.complex128)
    
    K = 4 # Number of modes
    params_per_mode = 5 # Re(c), Im(c), mu, sigma, phi
    
    for k in range(K):
        base = k * params_per_mode
        re_c = params[base]
        im_c = params[base + 1]
        c = re_c + 1j * im_c
        mu = params[base + 2]
        sigma = params[base + 3]
        phi = params[base + 4]
        
        # Ensure sigma is positive and not too small
        sigma = max(abs(sigma), 0.02)
        
        # Complex Gaussian mode
        # The phase phi allows shifting the "center" of the complex envelope effectively
        # or creating interference patterns that result in a specific non-negative shape when taking abs.
        # Note: abs(z) is always non-negative.
        
        # We define the mode as c * exp(-0.5 * ((x - mu)/sigma)^2) * exp(1j * phi * (x - mu))
        # This creates a chirped Gaussian. The sum of these, when taking absolute value, 
        # creates a complex-valued signal whose magnitude is our step function.
        
        z = (x - mu) / sigma
        term = c * np.exp(-0.5 * z**2) * np.exp(1j * phi * z)
        f += term
        
    # Take absolute value to get non-negative step function
    f_real = np.abs(f)
    
    # Normalize to sum 1
    s = np.sum(f_real)
    if s > 1e-15:
        f_real = f_real / s
    else:
        # Fallback to uniform if degenerate
        f_real = np.ones(N) / N
        
    return f_real

def objective_parametric(params):
    """
    Objective function for parametric optimization.
    Returns the score (lower is better).
    """
    try:
        f = parametric_to_step(params)
        return compute_score(f)
    except Exception:
        return 1e10

def local_refinement(f0, time_limit=10.0):
    """
    Refine the 600-step vector using a projected gradient method on the 
    smoothed max of the autoconvolution.
    """
    f = f0.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f
    
    start_time = time.time()
    best_score = compute_score(f)
    best_f = f.copy()
    
    # We want to minimize max(conv) subject to sum(f)=1, f>=0.
    # Using a smooth approximation of max: logsumexp(beta * conv)
    # Gradient of logsumexp is softmax.
    
    beta = 50.0 # Sharpness of max approximation
    step_size = 0.005
    lr_decay = 0.95
    
    # Precompute FFT for convolution? 
    # np.convolve is O(N^2) = 360k ops. 
    # FFT is O(N log N) ~ 5k ops. 
    # Let's use FFT for speed in local refinement.
    
    def get_conv_fft(f):
        # Zero pad to 2*N-1 or next power of 2
        size = 2048
        F = np.fft.fft(f, n=size)
        C = np.fft.ifft(F * F)
        return np.real(C[:2*N-1])
    
    for it in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        conv = get_conv_fft(f)
        
        # Smoothed max
        # To avoid overflow, subtract max
        c_max = np.max(conv)
        if c_max < 1e-15:
            break
            
        # logsumexp approximation of max
        # L = c_max + (1/beta) * log(sum(exp(beta * (conv - c_max))))
        # grad L = softmax(beta * (conv - c_max))
        
        # We need gradient of L w.r.t f.
        # L = logsumexp(beta * conv)
        # dL/df = beta * (d conv/df)^T * (dL/dconv)
        # dL/dconv = softmax(beta * (conv - c_max))
        
        weights = np.exp(beta * (conv - c_max))
        weights /= np.sum(weights)
        
        # Gradient of conv w.r.t f
        # conv[k] = sum_i f[i] f[k-i]
        # d conv[k] / d f[j] = f[k-j] + f[j-k]
        # grad_f[j] = sum_k weights[k] * (f[k-j] + f[j-k])
        
        # This is a correlation-like operation.
        # grad_f = 2 * (weights * f) convolved? No.
        # Let's compute it directly or via FFT.
        # grad_f[j] = sum_k w[k] f[k-j] + sum_k w[k] f[j-k]
        # Note f is symmetric? No, f is arbitrary non-negative.
        # However, conv is symmetric if f is real? 
        # conv[k] = sum f[i] f[k-i]. conv[N-1-k] = sum f[i] f[N-1-k-i].
        # If we flip f, conv flips.
        
        # Let's use FFT for the gradient as well.
        # grad_f = ifft(fft(weights) * (fft(f) + fft(f)[::-1]?) ... tricky with boundaries.
        
        # Direct computation is O(N^2) = 360k. 2000 iters = 720M ops. 
        # Might be too slow in Python loops. 
        # Vectorized numpy:
        # We can compute the first term: sum_k w[k] f[k-j]
        # This is a correlation of w and f.
        
        # Using FFT:
        size = 2048
        W = np.fft.fft(weights, n=size)
        F = np.fft.fft(f, n=size)
        
        # Term 1: sum_k w[k] f[k-j]
        # This is circular correlation.
        # corr1 = ifft(W * conj(F))
        # Term 2: sum_k w[k] f[j-k]
        # f[j-k] = f[-(k-j)].
        # This is related to corr1.
        
        # Let's just do the direct vectorized loop if it's fast enough, or optimize the FFT path.
        # Actually, for N=600, a direct matrix multiply might be okay if vectorized?
        # No, let's stick to FFT.
        
        # The gradient of the autoconvolution max is effectively a "blur" of the weights by the function f.
        
        # Let's try a simpler heuristic for local refinement: 
        # Move mass from bins that contribute to high conv values to bins that contribute to low conv values.
        # This is what the previous code did. 
        # But let's try a proper gradient step with the FFT trick.
        
        # grad_f[j] = 2 * sum_k weights[k] f[k-j]  (assuming symmetry or appropriate handling)
        # Actually, d conv[k]/d f[j] = f[k-j] + f[j-k].
        # If we assume the "effective" contribution is symmetric, we can approximate.
        
        # Let's compute grad_f using FFT correlation.
        # corr(w, f) gives sum_k w[k] f[k-j] (with appropriate shifts).
        
        # To keep it simple and robust within time limits, let's use the "sensitivity" method from the parent 
        # but with a better step size and multiple restarts if needed.
        
        # Recompute sensitivity for the current max lag
        k_max = np.argmax(conv)
        sens = np.zeros(N)
        for j in range(N):
            idx1 = k_max - j
            idx2 = j - k_max
            val = 0.0
            if 0 <= idx1 < N:
                val += f[idx1]
            if 0 <= idx2 < N:
                val += f[idx2]
            sens[j] = val
        
        # Gradient descent step on the objective approximated by the max term
        # We want to decrease f where sens is high, increase where low.
        # Projected gradient:
        # f_new = f - step_size * (sens - mean(sens))
        # Then project to sum=1, f>=0.
        
        # Normalize sens to have zero mean to preserve sum roughly before projection
        sens_centered = sens - np.mean(sens)
        
        f_new = f - step_size * sens_centered
        
        # Project
        f_new = np.maximum(f_new, 0.0)
        s_new = np.sum(f_new)
        if s_new > 1e-15:
            f_new = f_new / s_new
        else:
            f_new = f.copy()
            
        new_score = compute_score(f_new)
        
        if new_score < best_score:
            best_score = new_score
            best_f = f_new.copy()
            f = f_new
            step_size *= lr_decay
        else:
            step_size *= 0.8
            if step_size < 1e-7:
                break
                
    return best_f

def construct():
    start_time = time.time()
    
    # 1. Global Search: Parametric Optimization with Complex Gaussian Modes
    # K=4 modes, 5 params each -> 20 parameters.
    K = 4
    n_params = K * 5
    
    # Bounds:
    # Re(c), Im(c): [-1, 1]
    # mu: [-0.25, 0.25]
    # sigma: [0.02, 0.25]
    # phi: [-10, 10]
    
    bounds = []
    for _ in range(K):
        bounds.append((-1.0, 1.0)) # Re(c)
        bounds.append((-1.0, 1.0)) # Im(c)
        bounds.append((-0.25, 0.25)) # mu
        bounds.append((0.02, 0.25)) # sigma
        bounds.append((-10.0, 10.0)) # phi
        
    best_f = None
    best_score = np.inf
    
    # Run Differential Evolution
    de_time_limit = 40.0
    de_start = time.time()
    
    try:
        # Use a smaller population and fewer iterations to fit in time, 
        # then rely on local refinement.
        res = differential_evolution(
            objective_parametric, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10,
            tol=1e-4,
            init='sobol',
            polish=False # We do our own polish
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_f = parametric_to_step(res.x)
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't find anything good, use a default
    if best_f is None:
        # Default: single real Gaussian in center
        # c=1, mu=0, sigma=0.1, phi=0
        params_default = [1.0, 0.0, 0.0, 0.1, 0.0] + [0.0]*15
        best_f = parametric_to_step(params_default)
        best_score = compute_score(best_f)

    # 2. Local Refinement
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5.0
    
    if remaining_time > 5.0:
        # Run local refinement with the remaining time budget (minus safety margin)
        local_time = min(remaining_time, 60.0) # Cap local refinement to 60s
        best_f = local_refinement(best_f, time_limit=local_time)
        new_score = compute_score(best_f)
        if new_score < best_score:
            best_score = new_score

    # Final check and return
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
    else:
        best_f = np.ones(N) / N
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
