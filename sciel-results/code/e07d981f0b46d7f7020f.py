import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    # Use FFT for faster convolution
    conv = np.fft.irfft(np.fft.rfft(f) * np.fft.rfft(f), n=2*N-1)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

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

def structured_initialization(rng, style='gaussian', width=0.05):
    """Create structured initializations with different shapes."""
    x = np.linspace(-0.25, 0.25, N)
    
    if style == 'gaussian':
        f = np.exp(-x**2 / (2 * width**2))
    elif style == 'rectangular':
        f = np.ones(N)
        center = N // 2
        half_width = max(1, int(N * width * 2))
        f = np.zeros(N)
        f[center - half_width:center + half_width + 1] = 1.0
    elif style == 'triangle':
        f = np.maximum(0, 1 - np.abs(x) / width)
    elif style == 'sinc':
        f = np.sinc(x / width)
        f = np.maximum(f, 0)
    elif style == 'bump':
        y = x / width
        f = np.where(np.abs(y) < 1, np.exp(-1/(1-y**2)), 0)
    elif style == 'plateau':
        y = x / width
        f = np.where(np.abs(y) < 0.5, 1.0, 
                     np.where(np.abs(y) < 1.0, 0.5 + 0.5*np.cos(np.pi*(np.abs(y)-0.5)), 0))
    elif style == 'double_peak':
        f = np.exp(-(x - 0.1)**2 / (2*width**2)) + np.exp(-(x + 0.1)**2 / (2*width**2))
    elif style == 'comb':
        f = np.maximum(0, np.cos(2*np.pi*x*20) + 0.5)
    elif style == 'bell':
        f = 1.0 / (1 + (x/width)**4)
    else:
        f = np.ones(N)
    
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s > 1e-12:
        f = f * (N / s)
    
    return f

def objective_from_params(params, style, width):
    """Objective function for parameter optimization."""
    if len(params) < 2:
        return 1e10
    
    # params[0] = width, params[1:] = shape parameters
    w = max(0.01, params[0])
    shape_params = params[1:]
    
    x = np.linspace(-0.25, 0.25, N)
    
    if style == 'gaussian':
        f = np.exp(-x**2 / (2 * w**2))
    elif style == 'rectangular':
        center = N // 2
        half_width = max(1, int(N * w * 2))
        f = np.zeros(N)
        f[center - half_width:center + half_width + 1] = 1.0
    elif style == 'triangle':
        f = np.maximum(0, 1 - np.abs(x) / w)
    elif style == 'sinc':
        f = np.sinc(x / w)
        f = np.maximum(f, 0)
    elif style == 'bump':
        y = x / w
        f = np.where(np.abs(y) < 1, np.exp(-1/(1-y**2)), 0)
    elif style == 'plateau':
        y = x / w
        f = np.where(np.abs(y) < 0.5, 1.0, 
                     np.where(np.abs(y) < 1.0, 0.5 + 0.5*np.cos(np.pi*(np.abs(y)-0.5)), 0))
    elif style == 'double_peak':
        offset = shape_params[0] if len(shape_params) > 0 else 0.1
        f = np.exp(-(x - offset)**2 / (2*w**2)) + np.exp(-(x + offset)**2 / (2*w**2))
    elif style == 'comb':
        freq = shape_params[0] if len(shape_params) > 0 else 20
        f = np.maximum(0, np.cos(2*np.pi*x*freq) + 0.5)
    elif style == 'bell':
        order = shape_params[0] if len(shape_params) > 0 else 4
        f = 1.0 / (1 + (x/w)**order)
    else:
        f = np.ones(N)
    
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s < 1e-12:
        return 1e10
    
    f = f * (N / s)
    return compute_score(f)

def local_refinement(h, start_time, max_time, alpha_init=0.08):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.fft.irfft(np.fft.rfft(current_h) * np.fft.rfft(current_h), n=2*N-1)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                indices = np.arange(j_min, j_max + 1)
                contrib = current_h[m - indices]
                sensitivity[indices] += contrib
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = np.fft.irfft(np.fft.rfft(h_new) * np.fft.rfft(h_new), n=2*N-1)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.4)
        else:
            alpha *= 0.88
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def spectral_phase_optimization(f_init, start_time, max_time, lr_init=0.05, beta=50.0):
    """Optimize phases in frequency domain to flatten autoconvolution peak."""
    # Initialize with random phases
    rng = np.random.RandomState(123)
    phases = rng.uniform(-np.pi, np.pi, N)
    
    # Get magnitude spectrum from initial signal
    F = np.fft.rfft(f_init)
    magnitudes = np.abs(F)
    # Ensure non-zero magnitudes for phase optimization
    magnitudes[magnitudes < 1e-10] = 1e-10
    
    best_val = compute_score(f_init)
    best_phases = phases.copy()
    lr = lr_init
    
    while time.time() - start_time < max_time:
        # Reconstruct signal from magnitudes and current phases
        F_current = magnitudes * np.exp(1j * np.rfft.phases(F, n=N))
        # Actually, let's work directly with complex spectrum
        F_current = magnitudes * np.exp(1j * phases)
        f_current = np.fft.irfft(F_current, n=N)
        f_current = np.maximum(f_current, 0.0)
        
        s = np.sum(f_current)
        if s < 1e-12:
            break
        f_current = f_current * (N / s)
        
        val = compute_score(f_current)
        
        if val < best_val:
            best_val = val
            best_phases = phases.copy()
        
        # Compute gradient of smooth max of autoconvolution w.r.t. phases
        conv = np.fft.irfft(np.fft.rfft(f_current) * np.fft.rfft(f_current), n=2*N-1)
        conv = np.maximum(conv, 0)
        
        # Smooth max gradient
        x_max = np.max(conv)
        if x_max < 1e-15:
            break
        weights = np.exp(beta * (conv - x_max)) / np.sum(np.exp(beta * (conv - x_max)))
        
        # Gradient of conv[m] w.r.t. f_current[j] is f_current[m-j] (with appropriate bounds)
        # Gradient w.r.t. phase[k] = sum_m weights[m] * d(conv[m])/d(phase[k])
        # d(conv[m])/d(f[j]) = f[m-j] for valid m,j
        # d(f[j])/d(phase[k]) = -imag(FFT^{-1}[F * e^{i*phase} * i * delta_{k}])
        
        # Simplified: use finite differences or analytical gradient
        # Let's use a simpler approach: perturb phases and measure change
        
        # For now, use a heuristic: reduce phases where they contribute to peaks
        # This is approximate but can help
        
        # Compute sensitivity of peaks to phase changes
        # For each frequency k, how much does changing phase[k] affect the peak?
        
        # Use a random direction approach with line search
        direction = rng.normal(size=N)
        f_trial = np.fft.irfft(magnitudes * np.exp(1j * (phases + lr * direction)), n=N)
        f_trial = np.maximum(f_trial, 0.0)
        s_trial = np.sum(f_trial)
        if s_trial < 1e-12:
            lr *= 0.5
            continue
        f_trial = f_trial * (N / s_trial)
        val_trial = compute_score(f_trial)
        
        if val_trial < val:
            phases = phases + lr * direction
            lr = min(lr * 1.05, 0.5)
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
    
    # Reconstruct final signal
    F_final = magnitudes * np.exp(1j * best_phases)
    f_final = np.fft.irfft(F_final, n=N)
    f_final = np.maximum(f_final, 0.0)
    s_final = np.sum(f_final)
    if s_final > 1e-12:
        f_final = f_final * (N / s_final)
    
    return f_final, best_val

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Try multiple structured initializations
    styles = ['gaussian', 'rectangular', 'triangle', 'sinc', 'bump', 'plateau', 'double_peak', 'comb', 'bell']
    widths = [0.03, 0.05, 0.08, 0.12, 0.15]
    
    candidates = []
    
    for style in styles:
        if time.time() - start_time > TIME_LIMIT - 80:
            break
        for width in widths:
            if time.time() - start_time > TIME_LIMIT - 80:
                break
            
            f_init = structured_initialization(rng, style=style, width=width)
            if np.sum(f_init) < 1e-12:
                continue
            
            score = compute_score(f_init)
            candidates.append((score, f_init))
    
    # Sort candidates by score and keep top ones
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Refine top candidates
    top_k = min(8, len(candidates))
    
    for i in range(top_k):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        score_init, f_init = candidates[i]
        
        # Do local refinement
        if time.time() - start_time < TIME_LIMIT - 15:
            refined_f, refined_score = local_refinement(
                f_init, start_time, TIME_LIMIT - 15 - i * 5
            )
            if refined_score < best_score:
                best_score = refined_score
                best_f = refined_f.copy()
    
    # Phase 2.5: Try spectral phase optimization on best candidates
    if time.time() - start_time < TIME_LIMIT - 30:
        # Apply spectral phase optimization to the best few candidates
        for i in range(min(3, len(candidates))):
            if time.time() - start_time > TIME_LIMIT - 35:
                break
            
            _, f_init = candidates[i]
            
            try:
                spectral_f, spectral_score = spectral_phase_optimization(
                    f_init, start_time, TIME_LIMIT - 40
                )
                if spectral_score < best_score:
                    best_score = spectral_score
                    best_f = spectral_f.copy()
            except Exception:
                pass
    
    # Phase 3: Final polish on the best result
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
