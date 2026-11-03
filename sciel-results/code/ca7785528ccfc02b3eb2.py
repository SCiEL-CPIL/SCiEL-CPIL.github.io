"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs_and_grad(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    Returns (value, gradient w.r.t ac).
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(ac)
    
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    val = m + np.log(sum_exp) / beta
    
    # Gradient w.r.t ac
    # d/da (1/beta) log sum exp(beta |a|)
    # = (1/beta) * (1/sum) * sum (exp(beta |a_i|) * beta * sign(a_i))
    # = (sum exp(beta |a_i|) * sign(a_i)) / sum_exp
    # Note: sign(0) is 0, which is fine.
    grad_ac = (exp_scaled / sum_exp) * np.sign(ac)
    
    return val, grad_ac

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    We optimize this to find good candidates, then evaluate the true objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm, _ = _soft_max_abs_and_grad(ac, beta)
    return sm / (s * s)

def _objective_smooth_grad(h, beta=50.0):
    """
    Smooth objective and its gradient.
    f(h) = S(h) / (sum(h))^2
    where S(h) = soft_max_abs(ac(h))
    
    grad_h = (grad_S * (sum)^2 - S * 2 * sum * 1) / (sum)^4
            = (grad_S * sum - 2 * S) / (sum)^3
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_fft(h)
    sm, grad_ac = _soft_max_abs_and_grad(ac, beta)
    
    # Chain rule for autoconvolution gradient
    # ac[k] = sum_i h[i] h[i+k]
    # d ac[k] / d h[j] = h[j+k] + h[j-k] (with appropriate boundary handling)
    # This is effectively a convolution of h with itself, but we need the gradient of S w.r.t h.
    # grad_S = grad_ac . (d ac / d h)
    # d ac / d h is a circulant-like matrix if we pad, but here it's linear convolution.
    # Let's compute grad_h directly.
    # ac = ifft(fft(h)^2)
    # d ac / d h = ifft(2 * fft(h) * fft(dh))
    # grad_h = ifft(2 * fft(h) * fft(grad_ac_padded))
    # Wait, ac is real linear convolution.
    # Let's use the property: grad_h = 2 * ifft(fft(h) * fft(grad_ac)) truncated?
    # Actually, simpler: 
    # S = sum_k w_k |ac_k| approx
    # grad_h = sum_k w_k sign(ac_k) d ac_k / d h
    # d ac_k / d h_j = h_{j+k} + h_{j-k} (where defined)
    # This is equivalent to: grad_h = 2 * ifft(fft(h) * fft(pad(grad_ac)))?
    # Let's just compute it via FFT multiplication which is fast.
    # ac = ifft(fft(h) * fft(h))
    # dS/dh = ifft(2 * fft(h) * fft(dS/dac))
    # We need to pad grad_ac to the same size as the FFT used for ac.
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    H = np.fft.rfft(h, fsize)
    # grad_ac is length size (2n-1). We need to pad it to fsize for the FFT.
    # Note: ac from irfft(H*H, fsize) is length fsize, but we only care about first 2n-1.
    # The gradient w.r.t h is the inverse FFT of 2*H*G, where G is FFT of grad_ac padded.
    # Since ac is real, and we used rfft, we must be careful.
    # Let's use the full complex FFT for gradient calculation to avoid symmetry issues.
    H_full = np.fft.fft(h, fsize)
    G_full = np.fft.fft(np.pad(grad_ac, (0, fsize - size)))
    
    grad_h = np.fft.ifft(2 * H_full * G_full).real[:n]
    
    # Objective: J = S / s^2
    # dJ/dh = (grad_S * s^2 - S * 2 * s * 1) / s^4
    #        = (grad_S * s - 2 * S) / s^3
    
    denom = s**3
    if abs(denom) < 1e-9:
        return sm / (s*s), np.zeros_like(h)
        
    grad_obj = (grad_h * s - 2 * sm) / denom
    
    return sm / (s * s), grad_obj

def _gen_chirp(n):
    """Generate a chirp sequence: h[i] = exp(i * pi * i^2 / n)"""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    # Real part for signed step function
    return np.cos(phase)

def _gen_barker_like(n):
    """
    Generate a sequence with good autocorrelation properties.
    Uses a simple construction: alternating signs with some structure.
    """
    h = np.zeros(n)
    for i in range(n):
        # Use a simple pattern that tends to have lower sidelobes
        # This is a heuristic; not guaranteed to be optimal
        if i % 4 == 0:
            h[i] = 1.0
        elif i % 4 == 1:
            h[i] = -1.0
        elif i % 4 == 2:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses gradient hint to prioritize flips.
    """
    h = h.copy()
    best_val = _score(h)
    
    # Get gradient to guide search
    _, grad = _objective_smooth_grad(h, beta=100.0)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Sort indices by potential improvement (gradient magnitude)
        # Flipping h[i] changes h by -2*h[i]. 
        # Change in obj ~ -2*h[i] * grad[i]
        # We want negative change, so we prefer h[i] * grad[i] > 0
        indices = np.argsort(-np.abs(h * grad))
        
        for i in indices:
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
                # Update gradient approximation roughly
                _, grad = _objective_smooth_grad(h, beta=100.0)
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _peak_suppression(h, deadline, max_iters=50):
    """
    Directly target the argmax of the autoconvolution and adjust h to reduce it.
    This is a dedicated phase that complements the smooth surrogate optimization.
    """
    h = h.copy()
    best_val = _score(h)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        ac = _autoconv_fft(h)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        if peak_val < 1e-10:
            break
            
        # Gradient-like step: perturb h in direction that reduces the peak
        # The autoconvolution at lag k is sum_i h[i]*h[i+k]
        # To reduce |ac[peak_idx]|, we can try small perturbations
        
        # Try a few small perturbations around the peak
        step_sizes = [0.01, 0.005, 0.001]
        improved_this_iter = False
        
        for step in step_sizes:
            if time.time() > deadline:
                break
            
            # Perturb elements that contribute to the peak
            # ac[peak_idx] = sum_i h[i] * h[i+peak_idx] for valid i
            # We can try adjusting h[i] or h[i+peak_idx]
            
            # Simple approach: try small random perturbations weighted by contribution
            rng = np.random.default_rng(hash(_score(h)) % (2**32))
            
            for _ in range(5):
                if time.time() > deadline:
                    break
                    
                h_perturbed = h.copy()
                # Perturb a random subset of elements
                n_perturb = max(1, N // 10)
                indices = rng.choice(N, n_perturb, replace=False)
                h_perturbed[indices] += step * rng.standard_normal(n_perturb)
                
                # Normalize to maintain sum
                s = np.sum(h_perturbed)
                if abs(s) > 1e-9:
                    h_perturbed = h_perturbed / s
                
                val = _score(h_perturbed)
                if val < best_val:
                    h = h_perturbed
                    best_val = val
                    improved_this_iter = True
                    break
        
        if not improved_this_iter:
            # Try more structured perturbations
            for i in range(0, N, max(1, N//20)):
                if time.time() > deadline:
                    break
                
                for delta in [-step, step]:
                    h_perturbed = h.copy()
                    h_perturbed[i] += delta
                    s = np.sum(h_perturbed)
                    if abs(s) > 1e-9:
                        h_perturbed = h_perturbed / s
                    val = _score(h_perturbed)
                    if val < best_val:
                        h = h_perturbed
                        best_val = val
                        improved_this_iter = True
                        break
                if improved_this_iter:
                    break
        
        if not improved_this_iter:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 7. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    # Adaptive beta schedule: tighter progression
    beta_schedule = [60, 120, 250, 500]
    
    # Optimization loop using smooth surrogate
    for h_init in inits:
        if time.time() > deadline - 30:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Use smooth objective for gradient-based optimization with adaptive beta
        try:
            for beta in beta_schedule:
                if time.time() > deadline - 20:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                def obj_grad(x):
                    val, grad = _objective_smooth_grad(x, beta=beta)
                    return val, grad
                
                res = minimize(obj, h_norm, method='L-BFGS-B', 
                               jac=obj_grad,
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-5})
                h_opt = res.x
                
                # Check validity
                if abs(np.sum(h_opt) - 1.0) < 0.5:
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
                        # Update h_norm for next beta stage
                        h_norm = h_opt / np.sum(h_opt)
                else:
                    # Try to renormalize
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-6:
                        h_opt = h_opt / s_opt
                        val = _score(h_opt)
                        if val < best_score:
                            best_score = val
                            best_h = h_opt.copy()
                            h_norm = h_opt.copy()
        except Exception:
            pass

    # Dedicated peak-suppression phase
    if time.time() < deadline - 15:
        h_suppressed, val_suppressed = _peak_suppression(best_h, deadline - 10, max_iters=30)
        if val_suppressed < best_score:
            best_score = val_suppressed
            best_h = h_suppressed.copy()

    # Final high-precision polish using high beta gradient
    if time.time() < deadline - 5:
        h_polish = best_h.copy()
        s_polish = np.sum(h_polish)
        if abs(s_polish) > 1e-6:
            h_polish = h_polish / s_polish
            
        def obj_polish(x):
            return _objective_smooth(x, beta=1000.0)
        def grad_polish(x):
            val, grad = _objective_smooth_grad(x, beta=1000.0)
            return val, grad

        try:
            res_polish = minimize(obj_polish, h_polish, method='L-BFGS-B', jac=grad_polish,
                                  options={'maxiter': 50, 'disp': False, 'gtol': 1e-8})
            h_final = res_polish.x
            val_final = _score(h_final)
            if val_final < best_score:
                best_score = val_final
                best_h = h_final.copy()
        except Exception:
            pass

    # Final local search refinement (very tight)
    if time.time() < deadline - 2:
        h_refined, val_refined = _local_search_flip(best_h, max_passes=2, deadline=deadline-1)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
