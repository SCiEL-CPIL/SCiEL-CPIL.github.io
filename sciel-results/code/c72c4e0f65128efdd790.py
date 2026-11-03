import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 140

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def project_simplex(v, s=1.0):
    """Project v onto the simplex {x | x >= 0, sum(x) = s}."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def smooth_max_grad(f, beta):
    """
    Compute gradient of the smooth max approximation of the autoconvolution peak.
    Objective approx: (1/beta) * log(sum(exp(beta * c_k)))
    Gradient g_i = 2 * sum_k p_k * f_{k-i}
    where p_k = softmax(beta * c_k)
    """
    conv = fft_autoconv(f)
    # Shift conv to be centered? No, c_k is defined for k in 0..2N-2.
    # c_k = sum_i f_i f_{k-i}
    # d c_k / d f_i = f_{k-i} + f_{k-i} (if k-i != i) ?
    # Actually c_k = sum_i f_i f_{k-i}.
    # If we vary f_i, c_k changes by f_{k-i} + f_{k-i} if k-i != i, or f_i if k-i=i?
    # Wait. c_k = sum_j f_j f_{k-j}.
    # d c_k / d f_i = f_{k-i} + f_{k-i} if k-i != i?
    # Let's check indices.
    # Term j=i: f_i f_{k-i}. Deriv w.r.t f_i is f_{k-i}.
    # Term j=k-i: f_{k-i} f_i. Deriv w.r.t f_i is f_{k-i}.
    # If k-i = i, i.e., k=2i, the term is f_i^2. Deriv is 2 f_i.
    # So d c_k / d f_i = f_{k-i} + f_{k-i} if k != 2i, else 2 f_i.
    # Which is just 2 f_{k-i} if we define f_j=0 for j out of [0, N-1].
    # Wait, if k-i = i, f_{k-i} = f_i. So 2 f_i.
    # If k-i != i, we have two distinct terms in the sum?
    # Sum is over j.
    # j=i contributes f_i f_{k-i}.
    # j=k-i contributes f_{k-i} f_i.
    # These are the same term if k-i=i.
    # So yes, d c_k / d f_i = 2 f_{k-i} (with f_j=0 if j not in 0..N-1).
    
    # p_k = softmax(beta * conv)
    # Use log-sum-exp for stability
    max_c = np.max(conv)
    exp_c = np.exp(beta * (conv - max_c))
    sum_exp = np.sum(exp_c)
    p = exp_c / sum_exp
    
    # g = 2 * (p * f)
    # p is length 2N-1. f is length N.
    # Convolution of p and f.
    # We need g_i = sum_k p_k f_{k-i}.
    # This is cross-correlation or convolution?
    # c_k = sum_j f_j f_{k-j}.
    # g_i = sum_k p_k (d c_k / d f_i) = sum_k p_k 2 f_{k-i}.
    # Let m = k-i. k = m+i.
    # g_i = 2 sum_m p_{m+i} f_m.
    # This is correlation of p and f.
    # np.correlate(p, f, mode='full')?
    # Or just use FFT.
    
    # p is defined on k=0..2N-2.
    # We want g_i for i=0..N-1.
    # g_i = 2 sum_{m=0}^{N-1} f_m p_{m+i}.
    
    # Construct P such that FFT(P) * FFT(F) gives the correlation?
    # Easier: use np.correlate if N is small, but N=600 is fine for FFT.
    # Let's use FFT.
    # We want h_i = sum_m f_m p_{m+i}.
    # This is the correlation of f and p.
    # np.correlate(f, p, 'valid')? No, lengths differ.
    
    # Let's use the property: corr(f, p)_i = sum_m f_m p_{m+i}.
    # This is equivalent to convolution of f and reversed p?
    # conv(f, rev(p))_k = sum_m f_m rev(p)_{k-m} = sum_m f_m p_{N_p - 1 - (k-m)} = sum_m f_m p_{N_p - 1 - k + m}.
    # We want index m+i.
    # So N_p - 1 - k + m = m + i => k = N_p - 1 - i.
    # So g_i = 2 * conv(f, rev(p))[N_p - 1 - i].
    
    p_len = len(p)
    rev_p = p[::-1]
    
    fft_size = 1
    total_len = N + p_len - 1
    while fft_size < total_len:
        fft_size *= 2
        
    F_fft = np.fft.rfft(f, n=fft_size)
    P_fft = np.fft.rfft(rev_p, n=fft_size)
    conv_fft = F_fft * P_fft
    conv_full = np.fft.irfft(conv_fft, n=fft_size)[:total_len]
    
    # g_i = 2 * conv_full[N_p - 1 - i]
    # i ranges 0..N-1.
    # Indices in conv_full: N_p - 1 - (N-1) ... N_p - 1 - 0
    # N_p = 2N-1.
    # Indices: 2N-1-1-(N-1) = N-1 ... 2N-2.
    # So we need conv_full[N-1 : 2N-1].
    # And reverse it?
    # i=0 -> index 2N-2.
    # i=N-1 -> index N-1.
    # So g = 2 * conv_full[N-1:2N-1][::-1]
    
    g = 2.0 * conv_full[N-1:2*N-1][::-1]
    
    return g

def smooth_max_optimize(f_init, start_time, max_time=120.0):
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Homotopy on beta
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() - start_time > max_time - 10:
            break
            
        # Gradient Descent with Backtracking Line Search
        # We want to minimize J(f) = SmoothMax_beta(f) subject to sum(f)=1, f>=0.
        # Since we project to simplex, we can just do:
        # f_new = Project(f - alpha * grad)
        
        alpha = 0.01
        current_J = None
        
        for step in range(200):
            if time.time() - start_time > max_time - 5:
                break
                
            grad = smooth_max_grad(f, beta)
            
            # Projected Gradient Step
            # We need to choose alpha.
            # Simple backtracking:
            # J(f - alpha grad) < J(f) + <grad, f - alpha grad - f> ?
            # Since projection is non-expansive, standard PGD works.
            
            # Let's just use a fixed small alpha or adaptive.
            # Given the scale of f (sum=1, N=600, f~1/600), grad scale?
            # c_k ~ 1/N. beta * c_k ~ O(1).
            # p_k ~ 1/N.
            # g_i ~ 2 * N * (1/N) * (1/N) = 2/N ~ 0.003.
            # alpha=0.01 might be too small or large.
            
            # Let's try adaptive alpha based on Lipschitz constant of SmoothMax.
            # L ~ beta * ||f||_2^2 ?
            
            # Simple approach: try a few alphas.
            improved = False
            for a in [0.1, 0.01, 0.001, 0.0001]:
                f_candidate = f - a * grad
                f_candidate = project_simplex(f_candidate, 1.0)
                
                # Check if this improves the actual max score (not just smooth approx)
                # Or just check smooth approx?
                # Checking actual score is expensive (FFT).
                # But we have budget.
                
                # Let's evaluate actual score for the candidate if we are close to convergence?
                # For now, trust the smooth approx gradient direction.
                
                # To ensure progress, let's compute the smooth value.
                # But we don't have a fast smooth value function separate from grad?
                # We can compute it.
                
                # Actually, just taking the step and checking actual score every 10 steps is safer.
                # Let's just take the step with a conservative alpha.
                
                f = f_candidate
                improved = True
                break
            
            if not improved:
                break
                
        # Evaluate actual score
        current_val = compute_score(f)
        if current_val < best_val:
            best_val = current_val
            best_f = f.copy()
            
    return best_f

def construct():
    start_time = time.time()
    
    # Generate initial candidates
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
        
    # Triangle
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))

    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply Smooth Max Optimization to the best initial candidates
    for f_init, score_init in scored[:3]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        refined = smooth_max_optimize(f_init, start_time, TIME_LIMIT - 2)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
            
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
