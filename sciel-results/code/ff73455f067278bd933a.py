"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

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

def _pgd_waterfilling(h_init, deadline):
    """
    Projected Gradient Descent on the continuous relaxation.
    
    Objective: Minimize ||autoconv(h)||_2^2 subject to h_i in {-1, 1}.
    Relaxation: Minimize ||autoconv(h)||_2^2 + lambda * (||h||_2^2 - N)
    The gradient of ||autoconv(h)||_2^2 is 4 * autoconv(h) (circular approximation for FFT ease, 
    or we can use linear conv gradient).
    
    Actually, the gradient of sum_k (sum_j h_j h_{j+k})^2 is complex.
    A better proxy: Minimize the peak of the autoconvolution.
    
    Let's use the "Water Filling" heuristic directly on the time domain:
    1. Compute autoconvolution C.
    2. Identify peaks in C.
    3. Adjust h to reduce contributions to these peaks.
    
    Alternative Robust Strategy: 
    Minimize the spectral flatness. 
    H = FFT(h). 
    We want |H| to be flat. 
    Gradient of sum(|H|^2 - mean)^2 w.r.t h is linear in h.
    
    Let's implement a "Spectral Equalization" via Gradient Descent on the continuous domain, 
    followed by hard thresholding.
    """
    h = h_init.astype(np.float64).copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    best_h = h.copy()
    best_score = _score(np.sign(h))
    
    # We will iterate in the continuous domain
    # Objective: Minimize the variance of the power spectrum |H(f)|^2
    # This is equivalent to minimizing the peak-to-average power ratio (PAPR) of the autoconvolution
    
    # Gradient of sum_k (|H[k]|^2 - mean)^2 
    # Let P = |H|^2. We want to minimize sum (P - mean(P))^2.
    # d/dh sum (P - mean)^2 = 4 * sum (P - mean) * dP/dh
    # dP/dh is related to the inverse FFT of (P - mean) * H^* / h? No, simpler:
    # P = |H|^2. dP_i/dh_j = 2 Re(H_i^* * dH_i/dh_j) = 2 Re(H_i^* * H_i * delta_{ij}/N) ? No.
    
    # Let's use a simpler, effective heuristic for "Water Filling" the time domain:
    # 1. Compute C = autoconv(h)
    # 2. Find indices where |C| is high.
    # 3. The contribution of h[i] to C[k] is h[i]*h[i-k] + h[i]*h[i+k] (approx).
    # 4. We want to reduce C[k] for large k.
    #    If C[k] > 0, we want to reduce positive correlations at lag k.
    #    If C[k] < 0, we want to reduce negative correlations at lag k.
    
    # Efficient Gradient Step:
    # Let G = 4 * autoconv(h) (gradient of sum C^2 w.r.t h, assuming linear conv)
    # Wait, the gradient of sum_k C[k]^2 w.r.t h[i] is:
    # sum_k 2 C[k] * (dC[k]/dh[i])
    # C[k] = sum_j h[j] h[j+k]
    # dC[k]/dh[i] is non-zero if i=j or i=j+k.
    # This is equivalent to 4 * (h * h) * h ? No, it's 4 * autoconv(h) * h? 
    # Actually, for linear autoconvolution, grad = 4 * (h autoconv h) ? No.
    # Let's use the spectral domain which is cleaner.
    
    # Minimize f(h) = sum_k (|H[k]|^2 - mean)^2
    # H = FFT(h)
    # This is a smooth function. We can use Gradient Descent.
    
    # Precompute FFT size
    fsize = 1
    while fsize < 2*n:
        fsize *= 2
        
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(h)
    
    for step in range(2000):
        if time.time() > deadline:
            break
            
        H = np.fft.fft(h, fsize)
        P = np.abs(H)**2
        
        # We want to flatten P. 
        # Gradient of sum (P - mean(P))^2 w.r.t h:
        # Let E = P - mean(P).
        # d/dh sum E^2 = 4 * sum E * dE/dh
        # E is real. H = F h. P = H H^*.
        # This gradient is 4 * IFFT( E * H^* / N )? 
        # Actually, dP/dh = 2 * IFFT( P * H^* ) ? No.
        
        # Let's use the known result: 
        # The gradient of the PAPR-like objective can be approximated by:
        # grad = 4 * IFFT( (P - mean(P)) * H.conj() )
        # Then take real part.
        
        grad_complex = np.fft.ifft((P - np.mean(P)) * np.conj(H), fsize)
        grad = np.real(grad_complex[:n]) * 4.0
        
        # Gradient step with momentum
        v = momentum * v - lr * grad
        h = h + v
        
        # Project to sphere (norm constraint) or just keep continuous
        # The problem asks for signed step heights. 
        # We can project to the unit sphere to maintain scale, then threshold later.
        norm = np.linalg.norm(h)
        if norm > 1e-9:
            h = h / norm * np.sqrt(n)
            
        # Periodically check the hard-thresholded version
        if step % 50 == 0:
            h_hard = np.sign(h)
            # Ensure sum is not zero
            if abs(np.sum(h_hard)) < 1e-6:
                h_hard[0] = 1.0
            s_val = _score(h_hard)
            if s_val < best_score:
                best_score = s_val
                best_h = h_hard.copy()
                
    # Final hard threshold
    h_final = np.sign(h)
    if abs(np.sum(h_final)) < 1e-6:
        h_final[0] = 1.0
    final_score = _score(h_final)
    if final_score < best_score:
        best_score = final_score
        best_h = h_final.copy()
        
    return best_h, best_score

def _local_search_flip(h, deadline):
    """
    Fast local search using FFT updates.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    best_val = _score(h)
    
    # Precompute autoconv
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    
    # We will try flips that reduce the max absolute value of ac
    # But _score uses max|ac|.
    
    # To speed up, we can use a "soft" max or just check the current max.
    # Let's do a simple hill climbing: try flipping each bit, keep if it reduces _score.
    # To make it faster, we limit the number of passes.
    
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        # Shuffle order to avoid bias
        indices = np.random.permutation(n)
        for i in indices:
            if time.time() > deadline:
                break
            h[i] = -h[i]
            # Quick check: recompute score
            # This is O(N log N). 400 * 20 * log(400) is fine.
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
            
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    best_score = np.inf
    best_h = None
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        h = rng.choice([-1.0, 1.0], size=N)
        candidates.append(h)
        
    # Chirp
    x = np.arange(N)
    h_chirp = np.sign(np.cos(np.pi * x**2 / N))
    candidates.append(h_chirp)
    
    # Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(h_alt)
    
    # Block
    for bs in [2, 4, 8]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // bs) % 2 == 0 else -1.0
        candidates.append(h_block)
        
    # Optimize each candidate
    for h_init in candidates:
        if time.time() > deadline - 10:
            break
            
        # Ensure valid start
        s = np.sum(h_init)
        if abs(s) < 1e-9:
            h_init[0] = -h_init[0]
            
        # Phase 1: Spectral Water Filling (Continuous PGD)
        h_pgd, val_pgd = _pgd_waterfilling(h_init, deadline - 20)
        
        # Phase 2: Local Search Polish (Discrete Bit Flips)
        h_opt, val_opt = _local_search_flip(h_pgd, deadline - 5)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
