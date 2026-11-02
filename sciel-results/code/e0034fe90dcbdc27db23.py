"""Optimized signed step function for C3-prime minimization."""
import numpy as np
import time

N = 400

def _autoconv_fft(sig):
    """Compute autoconvolution of a signal using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    return ac

def _exact_score(sig):
    """Compute C3' = max|autoconv| / (integral)^2."""
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    ac = _autoconv_fft(sig)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _spectral_init(mag_type='flat', seed=0):
    """Generate signal from spectral phase retrieval with specified magnitude spectrum."""
    rng = np.random.RandomState(seed)
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    
    # Create magnitude spectrum
    freqs = np.arange(n_fft // 2 + 1)
    if mag_type == 'flat':
        mag = np.ones(n_fft // 2 + 1)
    elif mag_type == 'raised_cosine':
        mag = 0.5 + 0.5 * np.cos(np.pi * freqs / (n_fft // 2))
    elif mag_type == 'gaussian':
        mag = np.exp(-0.5 * (freqs / (n_fft // 8))**2)
    else:
        mag = np.ones(n_fft // 2 + 1)
    
    # Random phase
    phase = rng.uniform(0, 2*np.pi, size=n_fft // 2 + 1)
    phase[0] = 0  # DC component real
    
    # Construct spectrum
    spectrum = mag * np.exp(1j * phase)
    
    # Inverse FFT to get time-domain signal
    sig = np.fft.irfft(spectrum, n_fft)
    
    # Keep first N samples
    sig = sig[:N]
    
    # Normalize to have integral 1
    integral = np.sum(sig)
    if abs(integral) > 1e-10:
        sig = sig / integral
    
    # Threshold to signed values
    sig = np.sign(sig)
    sig[sig == 0] = 1.0
    
    return sig

def _local_refinement(sig, max_rounds=5, time_limit=30.0):
    """Local search: flip individual elements to improve score."""
    start_time = time.time()
    best_score = _exact_score(sig)
    
    for _ in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Random order for diversity
        order = np.random.permutation(N)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            new_sig = sig.copy()
            new_sig[i] = -new_sig[i]
            s = _exact_score(new_sig)
            
            if s < best_score - 1e-12:
                best_score = s
                sig = new_sig
                improved = True
        
        if not improved:
            break
    
    return sig, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_sig = None
    best_score = float('inf')
    
    # Phase 1: Generate diverse spectral initializations
    initializations = []
    
    # Flat-top magnitude with different seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        sig = _spectral_init('flat', seed)
        initializations.append(sig)
    
    # Raised-cosine magnitude
    for seed in range(5):
        if time.time() > deadline:
            break
        sig = _spectral_init('raised_cosine', seed + 100)
        initializations.append(sig)
    
    # Gaussian magnitude
    for seed in range(5):
        if time.time() > deadline:
            break
        sig = _spectral_init('gaussian', seed + 200)
        initializations.append(sig)
    
    # Traditional two-block structures
    sig_tb = np.zeros(N)
    sig_tb[:N//2] = 1.0
    sig_tb[N//2:] = -1.0
    initializations.append(sig_tb)
    
    sig_tb2 = np.zeros(N)
    sig_tb2[:N//3] = 1.0
    sig_tb2[N//3:2*N//3] = -1.0
    sig_tb2[2*N//3:] = 1.0
    initializations.append(sig_tb2)
    
    # Phase 2: Local refinement on each initialization
    for init_sig in initializations:
        if time.time() > deadline - 5.0:
            break
        
        refined_sig, refined_score = _local_refinement(
            init_sig, 
            max_rounds=3, 
            time_limit=15.0
        )
        
        if refined_score < best_score:
            best_score = refined_score
            best_sig = refined_sig.copy()
    
    if best_sig is None:
        # Fallback
        best_sig = np.zeros(N)
        best_sig[:N//2] = 1.0
        best_sig[N//2:] = -1.0
        best_score = _exact_score(best_sig)
    
    # Phase 3: Additional refinement on best result
    if time.time() < deadline - 2.0:
        refined_sig, refined_score = _local_refinement(
            best_sig,
            max_rounds=5,
            time_limit=30.0
        )
        
        if refined_score < best_score:
            best_sig = refined_sig
            best_score = refined_score
    
    return best_sig
