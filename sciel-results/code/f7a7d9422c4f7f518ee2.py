import numpy as np

N = 400

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We want to minimize: abs(2 * N * max(conv(h, h)) / sum(h)**2)
    # This is equivalent to minimizing max(conv(h, h)) / sum(h)**2 (since 2N is constant)
    
    best_h = np.ones(N)
    best_val = np.inf
    
    # Strategy: Use a chirp signal with a window to reduce sidelobes.
    # h[n] = window[n] * exp(1j * pi * mu * n^2)
    # We take the real part.
    # The autoconvolution of a real chirp is related to the magnitude of the complex chirp's autoconvolution.
    
    n = np.arange(N)
    # Center the chirp to handle negative indices in the quadratic phase properly if needed, 
    # but for autoconvolution, shifting doesn't change the magnitude of the sidelobes, 
    # only their position. The max is over all lags, so position doesn't matter for the max value.
    # However, the sum h is affected by the phase if we don't center? 
    # No, sum(exp(1j * pi * mu * n^2)) depends on mu.
    # To maximize the sum (denominator), we want the phases to align as much as possible.
    # A linear phase (mu=0) gives max sum. But mu=0 is a constant (or windowed constant).
    # A small mu gives a good balance.
    
    # Let's try a few values of mu and window types.
    # Window: Hann is good for sidelobe reduction.
    window = np.hanning(N)
    
    # Test a range of chirp rates mu
    # mu controls the "spread" of the frequency.
    # If mu is too large, the sum might become small due to phase cancellation.
    # If mu is 0, it's just a windowed constant.
    
    # For a windowed constant (mu=0):
    # sum(h) ~ N/2
    # max(conv) ~ sum(h^2) ~ N/10
    # val ~ 2*N*(N/10) / (N/2)^2 = 0.8 -> Score 2.5
    
    # Let's see if we can do better.
    # Maybe a different window? Hamming has lower first sidelobe but higher others?
    # Or maybe a slightly chirped signal improves the max/sum ratio?
    
    # Let's evaluate a few candidates quickly.
    candidates = []
    
    # Candidate 1: Hann window, mu=0
    h1 = window
    candidates.append(h1)
    
    # Candidate 2: Hann window, small chirp
    # The quadratic phase is exp(1j * pi * mu * n^2)
    # We want the real part to have a large DC component in the Fourier domain?
    # Actually, we want the autoconvolution peak (lag 0) to be large relative to sidelobes.
    # And we want sum(h) to be large.
    
    # Let's try a few mu values.
    # The "chirp" nature helps spread energy in frequency, which compresses in time (autoconv).
    # Wait, autoconvolution of a chirp is a triangle-like shape with low sidelobes.
    # The peak is at lag 0.
    
    # Let's just test mu = 0, 0.01, 0.05, 0.1
    mus = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    for mu in mus:
        # Phase term
        # We use n - N/2 to center the chirp, which often helps with numerical stability and symmetry
        n_centered = n - N / 2.0
        phase = np.exp(1j * np.pi * mu * n_centered**2)
        h = window * np.real(phase)
        
        # Ensure sum is not zero
        if abs(np.sum(h)) < 1e-10:
            continue
            
        candidates.append(h)
        
    # Also try Hamming window
    window_ham = np.hamming(N)
    for mu in mus:
        n_centered = n - N / 2.0
        phase = np.exp(1j * np.pi * mu * n_centered**2)
        h = window_ham * np.real(phase)
        if abs(np.sum(h)) < 1e-10:
            continue
        candidates.append(h)
        
    # Evaluate candidates
    for h in candidates:
        if time.time() > deadline:
            break
            
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
            
        # Compute autoconvolution
        # np.convolve is O(N^2) for N=400, which is 160,000 operations. Very fast.
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        
        val = abs(2 * N * max_conv / (s ** 2))
        
        if val < best_val:
            best_val = val
            best_h = h
            
    return best_h
