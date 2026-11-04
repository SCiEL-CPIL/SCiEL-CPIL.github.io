import numpy as np
import time

N = 400

def _autoconv_bound(f):
    """Compute a proxy for the C3 upper bound: max |autocorr| at non-zero lags."""
    # f is a 1D array of length N with values ±1
    # Compute autocorrelation via FFT
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    acf = np.fft.ifft(F * np.conj(F)).real
    # acf[0] is the peak (sum of squares = N for ±1)
    # Wrap-around: the autocorrelation for lags -N+1 to N-1
    # acf[1:N] gives lags 1 to N-1, acf[N+1:] gives lags -1 to -(N-1)
    # For a real signal, acf is symmetric, so we just need lags 1 to N-1
    non_zero = np.concatenate([acf[1:N], acf[N+1:2*N]])
    # Actually for real signals, ifft gives acf[k] for k=0..N-1 and acf[N-k] for the negative lags
    # Let's just take the max abs of all non-zero lag autocorrelations
    # acf has length n_fft = 2N
    # Lags: 0, 1, ..., N-1, -N+1, ..., -1 (wrapped)
    # So acf[0] = lag 0, acf[1:N] = lags 1 to N-1, acf[N+1:2*N] = lags -1 to -(N-1)
    # Due to symmetry, max of abs(acf[1:N]) should suffice
    return np.max(np.abs(acf[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    best_f = None
    best_score = np.inf
    
    # Strategy 1: Quasi-periodic construction
    # Use real parts of sums of complex exponentials with irrational frequencies
    # Then threshold to ±1
    
    # Try several quasi-periodic candidates
    np.random.seed(42)
    
    # Generate quasi-periodic signals
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        # Random irrational-like frequencies
        n_harmonics = np.random.randint(3, 15)
        freqs = np.random.uniform(0.1, 5.0, n_harmonics) * (1 + np.sqrt(2))  # Scale by irrational
        phases = np.random.uniform(0, 2 * np.pi, n_harmonics)
        amps = np.random.uniform(0.5, 2.0, n_harmonics)
        
        x = np.linspace(-0.25, 0.25, N, endpoint=False)
        signal = np.zeros(N)
        for i in range(n_harmonics):
            signal += amps[i] * np.cos(2 * np.pi * freqs[i] * x + phases[i])
        
        # Threshold to ±1
        f = np.sign(signal)
        f[f == 0] = 1  # Handle zeros
        
        score = _autoconv_bound(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Strategy 2: Random search with local improvement
    for trial in range(200):
        if time.time() > deadline - 10.0:
            break
        
        f = np.random.choice([-1, 1], size=N)
        score = _autoconv_bound(f)
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
        
        # Local search: flip individual elements to improve
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            for _ in range(N):
                if time.time() > deadline - 15.0:
                    break
                idx = np.random.randint(N)
                old_val = f[idx]
                f[idx] = -old_val
                new_score = _autoconv_bound(f)
                if new_score < score - 1e-12:
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_f = f.copy()
                else:
                    f[idx] = old_val
    
    # Strategy 3: If we have a good candidate, do more intensive local search
    if best_f is not None and time.time() < deadline - 20.0:
        f = best_f.copy()
        score = _autoconv_bound(f)
        # Hill climbing with random restarts within the local neighborhood
        for iteration in range(500):
            if time.time() > deadline - 25.0:
                break
            # Try flipping a random subset of 1-3 elements
            n_flips = np.random.randint(1, 4)
            idxs = np.random.choice(N, size=n_flips, replace=False)
            old_vals = f[idxs].copy()
            f[idxs] = -old_vals
            new_score = _autoconv_bound(f)
            if new_score < score - 1e-12:
                score = new_score
                if score < best_score:
                    best_score = score
                    best_f = f.copy()
            else:
                f[idxs] = old_vals
    
    if best_f is None:
        return np.ones(N)
    
    return best_f
