<<<<<<< SEARCH
def _spectral_design(n=N):
    """
    Design a sequence by directly specifying its DFT magnitude to be flat
    (maximally spread), then taking the real part. This is a structural
    approach: we design in the frequency domain.
    """
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # Start with a random phase spectrum with flat magnitude
    rng = np.random.default_rng(123)
    for trial in range(20):
        # Random phases
        phases = rng.uniform(-np.pi, np.pi, size // 2 + 1)
        # Flat magnitude (all ones) - this gives a sequence with flat spectrum
        mag = np.ones(size // 2 + 1)
        # Ensure Hermitian symmetry for real signal
        spectrum = mag * np.exp(1j * phases)
        spectrum[0] = mag[0]  # DC term real
        if size % 2 == 0:
            spectrum[-1] = mag[-1]  # Nyquist term real
        
        x_full = np.fft.irfft(spectrum, size)
        x = x_full[:n]
        
        # Try different DC offsets
        for dc in [0.0, 0.1, -0.1, 0.5, -0.5]:
            x_test = x + dc
            s = np.sum(x_test)
            if abs(s) < 1e-10:
                continue
            score = _metric_offdiag(x_test)
            if score < 0.5:  # Good enough threshold
                return x_test, score
    
    return None, float('inf')
=======
def _spectral_design(n=N, time_budget=15.0):
    """
    Design a sequence by directly specifying its DFT magnitude to be flat
    (maximally spread), then taking the real part. This is a structural
    approach: we design in the frequency domain.
    
    Improved: optimize phases using a low-dimensional parameterization
    and try multiple random initializations.
    """
    start = time.time()
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    best_x = None
    best_score = float('inf')
    
    rng = np.random.default_rng(123)
    n_trials = 50
    
    for trial in range(n_trials):
        if time.time() - start > time_budget:
            break
        
        # Random phases
        phases = rng.uniform(-np.pi, np.pi, size // 2 + 1)
        # Flat magnitude
        mag = np.ones(size // 2 + 1)
        # Ensure Hermitian symmetry for real signal
        spectrum = mag * np.exp(1j * phases)
        spectrum[0] = mag[0]
        if size % 2 == 0:
            spectrum[-1] = mag[-1]
        
        x_full = np.fft.irfft(spectrum, size)
        x = x_full[:n]
        
        # Try different DC offsets
        for dc in [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3]:
            x_test = x + dc
            s = np.sum(x_test)
            if abs(s) < 1e-10:
                continue
            score = _metric_offdiag(x_test)
            if score < best_score:
                best_score = score
                best_x = x_test.copy()
    
    if best_x is not None:
        return best_x, best_score
    
    return None, float('inf')

def _spectral_phase_optimize(n=N, time_budget=20.0):
    """
    Optimize spectral phases using L-BFGS-B.
    Parameterize: x = IFFT(|X| * exp(i*theta)) where |X| is fixed (flat)
    and theta are the optimization variables.
    
    To ensure real output, use a symmetric phase parameterization:
    theta[k] for k=1..size/2-1, with theta[0]=0, theta[size/2]=0 (if even)
    and theta[size-k] = -theta[k] for Hermitian symmetry.
    """
    start = time.time()
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # Number of free phase parameters
    n_phases = size // 2 - 1  # k=1 to size/2-1 (k=0 and k=size/2 are fixed to 0)
    
    if n_phases <= 0:
        return None, float('inf')
    
    def phase_to_signal(theta):
        """Convert phase vector to real signal."""
        spectrum = np.ones(size // 2 + 1, dtype=complex)
        # k=0: DC, phase=0
        # k=1 to size/2-1: free phases
        spectrum[1:n_phases+1] = np.exp(1j * theta)
        # k=size/2: Nyquist, phase=0 (if size is even)
        if size % 2 == 0:
            spectrum[size//2] = 1.0
        else:
            spectrum[-1] = 1.0
        
        x_full = np.fft.irfft(spectrum, size)
        return x_full[:n]
    
    def objective(theta):
        if time.time() - start > time_budget:
            return 1e10
        x = phase_to_signal(theta)
        # Add small DC to ensure non-zero sum
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        c = _autoconv_lin(x)
        if len(c) <= 1:
            return 0.0
        abs_c = np.abs(c[1:])
        m = np.max(abs_c)
        if m < 1e-15:
            return 0.0
        beta = 30.0
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        return soft_max / (s * s)
    
    # Initialize with random phases
    rng = np.random.default_rng(42)
    best_x = None
    best_score = float('inf')
    
    n_starts = 5
    budget_per_start = time_budget / n_starts
    
    for start_idx in range(n_starts):
        if time.time() - start > time_budget - 5.0:
            break
        
        theta0 = rng.uniform(-np.pi, np.pi, n_phases)
        
        try:
            result = scipy.optimize.minimize(
                objective,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8},
                tol=1e-10
            )
            if time.time() - start < time_budget:
                x = phase_to_signal(result.x)
                # Try with and without DC offset
                for dc in [0.0, 0.1, -0.1]:
                    x_test = x + dc
                    s = np.sum(x_test)
                    if abs(s) < 1e-10:
                        continue
                    score = _metric_offdiag(x_test)
                    if score < best_score:
                        best_score = score
                        best_x = x_test.copy()
        except Exception:
            pass
    
    if best_x is not None:
        return best_x, best_score
    
    return None, float('inf')
>>>>>>> REPLACE

