"""Signed autocorrelation maximum absolute-value variant - C3' minimization."""
import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

# EVOLVE-BLOCK-START

def _autoconv(x):
    """Compute autoconvolution of x (full, not circular)."""
    return fftconvolve(x, x, mode='full')

def _score(x):
    """Compute C3' = max|autoconv| / (sum(x))^2. Lower is better."""
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    integral = np.sum(x)
    if integral == 0:
        return np.inf
    return max_abs / (integral * integral)

def _barker_init(n):
    """Generate a Barker-like sequence using inverse FFT of phase-only mask."""
    # Random phase mask in frequency domain
    rng = np.random.default_rng(42)
    freq = np.fft.rfftfreq(n)
    # Create a band-limited mask to ensure smoothness
    bandwidth = 0.3
    mask = (np.abs(freq) < bandwidth).astype(float)
    # Random phase
    phases = rng.uniform(0, 2 * np.pi, size=len(freq))
    # Build spectrum: magnitude is flat where mask is 1, phase is random
    spectrum = mask * np.exp(1j * phases)
    # Inverse FFT to get time-domain signal
    sig = np.fft.irfft(spectrum, n=n)
    # Threshold to get ±1 sequence
    x = np.sign(sig)
    x[x == 0] = 1
    return x

def _random_init(n):
    """Random ±1 sequence."""
    rng = np.random.default_rng()
    return rng.choice([-1, 1], size=n)

def _greedy_flip(x, max_iter=5000, deadline=None):
    """Greedy peak shaving: flip sign of element most correlated with max AC peak."""
    x = x.copy().astype(float)
    best_score = _score(x)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(x)
        max_abs = np.max(np.abs(ac))
        integral = np.sum(x)
        if integral == 0:
            break
        current_score = max_abs / (integral * integral)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
        
        # Find the lag with maximum absolute autoconvolution
        peak_lag = np.argmax(np.abs(ac))
        
        # The autoconvolution at lag k is sum_j x[j] * x[j+k]
        # Flipping x[m] changes ac[k] by -2*x[m]*(x[m-k] + x[m+k] - x[m] if in range)
        # More precisely: ac_new[k] = ac[k] - 2*x[m]*(x[m-k] + x[m+k]) + 4*x[m]^2*delta
        # But since we're flipping, the change in ac at lag k due to flipping position m:
        # For each pair (j, j+k), if j==m or j+k==m, the product changes sign
        # Delta ac[k] = -2 * x[m] * (x[m-k] + x[m+k]) where defined
        # And for k=0, delta = -2*x[m]*x[m] + ... actually let's think more carefully
        
        # ac[k] = sum_j x[j]*x[j+k]
        # After flipping x[m]: x[m] -> -x[m]
        # New ac[k] = sum_j x'[j]*x'[j+k]
        # Terms involving m: j=m gives -x[m]*x[m+k], j=m-k gives x[m-k]*(-x[m]) = -x[m-k]*x[m]
        # Original terms: j=m gives x[m]*x[m+k], j=m-k gives x[m-k]*x[m]
        # So delta ac[k] = -2*x[m]*x[m+k] - 2*x[m]*x[m-k] (for valid indices)
        # For k=0: j=m gives -x[m]^2, original x[m]^2, delta = -2*x[m]^2 = -2
        
        # We want to minimize max|ac|, so we should flip the element that most reduces
        # the current peak. Let's compute for the peak lag which flip helps most.
        
        peak_val = ac[peak_lag]
        
        # For each position m, compute how much flipping it would change ac[peak_lag]
        k = peak_lag
        # delta_ac[k] from flipping m:
        # = -2*x[m]*(x[m-k] if m-k>=0 else 0 + x[m+k] if m+k<N else 0)
        # Special case k=0: delta = -2
        
        if k == 0:
            # Flipping any element reduces ac[0] by 2 (since ac[0]=sum x^2 = N always for ±1)
            # Wait, ac[0] = sum x[j]^2 = N for ±1, so flipping doesn't change ac[0]!
            # Actually for ±1, x[m]^2 = 1 always, so ac[0] is invariant.
            # So we can't reduce the peak if it's at lag 0 by flipping.
            # In that case, try to reduce the second largest peak.
            abs_ac = np.abs(ac).copy()
            abs_ac[0] = -1  # Exclude lag 0
            peak_lag2 = np.argmax(abs_ac)
            if abs_ac[peak_lag2] <= 0:
                break
            k = peak_lag2
        
        # Compute delta for each position
        delta = np.zeros(N)
        for m in range(N):
            d = 0.0
            if m - k >= 0:
                d -= 2.0 * x[m] * x[m - k]
            if m + k < N:
                d -= 2.0 * x[m] * x[m + k]
            delta[m] = d
        
        # We want to flip the m that most reduces |ac[peak_lag]|
        # New value at peak: ac[peak] + delta[m]
        # We want to minimize |ac[peak] + delta[m]|
        new_vals = ac[k] + delta
        # Find m that minimizes |new_vals[m]|
        best_m = np.argmin(np.abs(new_vals))
        
        # Only flip if it actually improves
        if np.abs(new_vals[best_m]) < np.abs(ac[k]) - 1e-12:
            x[best_m] *= -1
        else:
            # Try second best
            order = np.argsort(np.abs(new_vals))
            flipped = False
            for m in order[:10]:
                if np.abs(new_vals[m]) < np.abs(ac[k]) - 1e-12:
                    x[m] *= -1
                    flipped = True
                    break
            if not flipped:
                break
    
    return x, best_score

def _local_search(x, deadline=None, max_iter=2000):
    """Local search: try flipping pairs of elements to reduce max|AC|."""
    x = x.copy().astype(float)
    best_score = _score(x)
    best_x = x.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(x)
        abs_ac = np.abs(ac)
        max_abs = np.max(abs_ac)
        integral = np.sum(x)
        if integral == 0:
            break
        current_score = max_abs / (integral * integral)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
        
        # Find top 3 peaks
        peak_indices = np.argsort(abs_ac)[-3:]
        
        improved = False
        for k in peak_indices:
            if abs_ac[k] <= 0:
                continue
            
            # For this peak lag k, find best single flip
            delta = np.zeros(N)
            for m in range(N):
                d = 0.0
                if m - k >= 0:
                    d -= 2.0 * x[m] * x[m - k]
                if m + k < N:
                    d -= 2.0 * x[m] * x[m + k]
                delta[m] = d
            
            new_vals = ac[k] + delta
            best_m = np.argmin(np.abs(new_vals))
            
            if np.abs(new_vals[best_m]) < abs_ac[k] - 1e-12:
                x[best_m] *= -1
                improved = True
                break
        
        if not improved:
            break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_x = None
    best_score = np.inf
    
    # Try multiple initializations
    inits = []
    
    # Barker-like inits with different seeds
    for seed in range(16):
        rng = np.random.default_rng(seed)
        freq = np.fft.rfftfreq(N)
        bandwidth = 0.2 + 0.1 * (seed % 4)
        mask = (np.abs(freq) < bandwidth).astype(float)
        phases = rng.uniform(0, 2 * np.pi, size=len(freq))
        spectrum = mask * np.exp(1j * phases)
        sig = np.fft.irfft(spectrum, n=N)
        x = np.sign(sig)
        x[x == 0] = 1
        inits.append(x)
    
    # Random inits
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        inits.append(rng.choice([-1, 1], size=N))
    
    # All-ones and alternating
    inits.append(np.ones(N))
    inits.append(np.ones(N) * np.array([1, -1] * (N // 2)))
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Greedy flip
        x, sc = _greedy_flip(init, max_iter=3000, deadline=deadline)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
        
        if time.time() > deadline:
            break
        
        # Local search on the result
        x2, sc2 = _local_search(x, deadline=deadline, max_iter=1000)
        if sc2 < best_score:
            best_score = sc2
            best_x = x2.copy()
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x

# EVOLVE-BLOCK-END
