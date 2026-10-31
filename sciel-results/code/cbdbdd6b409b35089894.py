"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Returns a signed step function (values in {-1, 1}) that minimizes
    the maximum absolute autoconvolution divided by squared integral.
    
    Strategy: Use a greedy sign-flipping heuristic on a binary sequence.
    We start with a random or structured initial pattern and iteratively
    flip signs at positions that most reduce the peak autoconvolution
    magnitude. Since the integral is fixed at N (for unit steps, integral
    is sum of values, but wait - the task says 'squared integral').
    
    Let's re-read: "maximum absolute autoconvolution divided by squared integral"
    For a function f, autoconvolution c(k) = sum f(t) f(t+k).
    Integral I = sum f(t).
    We want to minimize max_k |c(k)| / I^2.
    
    If f(t) in {-1, 1}, then I = sum f(t). If we have roughly equal +1 and -1, I is small,
    which makes the denominator small, potentially increasing the ratio.
    However, if I=0, the ratio is undefined or infinite.
    
    The constant function f(t)=1 has I=N, I^2=N^2.
    c(0) = N. c(k>0) = N-k.
    max |c(k)| = N.
    Ratio = N / N^2 = 1/N.
    
    If we use a balanced sequence (+1 and -1 equally), I is small.
    But max |c(k)| might be smaller.
    For example, a random balanced sequence:
    I ~ sqrt(N) (random walk). I^2 ~ N.
    c(0) = N.
    So ratio ~ N/N = 1. This is worse than 1/N (for N=400, 1 vs 0.0025).
    
    So we want a large integral (many same signs) but low autoconvolution peaks.
    This is a trade-off.
    
    The constant function is a strong baseline. Can we do better?
    The theoretical lower bound for max |c(k)| is related to the L2 norm.
    By Parseval, sum |c(k)|^2 = (sum |f(t)|^2)^2 = N^2.
    If we spread the energy uniformly over 2N-1 lags, each c(k)^2 ~ N^2 / (2N) = N/2.
    So |c(k)| ~ sqrt(N/2).
    If I ~ N (mostly +1s), ratio ~ sqrt(N/2) / N^2 = 1 / (sqrt(2) * N^1.5).
    This is much smaller than 1/N for large N.
    
    So the goal is to find a sequence with large integral (dominated by one sign)
    but "random-like" autoconvolution (low peaks).
    
    A sequence of all +1s has high autoconvolution peaks.
    A sequence with a few -1s inserted to break the periodicity might help.
    
    Let's try a heuristic:
    1. Start with all +1s.
    2. Iteratively flip a +1 to -1 if it reduces the max |c(k)| sufficiently
       while not reducing I^2 too much.
       
    Alternatively, use a known low-peak sequence, like a Barker sequence or a Legendre sequence,
    but those are balanced (I=0 or small).
    
    Let's look at the "sparse signed sequences" hint.
    "Sparse signed sequences with local optimization exploit the probabilistic decay of autoconvolution peaks in sparse structures... Measured: 1.051 -> 1.125."
    This suggests that starting from something and improving yields better scores.
    
    Let's implement a local search:
    1. Initialize f with all 1s.
    2. Compute c = np.convolve(f, f[::-1], 'same') or similar.
    3. Find the lag k with max |c(k)|.
    4. Try flipping signs at positions that contribute to this peak.
    5. Accept if the new max |c(k)| / I^2 is lower.
    
    To be efficient, we can use FFT for autoconvolution.
    
    Deadline guard:
    Limit the number of iterations or time spent.
    """
    start_time = time.time()
    deadline = 110  # 110 seconds to be safe
    
    # Initial state: all +1
    f = np.ones(N, dtype=np.int8)
    
    # Helper to compute score
    def compute_score(f):
        I = np.sum(f)
        if I == 0:
            return np.inf
        # Autoconvolution via FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(f, n=n_fft)
        C = np.fft.ifft(F * F.conj()).real
        # The autoconvolution is centered. 
        # Standard convolution c(k) = sum f(t) f(t+k).
        # FFT gives circular convolution. 
        # We need the linear autoconvolution.
        # c_full = np.fft.ifft(np.fft.fft(f, n_fft) * np.fft.fft(f, n_fft).conj()).real
        # This gives correlation.
        # Let's stick to a simple loop for small N or use scipy if allowed? 
        # SciPy is available.
        from scipy.signal import fftconvolve
        c = fftconvolve(f, f[::-1], mode='full')
        # c is length 2N-1. c[0] = f[0]*f[0]... no, c[k] = sum f[i] f[i+k]?
        # fftconvolve(f, f[::-1]) computes sum f[i] f_rev[j] where i+j=k.
        # This is correlation.
        # Let's just use np.convolve for simplicity if N=400 is small enough?
        # 400*400 = 160,000 ops. Fast enough in Python? No, maybe slow.
        # FFT is better.
        
        # Correct FFT autoconvolution:
        # c = ifft(fft(f) * conj(fft(f))) gives circular autocorrelation.
        # For linear, pad with zeros.
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(f, n=n_fft)
        C = np.fft.irfft(F * F.conj(), n=n_fft)
        # C[k] = sum_{t} f[t] f[t+k] (with wrap-around handled by padding)
        # The linear autocorrelation is in C[0:N] and C[N-1:0:-1]?
        # Actually, C[k] for k=0..N-1 is the positive lags.
        # C[N-1] is lag -(N-1).
        # So max |C| over all k.
        max_abs_c = np.max(np.abs(C[:N]))  # Check positive lags
        # Wait, the negative lags are symmetric for real f.
        # So checking 0..N-1 is enough.
        
        return max_abs_c / (I * I)
    
    current_score = compute_score(f)
    best_f = f.copy()
    best_score = current_score
    
    # Simple local search
    # Try flipping random indices
    rng = np.random.default_rng(42)
    
    max_iter = 20000  # Adjust based on time
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random index to flip
        idx = rng.integers(0, N)
        
        # Flip
        f[idx] *= -1
        
        # Compute new score
        new_score = compute_score(f)
        
        # Accept if better
        if new_score < best_score:
            best_score = new_score
            best_f = f.copy()
        else:
            # Revert
            f[idx] *= -1
            
        # Occasionally try a multi-flip or different heuristic?
        # For now, single flips.
        
        # If we get stuck, maybe try a larger step?
        # Let's stick to single flips for robustness.
        
    return best_f

# EVOLVE-BLOCK-END
