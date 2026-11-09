"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _autoconvolution(s):
    """Compute the autoconvolution of a signed step function with N intervals.
    
    The autoconvolution c[k] = sum_i s[i] * s[i+k] for k in [-(N-1), N-1].
    We use FFT for efficiency.
    """
    # Zero-pad to length >= 2*N-1 for linear convolution
    n_fft = 1 << (2 * N - 1).bit_length()
    f_s = np.fft.rfft(s, n=n_fft)
    # Autoconvolution is the inverse FFT of |F|^2, but we need the full convolution
    # Actually, autoconvolution c[k] = sum_i s[i]*s[i+k] is the cross-correlation,
    # which equals the convolution of s with reversed s.
    # Alternatively: c[k] = (s * s_flipped)[-N+1+k] where * is linear conv
    # Simpler: use FFT. The autoconvolution (lag k) is sum_i s[i]*s[i+k].
    # This is the same as the aperiodic autocorrelation.
    # c[k] for k=0..N-1: sum_{i=0}^{N-1-k} s[i]*s[i+k]
    # For negative k, by symmetry c[-k] = c[k] if we define it properly? No.
    # Let's just compute it directly via FFT.
    # The autocorrelation R[k] = sum_i s[i+k]*s[i] = conv(s, s_reversed)
    s_rev = s[::-1]
    f_s_rev = np.fft.rfft(s_rev, n=n_fft)
    conv = np.fft.irfft(f_s * f_s_rev, n=n_fft)
    # The convolution of s and s_rev has length 2N-1.
    # R[k] for lag k (where k ranges from -(N-1) to N-1)
    # conv[i] = sum_j s[j] * s_rev[i-j] = sum_j s[j] * s[N-1-(i-j)] = sum_j s[j]*s[N-1-i+j]
    # Let m = i - (N-1). Then conv[N-1+m] = sum_j s[j]*s[m+j] = R[m]
    # So R[m] = conv[N-1+m] for m in [-(N-1), N-1]
    # We want the sum of squares of all lags, or some upper bound on C3.
    # The task says "upper bound on C3". C3 is likely the third moment or a specific constant.
    # In the context of "Third autocorrelation inequality", C3 might be related to
    # the maximum of the autocorrelation or the L2 norm squared.
    # Given the seed program returns np.ones(N), and the score is seed/value,
    # let's assume the score is the value of some functional.
    # If s = ones, autoconv is a triangle peak.
    # Let's compute the sum of squares of the autocorrelation coefficients.
    # Or perhaps the max absolute value of the autocorrelation?
    # Or the L-infinity norm?
    # "improve an autoconvolution upper bound" -> minimize the upper bound.
    # A common bound is the L2 norm of the autocorrelation, or the peak sidelobe.
    # Let's try the sum of squares of the non-zero lags (excluding lag 0).
    # Or the max of |R[k]| for k != 0.
    # Let's look at the "Third autocorrelation inequality".
    # This might refer to a specific result in additive combinatorics or signal processing.
    # Without the paper, I'll guess the objective is to minimize the L2 norm of the
    # autocorrelation excluding the peak, or the peak sidelobe level.
    # Let's try minimizing the sum of squares of all lags except 0.
    # R[0] = N (for ones).
    # Let's compute the full autocorrelation vector.
    R = np.zeros(2*N - 1)
    R[N-1:N-1+N] = conv[N-1:2*N-1]
    # R is indexed from 0 to 2N-2, corresponding to lags -(N-1) to N-1.
    # R[N-1] is lag 0.
    # Let's return the sum of squares of R excluding the center.
    # Or maybe just the max?
    # Let's try the L2 norm squared of the sidelobes.
    sidelobes = np.concatenate([R[:N-1], R[N:]])
    return np.sum(sidelobes**2)


def _c3_bound(s):
    """Compute the C3 upper bound for a given signed step function."""
    # The task mentions "Third autocorrelation inequality".
    # Let's assume C3 is related to the autocorrelation.
    # A common quantity is the normalized autocorrelation.
    # Let's try the max absolute value of the autocorrelation sidelobes.
    # Or the sum of squares.
    # Let's use the sum of squares of the sidelobes as the objective to minimize.
    return _autoconvolution(s)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initial guess: Thue-Morse sequence
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    def thue_morse(n):
        # Compute Thue-Morse sequence of length n
        tm = np.zeros(n, dtype=np.int8)
        tm[0] = 1
        for i in range(1, n):
            # t(i) = t(i // 2) * (-1)^(i % 2) ? No.
            # t(2k) = t(k), t(2k+1) = 1 - t(k)
            # Let's build it iteratively
            pass
        # Simpler: use the property that t(n) = sum of binary digits mod 2
        tm = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = 0
            x = i
            while x:
                cnt += x & 1
                x >>= 1
            tm[i] = 1 if cnt % 2 == 0 else -1
        return tm
    
    s_init = thue_morse(N).astype(np.float64)
    
    # Try a few different initializations and optimize each
    best_s = s_init
    best_score = _c3_bound(s_init)
    
    # Also try a sinusoidal initialization
    t = np.linspace(0, 4*np.pi, N)
    s_sine = np.sign(np.sin(t))
    if np.all(s_sine == 0):
        s_sine[0] = 1
    score_sine = _c3_bound(s_sine)
    if score_sine < best_score:
        best_score = score_sine
        best_s = s_sine
    
    # Try a random initialization
    np.random.seed(42)
    s_rand = np.random.choice([-1, 1], N)
    score_rand = _c3_bound(s_rand)
    if score_rand < best_score:
        best_score = score_rand
        best_s = s_rand
    
    # Use L-BFGS to optimize the continuous version, then threshold
    def objective_continuous(x):
        if time.time() > deadline:
            return np.inf
        # x is a continuous vector, we threshold it to get signs
        s = np.sign(x)
        # Handle zeros: if x[i] == 0, keep previous sign or default to 1
        # For smoothness, use a soft sign or just threshold
        s = np.where(s == 0, 1, s)
        return _c3_bound(s)
    
    # We can't use L-BFGS directly on a discrete objective.
    # Instead, let's use a local search: flip individual bits and check improvement.
    
    # Local search: try flipping each bit
    s_current = best_s.copy()
    score_current = best_score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            s_trial = s_current.copy()
            s_trial[i] *= -1
            score_trial = _c3_bound(s_trial)
            if score_trial < score_current - 1e-10:
                s_current = s_trial
                score_current = score_trial
                improved = True
    
    # Try a few random restarts with local search
    for restart in range(5):
        if time.time() > deadline:
            break
        np.random.seed(1000 + restart)
        s_trial = np.random.choice([-1, 1], N)
        score_trial = _c3_bound(s_trial)
        
        # Local search on this random start
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                s_flip = s_trial.copy()
                s_flip[i] *= -1
                score_flip = _c3_bound(s_flip)
                if score_flip < score_trial - 1e-10:
                    s_trial = s_flip
                    score_trial = score_flip
                    improved = True
        
        if score_trial < best_score:
            best_score = score_trial
            best_s = s_trial
    
    return best_s
# EVOLVE-BLOCK-END
