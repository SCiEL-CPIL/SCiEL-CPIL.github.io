"""Signed step-function optimisation for the C3' autoconvolution objective."""
import time
import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize

N = 400
NFFT = 512  # next power of two >= 2N-1 = 799? Actually 2*400-1=799, so need 1024
NFFT = 1024


def _score(x):
    """Compute max |autoconv| / (integral)^2 for a signed step vector x of length N."""
    # Pad to NFFT for circular convolution to equal linear autoconvolution
    xp = np.zeros(NFFT)
    xp[:N] = x
    F = fft(xp)
    autoconv = np.real(ifft(F * F))
    # The autoconvolution (linear) is the first 2N-1 samples of the circular one
    # but for max |.| we can just take the whole thing (padding zeros beyond N)
    max_abs = np.max(np.abs(autoconv[:2 * N - 1]))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    return max_abs / (integral * integral)


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s limit

    # Start with a random signed vector, normalized
    np.random.seed(42)
    best_x = np.ones(N)
    best_val = _score(best_x)

    # Try several random restarts with Powell
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Random initial point: mixture of blocks
        if restarts % 3 == 1:
            # Random signs
            x0 = np.random.choice([-1.0, 1.0], size=N)
        elif restarts % 3 == 2:
            # Block structure: alternating blocks of +1 and -1 with random lengths
            x0 = np.zeros(N)
            i = 0
            sign = 1.0
            while i < N:
                length = np.random.randint(5, 40)
                x0[i:i + length] = sign
                sign *= -1
                i += length
        else:
            # Smooth random with low-frequency bias
            freqs = np.random.randn(N) * 0.3
            x0 = np.tanh(freqs)

        # Normalize to zero mean? Not required, but helps avoid integral ~ 0
        # Actually we want to minimize max|autoconv|/(integral)^2, so scaling doesn't matter
        # (both numerator and denominator scale as x^2). So normalization is irrelevant for score.
        # But to avoid numerical issues with integral near 0, ensure |sum| > threshold.
        s = np.sum(x0)
        if abs(s) < 1.0:
            x0 = x0 + np.sign(s if s != 0 else 1.0) * (1.0 - abs(s)) / N

        # Powell minimization on the unnormalized vector (scale-invariant objective)
        def objective(x):
            # Re-normalize to keep scale fixed; actually objective is scale-invariant
            # but Powell may drift in scale. Pin scale by normalizing.
            s = np.sum(x)
            if abs(s) < 1e-10:
                return 1e12
            x_norm = x / s  # now sum = 1
            return _score(x_norm)

        result = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'xtol': 1e-4, 'ftol': 1e-4})

        x_candidate = result.x
        s = np.sum(x_candidate)
        if abs(s) < 1e-10:
            continue
        x_candidate = x_candidate / s
        val = _score(x_candidate)
        if val < best_val:
            best_val = val
            best_x = x_candidate.copy()

        # Early stop if we've hit a very good value
        if best_val < 0.01:
            break

        # Check deadline
        if time.time() > deadline - 5.0:
            break

    return best_x
