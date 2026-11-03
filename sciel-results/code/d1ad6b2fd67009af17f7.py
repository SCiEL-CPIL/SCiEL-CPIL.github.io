"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    rng = random.Random(42)
    
    # Strategy: Spectral Convex Optimization
    # 1. We reformulate the problem in the frequency domain.
    #    Let H[k] be the DFT of h. The autoconvolution is the inverse DFT of H^2.
    #    Minimizing the max absolute value of the time-domain signal is equivalent to
    #    minimizing the peak of the inverse DFT of |H|^2 (since h is real, H has symmetry).
    #    Actually, for real h, ac[n] = IDFT(H^2)[n].
    #    The magnitude of H is determined by h.
    #    However, a more direct convex approach:
    #    Minimize t such that |IDFT(H^2)[n]| <= t for all n.
    #    This is not convex in H directly because of the H^2 term.
    #    
    #    Alternative: Minimize the L2 norm of the autoconvolution? No, we need max.
    #    
    #    Let's use a smooth surrogate for the max absolute value:
    #    f(h) = log(sum(exp(beta * |ac[n]|^2))) / beta
    #    This is a smooth approximation to max |ac[n]|.
    #    Since ac[n] is linear in h? No, ac is quadratic in h.
    #    So f(h) is a quartic function. It is not convex.
    #
    #    However, if we fix the sum S, and we work in the subspace where sum(h)=S,
    #    we can try to minimize the spectral flatness or similar metrics.
    #
    #    Let's try a different angle: Randomized Spectral Shaping.
    #    Generate random phases for the DFT, set magnitudes to a target shape (e.g., flat or tapered),
    #    inverse FFT, project to signs, and refine.
    #    This is essentially the "Gaussian random field" approach with spectral shaping.
    
    # Precompute twiddles for fast updates if needed, but let's try a cleaner spectral init first.
    
    # Phase 1: Spectral Initialization
    # We want to minimize the peak of the autoconvolution.
    # The autoconvolution is the square of the spectrum in frequency domain (squared magnitude? No, H^2).
    # For real signals, H[k] = conj(H[-k]).
    # ac[n] = sum_k H[k]^2 e^{i 2 pi k n / fsize}.
    # To minimize the peak of ac, we want H[k]^2 to be as "flat" as possible in frequency?
    # No, if H[k]^2 is flat, ac[n] is a delta function (if H is delta).
    # If H is constant, ac is a delta.
    # We want ac to be as small as possible everywhere except maybe n=0?
    # Actually, ac[0] = sum h^2 = N. This is fixed for binary signs.
    # So we cannot make ac[0] small. The max is at least N.
    # We want the other peaks to be small compared to N.
    # Ideally, ac[n] = 0 for n != 0. This is a Barker sequence (or similar).
    # Barker sequences only exist for small N. For N=400, we want "almost" Barker.
    # This means we want the aperiodic autocorrelation to be low.
    # This is equivalent to minimizing the sidelobes of the spectrum |H|^2?
    # No, the autocorrelation is the IFFT of |H|^2.
    # So we want |H|^2 to be as flat as possible?
    # If |H|^2 is flat, then ac is a delta.
    # So the goal is to find a binary sequence whose power spectrum is as flat as possible.
    
    # Algorithm:
    # 1. Start with a random binary sequence.
    # 2. Compute its spectrum |H|^2.
    # 3. Modify the phases of H to make |H|^2 flatter?
    #    We can't change |H| without changing h.
    #    But we can use an iterative algorithm:
    #    a. Compute H.
    #    b. Compute target magnitude |H_target| such that IFFT(|H_target|^2) is a delta (or close to it).
    #       Actually, if we want ac to be delta, |H|^2 must be constant.
    #       So set |H_target| = sqrt(N) (constant).
    #    c. Create H_new = |H_target| * exp(i * angle(H)).
    #    d. h_new = IFFT(H_new).
    #    e. Project h_new to signs: h_sign = sign(h_new).
    #    f. Repeat.
    # This is the "Gershgorin" or "Gerchberg-Saxton" type algorithm adapted for autocorrelation.
    
    def _spectral_shaping_init(seed):
        rng_local = random.Random(seed)
        h = np.array([1.0 if rng_local.random() > 0.5 else -1.0 for _ in range(N)])
        # Ensure sum is not zero
        if np.sum(h) == 0:
            h[0] = 1.0
            
        for it in range(500):
            if time.time() > deadline - 20:
                break
            H = np.fft.rfft(h, fsize)
            
            # Current Power Spectrum
            P = np.abs(H)**2
            
            # Target: We want the autocorrelation (IFFT of P) to be a delta at 0.
            # This implies P should be constant.
            # However, we have a constraint on the mean (DC component).
            # H[0] = sum(h). Let S = sum(h).
            # We want to minimize the side lobes of ac.
            # Let's aim for a flat spectrum except at DC.
            # Target P[k] = C for k > 0, and P[0] = S^2.
            # To keep the total energy consistent? No, h is binary, energy is N.
            # Parseval: sum |h|^2 = N = (1/fsize) sum |H|^2.
            # So sum |H|^2 = N * fsize.
            # If we set P[k] = C for k>0, then S^2 + C * (fsize/2) = N * fsize.
            # C = (N * fsize - S^2) / (fsize/2) = 2N - 2*S^2/fsize.
            
            S = np.sum(h)
            P_target = np.zeros_like(P)
            P_target[0] = S * S
            if fsize > 1:
                C = (N * fsize - S * S) / (fsize // 2)
                if C < 0: C = 0
                P_target[1:] = C
                
            # Update H phases to match target magnitude
            # H_new[k] = sqrt(P_target[k]) * exp(i * angle(H[k]))
            # Handle k=0 carefully (real and positive)
            angles = np.angle(H)
            H_new = np.sqrt(P_target) * np.exp(1j * angles)
            H_new[0] = S # Keep DC fixed to maintain sum
            
            # Inverse FFT
            h_cont = np.fft.irfft(H_new, fsize)
            h_cont = h_cont[:N]
            
            # Project to signs
            h_new = np.sign(h_cont)
            # Handle zeros
            zeros = h_new == 0
            if np.any(zeros):
                idxs = np.where(zeros)[0]
                for i in idxs:
                    h_new[i] = 1.0 if rng_local.random() > 0.5 else -1.0
                    
            # Check improvement
            score_new = _score(h_new)
            if score_new < best_val:
                best_val = score_new
                best_h = h_new.copy()
                
            # Check convergence
            if np.all(h_new == h):
                break
            h = h_new
            
        return h, best_val, best_h

    # Run spectral shaping with a few different seeds
    for seed in [42, 123, 999, 555]:
        if time.time() > deadline - 30:
            break
        h_init, val, bh = _spectral_shaping_init(seed)
        
        # Refine the best candidate from this seed with local search
        h_local = bh.copy()
        H_local = np.fft.rfft(h_local, fsize)
        k_indices = np.arange(fsize // 2 + 1)
        twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
        
        current_val = _score(h_local)
        
        # Greedy Hill Climb
        improved = True
        passes = 0
        while improved and time.time() < deadline - 10:
            improved = False
            passes += 1
            indices = list(range(N))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 10:
                    break
                old_val = h_local[i]
                new_val = -old_val
                h_local[i] = new_val
                delta = -2 * old_val
                H_local += delta * twiddles[i]
                
                ac = np.fft.irfft(H_local * H_local, fsize)
                new_ac_max = np.max(np.abs(ac[:size]))
                s = np.sum(h_local)
                if abs(s) < 1e-9:
                    new_score = 1e10
                else:
                    new_score = new_ac_max / (s * s)
                
                if new_score < current_val:
                    current_val = new_score
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h_local.copy()
                    improved = True
                else:
                    h_local[i] = old_val
                    H_local -= delta * twiddles[i]

    # Final Polish with Simulated Annealing if time permits
    if time.time() < deadline - 5:
        h = best_h.copy()
        H = np.fft.rfft(h, fsize)
        k_indices = np.arange(fsize // 2 + 1)
        twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
        
        current_val = _score(h)
        T = 0.1
        steps = 5000
        for _ in range(steps):
            if time.time() > deadline - 2:
                break
            i = rng.randint(0, N - 1)
            old_val = h[i]
            new_val = -old_val
            h[i] = new_val
            delta = -2 * old_val
            H += delta * twiddles[i]
            
            ac = np.fft.irfft(H * H, fsize)
            new_ac_max = np.max(np.abs(ac[:size]))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = new_ac_max / (s * s)
            
            delta_score = new_score - current_val
            if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
                current_val = new_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] = old_val
                H -= delta * twiddles[i]
            T *= 0.999

    return best_h

# EVOLVE-BLOCK-END
