"""Improved program: discrete simulated annealing on signed step functions."""
N = 400
import numpy as np
import time

def compute_autocorrelation_bound(f):
    """Compute an upper bound on C3 based on autocorrelation sidelobes."""
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N]
    if abs(acf[0]) > 1e-12:
        sidelobe_energy = np.sum(acf[1:] ** 2)
        peak_energy = acf[0] ** 2
        bound = np.sqrt(sidelobe_energy / peak_energy)
    else:
        bound = np.sqrt(np.sum(acf[1:] ** 2))
    return bound

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    def lfsr_sequence(degree, poly_taps, length):
        state = [1] * degree
        seq = []
        for _ in range(length):
            seq.append(state[-1])
            fb = 0
            for tap in poly_taps:
                if tap == degree:
                    continue
                if tap == 0:
                    fb ^= state[0]
                else:
                    fb ^= state[tap - 1]
            state = state[:-1] + [fb]
        return seq

    # Generate candidate m-sequences
    candidates = []
    
    # Poly 1: x^9 + x^4 + 1
    taps1 = [9, 4, 0]
    seq1 = lfsr_sequence(9, taps1, 511)
    f1 = np.array([1 if b == 1 else -1 for b in seq1[:N]])
    candidates.append(f1)
    
    # Poly 2: x^9 + x^5 + x^3 + x^2 + 1
    taps2 = [9, 5, 3, 2, 0]
    seq2 = lfsr_sequence(9, taps2, 511)
    f2 = np.array([1 if b == 1 else -1 for b in seq2[:N]])
    candidates.append(f2)
    
    # Poly 3: x^9 + x^5 + x^4 + x^2 + 1
    taps3 = [9, 5, 4, 2, 0]
    seq3 = lfsr_sequence(9, taps3, 511)
    f3 = np.array([1 if b == 1 else -1 for b in seq3[:N]])
    candidates.append(f3)
    
    # Poly 4: x^9 + x^4 + x^3 + x^2 + 1
    taps4 = [9, 4, 3, 2, 0]
    seq4 = lfsr_sequence(9, taps4, 511)
    f4 = np.array([1 if b == 1 else -1 for b in seq4[:N]])
    candidates.append(f4)
    
    # Add some phase shifts and sign flips for diversity
    for f in [f1, f2, f3, f4]:
        candidates.append(np.roll(f, N // 2))
        candidates.append(-f)
        candidates.append(-np.roll(f, N // 2))
    
    # Evaluate initial candidates
    best_f = None
    best_bound = float('inf')
    
    for f in candidates:
        bound = compute_autocorrelation_bound(f)
        if bound < best_bound:
            best_bound = bound
            best_f = f.copy()
    
    # Local search: bit flipping
    f = best_f
    bound = best_bound
    n_flips = 0
    
    while time.time() < deadline:
        # Randomly flip a bit
        idx = np.random.randint(0, N)
        old_val = f[idx]
        f[idx] = -f[idx]
        
        new_bound = compute_autocorrelation_bound(f)
        
        if new_bound < bound:
            bound = new_bound
            n_flips += 1
        else:
            f[idx] = old_val
    
    return f
