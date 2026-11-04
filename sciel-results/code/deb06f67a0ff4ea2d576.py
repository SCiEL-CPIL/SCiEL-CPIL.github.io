"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def autoconv_peak(h):
        """Compute the peak (max absolute value) of the autoconvolution of h."""
        n = len(h)
        size = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, size)
        ac = np.fft.irfft(H * np.conj(H), size)[:2*n-1]
        return np.max(np.abs(ac))

    def thue_morse(n):
        """Generate Thue-Morse sequence mapped to +1/-1."""
        seq = np.zeros(n, dtype=np.float64)
        for i in range(n):
            # Count set bits
            count = bin(i).count('1')
            seq[i] = 1.0 if count % 2 == 0 else -1.0
        return seq

    # Initialize with Thue-Morse
    h = thue_morse(N)
    best_h = h.copy()
    best_peak = autoconv_peak(h)

    # Also try all-ones (the seed)
    h_ones = np.ones(N)
    peak_ones = autoconv_peak(h_ones)
    if peak_ones < best_peak:
        best_h = h_ones.copy()
        best_peak = peak_ones

    # Greedy local search with random restarts
    rng = np.random.RandomState(42)
    
    # Phase 1: Greedy local search from best initial
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        # Random order for diversity
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 30:
                break
            old_val = h[i]
            h[i] = -old_val
            peak = autoconv_peak(h)
            if peak < best_peak:
                best_peak = peak
                best_h = h.copy()
                improved = True
            else:
                h[i] = old_val

    # Phase 2: Random restarts with local search
    while time.time() < deadline - 5:
        # Random initialization
        h = rng.choice([-1.0, 1.0], size=N)
        peak = autoconv_peak(h)
        
        # Local search from this random start
        local_improved = True
        local_peak = peak
        local_h = h.copy()
        
        while local_improved and time.time() < deadline - 5:
            local_improved = False
            order = rng.permutation(N)
            for i in order:
                if time.time() > deadline - 5:
                    break
                old_val = local_h[i]
                local_h[i] = -old_val
                p = autoconv_peak(local_h)
                if p < local_peak:
                    local_peak = p
                    local_improved = True
                else:
                    local_h[i] = old_val
        
        if local_peak < best_peak:
            best_peak = local_peak
            best_h = local_h.copy()
            h = local_h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
