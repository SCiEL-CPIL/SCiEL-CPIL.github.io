"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _autocorr_periodic(h):
    """Compute periodic autocorrelation of a ±1 sequence using FFT."""
    f = np.fft.fft(h.astype(float))
    power = f * np.conj(f)
    acorr = np.fft.ifft(power).real
    return acorr

def _autocorr_peak_periodic(h):
    """Compute max |autocorr| over non-zero lags for periodic autocorrelation."""
    acorr = _autocorr_periodic(h)
    # For periodic autocorrelation, acorr[0] = N, and acorr[k] for k=1..N-1
    # are the periodic autocorrelations at lags k
    if N > 1:
        return np.max(np.abs(acorr[1:]))
    else:
        return 0.0

def _generate_lfsr_sequence(n, seed=1):
    """Generate an LFSR m-sequence truncated/padded to length n."""
    # Use a 9-bit LFSR (maximal length 511) for good autocorrelation properties
    # Then truncate or repeat to get length n
    if n <= 511:
        # Generate 511-bit m-sequence
        lfsr = 1 << 8  # Start with 0b100000000 (9 bits, MSB set)
        poly = 0x199  # x^9 + x^5 + 1 (primitive polynomial for 9-bit LFSR)
        seq = []
        for _ in range(511):
            seq.append(1 if (lfsr & 1) else -1)
            # Shift right and apply feedback
            if lfsr & 1:
                lfsr = (lfsr >> 1) ^ poly
            else:
                lfsr >>= 1
        # Truncate to length n
        return np.array(seq[:n], dtype=np.float64)
    else:
        # For n > 511, generate multiple periods
        result = []
        while len(result) < n:
            lfsr = 1 << 8
            poly = 0x199
            for _ in range(511):
                if len(result) >= n:
                    break
                result.append(1 if (lfsr & 1) else -1)
                if lfsr & 1:
                    lfsr = (lfsr >> 1) ^ poly
                else:
                    lfsr >>= 1
        return np.array(result[:n], dtype=np.float64)

def _local_search(h, deadline):
    """Perform local search with periodic autocorrelation metric."""
    h = h.copy()
    score = _autocorr_peak_periodic(h)
    
    # Try flips in random order, multiple passes
    improved = True
    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        # Random permutation of positions
        perm = np.random.permutation(N)
        for i in perm:
            if time.time() > deadline:
                break
            h[i] = -h[i]
            new_score = _autocorr_peak_periodic(h)
            if new_score < score:
                score = new_score
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    
    return h, score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Generate LFSR-based initializations
    best_h = None
    best_score = float('inf')
    
    # Try different LFSR seeds/initializations
    for seed_offset in range(20):
        if time.time() > deadline:
            break
        
        # Generate LFSR sequence with slight variations
        h0 = _generate_lfsr_sequence(N)
        
        # Apply random sign flips to create diversity
        n_flips = np.random.randint(0, 10)
        for _ in range(n_flips):
            idx = np.random.randint(0, N)
            h0[idx] *= -1
        
        # Run local search
        h, score = _local_search(h0, deadline)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Also try a pure random initialization as fallback
    if time.time() < deadline:
        h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
        h, score = _local_search(h_rand, deadline)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
