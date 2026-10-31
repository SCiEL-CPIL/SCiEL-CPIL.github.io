"""Signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def score(x):
        """Compute max |autocorr| / (sum)^2. Lower is better."""
        S = np.sum(x)
        if abs(S) < 1e-12:
            return float('inf')
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        X = np.fft.rfft(x, size)
        acf = np.fft.irfft(X * np.conj(X), size)[:n]
        max_abs = np.max(np.abs(acf))
        return max_abs / (S * S)

    best_x = np.ones(N)
    best_score = score(best_x)
    
    # Structural Change: Use LFSR sequences
    # LFSR sequences (m-sequences) have ideal periodic autocorrelation.
    # We map binary {0,1} to signed {-1,1}.
    
    def generate_lfsr(d, taps, init_state, length):
        """
        Generate an LFSR sequence.
        d: degree
        taps: bitmask of feedback taps
        init_state: integer representing initial state
        length: number of bits to generate
        """
        if d == 0:
            return np.zeros(length)
        
        if init_state == 0:
            init_state = 1
            
        state = init_state & ((1 << d) - 1)
        seq = np.zeros(length)
        
        for i in range(length):
            bit = state & 1
            seq[i] = 1.0 if bit else -1.0
            
            fb = 0
            for t in range(d):
                if (taps >> t) & 1:
                    fb ^= (state >> t) & 1
            
            state = (state >> 1) | (fb << (d - 1))
            state &= (1 << d) - 1
            
        return seq

    rng = np.random.RandomState(42)
    
    degrees_to_try = [9, 10, 11, 12]
    
    for d in degrees_to_try:
        if time.time() > deadline - 5:
            break
            
        max_period = (1 << d) - 1
        if max_period < N:
            continue
            
        found_primitive = 0
        attempts = 0
        max_attempts = 20
        
        while found_primitive < 5 and attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            taps = rng.randint(1, 1 << (d-1))
            
            # Check if it's primitive
            init_state = 1
            state = init_state
            is_primitive = True
            for _ in range(max_period):
                fb = 0
                for t in range(d):
                    if (taps >> t) & 1:
                        fb ^= (state >> t) & 1
                state = (state >> 1) | (fb << (d - 1))
                state &= (1 << d) - 1
                if state == 0:
                    is_primitive = False
                    break
            
            if is_primitive and state == init_state:
                found_primitive += 1
                
                for phase in range(10):
                    if time.time() > deadline - 5:
                        break
                    
                    init_state = 1 + phase
                    seq = generate_lfsr(d, taps, init_state, N)
                    s = score(seq)
                    
                    if s < best_score:
                        best_score = s
                        best_x = seq.copy()
                        
    # Fallback: If LFSR didn't find anything better, do a quick random search
    if time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            x = rng.choice([-1, 1], N)
            s = score(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()

    return best_x
