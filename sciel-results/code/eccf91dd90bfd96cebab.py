import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _quadratic_residue_sign(n, p):
    """
    Generate sign sequence based on quadratic residues mod p.
    For i in range(n): sign = 1 if i^2 is a quadratic residue mod p, else -1.
    This creates a sequence with good autocorrelation properties.
    """
    x = np.zeros(n)
    for i in range(n):
        # Check if i*i is a quadratic residue mod p (excluding 0)
        if i == 0:
            x[i] = 1.0
        else:
            # Use Euler's criterion: a^((p-1)/2) mod p is 1 if QR, -1 if non-QR
            val = pow(i * i % p, (p - 1) // 2, p)
            if val == 1:
                x[i] = 1.0
            elif val == p - 1:  # -1 mod p
                x[i] = -1.0
            else:
                x[i] = 1.0  # shouldn't happen for prime p
    return x

def _m_sequence(n, degree, poly):
    """
    Generate an m-sequence (maximal length LFSR sequence).
    degree: degree of the primitive polynomial
    poly: integer representation of the polynomial (e.g., 0b10011 for x^4+x+1)
    """
    x = np.zeros(n)
    # Initialize LFSR state
    state = 1  # non-zero initial state
    
    for i in range(n):
        x[i] = 1.0 if (state & 1) else -1.0
        # Shift right and compute feedback
        feedback = (state & 1)
        # XOR with polynomial taps
        for bit in range(degree - 1, -1, -1):
            if (poly >> bit) & 1:
                feedback ^= (state >> bit) & 1
        state = (state >> 1) | (feedback << (degree - 1))
        # Keep state in range
        state &= (1 << degree) - 1
    
    return x

def _barker_like(n):
    """
    Generate a Barker-like sequence using greedy construction.
    Try to minimize peak sidelobe of autocorrelation.
    """
    x = np.zeros(n)
    x[0] = 1.0
    for i in range(1, n):
        best_sign = 1.0
        best_max = float('inf')
        for sign in [1.0, -1.0]:
            x[i] = sign
            ac = _autoconv(x[:i+1])
            # Only consider sidelobes (exclude main lobe)
            max_sidelobe = np.max(np.abs(ac[1:])) if len(ac) > 1 else 0.0
            if max_sidelobe < best_max:
                best_max = max_sidelobe
                best_sign = sign
        x[i] = best_sign
    return x

def _hadamard_row(n):
    """
    Generate a row of a Hadamard matrix (for n being power of 2).
    Uses Sylvester construction.
    """
    if n <= 1:
        return np.array([1.0])
    
    # Find next power of 2 >= n
    h = 1
    while h < n:
        h *= 2
    
    # Generate Hadamard matrix of size h
    H = np.array([[1.0]])
    for _ in range(int(np.log2(h))):
        H = np.block([[H, H], [H, -H]])
    
    # Take the first n elements of the second row (or any non-DC row)
    return H[1, :n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s before 150s limit)

    # Baseline: All ones
    x_best = np.ones(N, dtype=np.float64)
    best_score = _score(x_best)
    
    candidates = []
    
    # 1. Quadratic residue sequences with various primes
    primes = [401, 409, 419, 421, 431, 433, 439, 443, 449, 457]
    for p in primes:
        if p > N:
            x = _quadratic_residue_sign(N, p)
            candidates.append(x)
    
    # 2. M-sequences with various degrees and polynomials
    m_seq_configs = [
        (8, 0b10011),    # x^8 + x^4 + x^3 + x^2 + 1
        (8, 0b100011),   # x^8 + x^5 + x^3 + x^2 + 1  
        (9, 0b1001011),  # x^9 + x^5 + x^4 + x^2 + 1
        (9, 0b1010101),  # x^9 + x^6 + x^4 + x^2 + 1
        (10, 0b10001101), # x^10 + x^6 + x^5 + x^3 + 1
        (10, 0b10100101), # x^10 + x^7 + x^4 + x^2 + 1
    ]
    for degree, poly in m_seq_configs:
        if (1 << degree) - 1 >= N:
            x = _m_sequence(N, degree, poly)
            candidates.append(x)
    
    # 3. Barker-like sequence (greedy construction)
    if time.time() < deadline - 60:
        x_barker = _barker_like(N)
        candidates.append(x_barker)
    
    # 4. Hadamard-based sequences
    x_had = _hadamard_row(N)
    candidates.append(x_had)
    
    # 5. Mixed: combine Hadamard with quadratic residue
    if len(candidates) >= 2:
        x_mixed = 0.5 * (x_had + candidates[0])
        # Normalize to ±1
        x_mixed = np.sign(x_mixed)
        x_mixed[x_mixed == 0] = 1.0
        candidates.append(x_mixed)
    
    # Evaluate all candidates
    for x in candidates:
        if time.time() > deadline - 40:
            break
        s = np.sum(x)
        if abs(s) < 1e-12:
            continue
        ac = _autoconv(x)
        max_ac = np.max(np.abs(ac))
        score = max_ac / (s * s)
        if score < best_score:
            best_score = score
            x_best = x.copy()
    
    # Local search: focus on positions that contribute to peak autoconvolution
    if time.time() < deadline - 15:
        x = x_best.copy()
        rng = np.random.default_rng(42)
        
        no_improve = 0
        max_no_improve = 100
        
        for _iter in range(8000):
            if time.time() > deadline - 5:
                break
            
            s = np.sum(x)
            if abs(s) < 1e-12:
                break
            ac = _autoconv(x)
            max_ac = np.max(np.abs(ac))
            cur_score = max_ac / (s * s)
            
            if cur_score < best_score:
                best_score = cur_score
                x_best = x.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            
            # Find the lag with maximum |autoconv| (excluding lag 0)
            ac_no_dc = ac.copy()
            ac_no_dc[0] = 0  # Exclude main lobe
            peak_lag = np.argmax(np.abs(ac_no_dc))
            
            # Positions that contribute to this peak: i and i+peak_lag
            # Try flipping positions that are heavily involved in the peak
            if peak_lag < N:
                # Sample from positions that contribute to the peak
                idx1 = rng.integers(N - peak_lag)
                idx2 = idx1 + peak_lag
                
                # Try flipping one of these positions
                if rng.random() < 0.5:
                    idx = idx1
                else:
                    idx = idx2
            else:
                idx = rng.integers(N)
            
            x_trial = x.copy()
            x_trial[idx] = -x_trial[idx]
            
            s_trial = np.sum(x_trial)
            if abs(s_trial) < 1e-12:
                continue
            ac_trial = _autoconv(x_trial)
            max_ac_trial = np.max(np.abs(ac_trial))
            trial_score = max_ac_trial / (s_trial * s_trial)
            
            if trial_score < cur_score:
                x = x_trial
            else:
                # Accept worse moves with decreasing probability
                accept_prob = 0.1 * (1.0 - _iter / 8000.0)
                if rng.random() < accept_prob:
                    x = x_trial
    
    # Final validation
    final_score = _score(x_best)
    if final_score < best_score:
        best_score = final_score
    
    return x_best
