"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_m_sequence(degree, poly, length):
    """
    Generate an m-sequence using a Linear Feedback Shift Register (LFSR).
    poly: integer representation of the primitive polynomial (excluding x^degree).
    Example: x^9 + x^4 + 1 is represented as (1 << 4) | 1 = 17.
    """
    # Initialize register with all 1s (non-zero state)
    register = (1 << degree) - 1
    seq = []
    
    for _ in range(length):
        # Output bit (LSB)
        bit = register & 1
        seq.append(1 if bit else -1)
        
        # Feedback bit (LSB before shift)
        feedback = register & 1
        
        # Shift right
        register >>= 1
        
        # If feedback is 1, XOR with polynomial taps
        if feedback:
            register ^= poly
            
        # Ensure register stays within degree bits
        # The XOR with poly (which has bits < degree) and shift right keeps it bounded
        
    return np.array(seq, dtype=np.float64)

def _autocorr(h):
    """Compute autocorrelation r[k] = sum_i h[i]*h[i+k] for k=0..N-1."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    return r[:N]

def _score(h):
    """Compute the upper bound on C3 (peak sidelobe metric)."""
    r = _autocorr(h)
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # 1. Try M-sequences as strong initial candidates
    # Degree 9: 511 length. Primitive polynomials: x^9+x^4+1 (17), x^9+x^5+1 (33), etc.
    # We try a few known primitive polynomials for degree 9.
    primitive_polys_deg9 = [
        (1 << 4) | 1,      # x^9 + x^4 + 1
        (1 << 5) | 1,      # x^9 + x^5 + 1
        (1 << 4) | (1 << 1) | 1, # x^9 + x^4 + x + 1
        (1 << 5) | (1 << 2) | 1, # x^9 + x^5 + x^2 + 1
    ]
    
    for poly in primitive_polys_deg9:
        # Generate slightly longer than N to allow windowing/phase shifting
        m_seq = _generate_m_sequence(9, poly, N + 10)
        
        # Try different phases (offsets)
        for offset in range(10):
            h_candidate = m_seq[offset:offset+N]
            s = _score(h_candidate)
            if s < best_score:
                best_score = s
                best_h = h_candidate.copy()
                
        # Also try inverted sequence
        m_seq_inv = -m_seq
        for offset in range(10):
            h_candidate = m_seq_inv[offset:offset+N]
            s = _score(h_candidate)
            if s < best_score:
                best_score = s
                best_h = h_candidate.copy()

    # If no M-sequence worked well, fall back to random
    if best_h is None:
        h_candidate = np.random.choice([-1, 1], N).astype(np.float64)
        best_score = _score(h_candidate)
        best_h = h_candidate.copy()

    # 2. Local Search Optimization
    # Use a greedy search with occasional random restarts if stuck
    current_h = best_h.copy()
    current_score = best_score
    n_no_improve = 0
    max_no_improve = 2000 # Increased patience
    
    # Pre-allocate for speed? No, numpy handles it.
    
    while time.time() < deadline - 2.0:
        if n_no_improve > max_no_improve:
            # Restart from a random point or slightly perturbed best
            # Try a random perturbation of the best solution
            current_h = best_h.copy()
            # Flip a few random bits
            for _ in range(10):
                idx = np.random.randint(N)
                current_h[idx] *= -1
            current_score = _score(current_h)
            n_no_improve = 0
            continue
            
        # Try flipping a random bit
        idx = np.random.randint(N)
        old_val = current_h[idx]
        current_h[idx] = -old_val
        
        new_score = _score(current_h)
        
        if new_score < current_score:
            current_score = new_score
            n_no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            current_h[idx] = old_val
            n_no_improve += 1

    return best_h
# EVOLVE-BLOCK-END
