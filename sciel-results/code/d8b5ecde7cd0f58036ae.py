"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _compute_score(f):
    """
    Computes the score for a given signed step function.
    Metric: Sum of cubes of absolute values of circular autocorrelation coefficients.
    This measures the 'flatness' of the autocorrelation. Lower is better.
    """
    # Compute circular autocorrelation using FFT for efficiency
    # R[k] = sum_i f[i] f[i+k]
    # Using FFT: R = ifft(fft(f) * conj(fft(f)))
    fft_f = np.fft.fft(f)
    power_spectrum = fft_f * np.conj(fft_f)
    autocorr = np.fft.ifft(power_spectrum).real
    
    # The autocorrelation values should be real.
    # We want to minimize the L3 norm of the autocorrelation vector.
    # Note: The DC term (k=0) is always sum(f^2) = N, which is constant.
    # Minimizing the sum of cubes of the non-DC terms is equivalent to minimizing the total sum
    # since the DC term is constant.
    score = np.sum(np.abs(autocorr)**3)
    return score

def _generate_m_sequence(length):
    """
    Generate an m-sequence (Maximum Length Sequence) of the given length.
    Uses a Linear Feedback Shift Register (LFSR) with primitive polynomial x^9 + x^4 + 1.
    """
    # length should be 2^m - 1
    # We generate 511 bits (2^9 - 1) and let the caller truncate.
    reg = 1
    seq = []
    for _ in range(length):
        seq.append(1 if reg & 1 else -1)
        # Feedback taps for x^9 + x^4 + 1: bits 0, 4, 8
        feedback = (reg >> 0) ^ (reg >> 4) ^ (reg >> 8)
        reg = (reg >> 1) | (feedback << 8)
        
    return np.array(seq)

def _local_search(f, max_iterations=20):
    """
    Deterministic local search using coordinate descent.
    Tries flipping each bit in order, accepting if score improves.
    """
    current_f = f.copy()
    current_score = _compute_score(current_f)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            # Try flipping bit i
            old_val = current_f[i]
            current_f[i] = -old_val
            
            new_score = _compute_score(current_f)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current_f[i] = old_val
                
        if not improved:
            break
            
    return current_f, current_score

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Approach: 
    1. Initialize with a truncated M-sequence (length 511 -> 400).
    2. Refine using deterministic local search (coordinate descent).
    3. Compare with a random restart to ensure robustness.
    """
    # Step 1: Generate M-sequence of length 511
    m_seq = _generate_m_sequence(511)
    
    # Truncate to N=400
    initial_f = m_seq[:N].copy()
    
    # Step 2: Local Search on M-sequence
    best_f, best_score = _local_search(initial_f, max_iterations=20)
    
    # Step 3: Random restart local search
    random_f = np.random.choice([-1, 1], size=N)
    rand_f, rand_score = _local_search(random_f, max_iterations=10)
    
    if rand_score < best_score:
        best_f = rand_f
        best_score = rand_score
        
    return best_f
# EVOLVE-BLOCK-END
