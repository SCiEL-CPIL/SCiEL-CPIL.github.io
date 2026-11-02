import numpy as np
import time

N = 100

def _compute_overlap_fft(bits):
    """
    Compute the maximum overlap using FFT.
    bits: binary array of 0s and 1s, length N, sum N/2.
    Returns: max overlap value.
    """
    # u = bits - 0.5
    u = bits.astype(np.float64) - 0.5
    
    # Autocorrelation via FFT
    U = np.fft.fft(u)
    power = np.abs(U) ** 2
    C = np.fft.ifft(power).real
    
    # C[0] is the DC term (sum of squares). We care about k=1..N-1.
    # Overlap O_k = N/2 - C_k
    # We want to minimize max(O_k) = N/2 - min(C_k)
    # So we want to maximize min(C_k) for k != 0.
    
    # C is circular, so C[k] corresponds to shift k.
    # Note: np.fft.ifft returns indices 0..N-1.
    # C[0] is lag 0. C[1] is lag 1, ..., C[N-1] is lag N-1 (which is -1).
    
    min_c = np.min(C[1:])
    return N / 2.0 - min_c

def _local_search_fft(bits, max_iter=100000, time_limit=5.0):
    """
    Local search using FFT evaluation.
    """
    n = len(bits)
    bits = bits.copy()
    current_score = _compute_overlap_fft(bits)
    best_score = current_score
    best_bits = bits.copy()
    
    ones = np.where(bits == 1)[0]
    zeros = np.where(bits == 0)[0]
    
    rng = np.random.default_rng(42)
    start = time.time()
    
    for _ in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Random swap
        i1 = rng.integers(len(ones))
        i0 = rng.integers(len(zeros))
        pos1 = ones[i1]
        pos0 = zeros[i0]
        
        if pos1 == pos0:
            continue
            
        bits[pos1] = 0
        bits[pos0] = 1
        
        new_score = _compute_overlap_fft(bits)
        
        if new_score < best_score:
            best_score = new_score
            best_bits = bits.copy()
            current_score = new_score
            # Update indices
            ones[i1] = pos0
            zeros[i0] = pos1
        else:
            # Revert
            bits[pos1] = 1
            bits[pos0] = 0
    
    return best_bits, best_score

def _simulated_annealing_fft(bits, time_limit=30.0, T_start=5.0, T_end=0.01):
    """
    Simulated Annealing using FFT evaluation.
    """
    n = len(bits)
    bits = bits.copy()
    best_bits = bits.copy()
    best_score = _compute_overlap_fft(bits)
    current_score = best_score
    
    ones = np.where(bits == 1)[0]
    zeros = np.where(bits == 0)[0]
    
    rng = np.random.default_rng(123)
    start = time.time()
    T = T_start
    # Estimate number of steps to tune cooling
    # We don't know exact steps, so use time-based cooling or fixed steps
    # Let's assume ~10000 steps per second?
    # Better: cool based on time elapsed.
    
    while time.time() - start < time_limit:
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        # Pick random 1 and 0 to swap
        i1 = rng.integers(len(ones))
        i0 = rng.integers(len(zeros))
        pos1 = ones[i1]
        pos0 = zeros[i0]
        
        if pos1 == pos0:
            continue

        bits[pos1] = 0
        bits[pos0] = 1
        
        new_score = _compute_overlap_fft(bits)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_score = new_score
            ones[i1] = pos0
            zeros[i0] = pos1
            if new_score < best_score:
                best_score = new_score
                best_bits = bits.copy()
        else:
            bits[pos1] = 1
            bits[pos0] = 0
        
        # Cool down
        # Simple exponential cooling based on time
        elapsed = time.time() - start
        T = T_start * (T_end / T_start) ** (elapsed / time_limit)
        T = max(T, T_end)
        
    return best_bits, best_score

def construct():
    """
    Construct step heights minimizing max overlap with shifted 1-h.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with a balanced binary sequence
    # Random initialization is usually better than sorted for SA
    rng = np.random.default_rng(99)
    
    best_bits = None
    best_score = float('inf')
    
    # Try multiple random starts
    num_starts = 10
    
    for i in range(num_starts):
        if time.time() > deadline - 20.0:
            break
        
        # Random balanced binary sequence
        bits = np.zeros(N, dtype=np.int8)
        idx = rng.permutation(N)
        bits[idx[:N//2]] = 1
        
        # Quick local search to get to a local minimum
        ls_bits, ls_score = _local_search_fft(bits, max_iter=5000, time_limit=2.0)
        
        if ls_score < best_score:
            best_score = ls_score
            best_bits = ls_bits
        
        # Simulated Annealing from this local minimum
        if time.time() < deadline - 10.0:
            sa_bits, sa_score = _simulated_annealing_fft(
                ls_bits, 
                time_limit=min(10.0, deadline - time.time() - 5.0)
            )
            if sa_score < best_score:
                best_score = sa_score
                best_bits = sa_bits
            
            # Final local search polish
            if time.time() < deadline - 5.0:
                final_bits, final_score = _local_search_fft(
                    sa_bits, 
                    max_iter=10000, 
                    time_limit=min(5.0, deadline - time.time() - 2.0)
                )
                if final_score < best_score:
                    best_score = final_score
                    best_bits = final_bits

    # Convert to float h in [0,1]
    h = best_bits.astype(np.float64)
    
    # Verify constraints
    assert np.all(h >= 0) and np.all(h <= 1), "h out of bounds"
    assert abs(np.sum(h) - N/2) < 1e-10, f"sum mismatch: {np.sum(h)}"
    
    return h
