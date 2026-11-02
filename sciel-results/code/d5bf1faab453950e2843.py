import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _halton_sequence(n, base):
    """Generate first n values of Halton sequence in given base."""
    result = np.zeros(n)
    for i in range(1, n + 1):
        f = 1.0 / base
        val = 0.0
        j = i
        while j > 0:
            f /= base
            val += f * (j % base)
            j //= base
        result[i - 1] = val
    return result

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Generate Halton sequences for different bases to create diverse initializations
    # Use bases that are coprime for better distribution
    bases = [2, 3, 5, 7, 11, 13, 17, 19]
    
    for base in bases:
        if time.time() > deadline:
            break
        
        # Generate Halton sequence and map to signs
        halton_vals = _halton_sequence(N, base)
        # Map fractional parts to signs: +1 if val < 0.5, -1 otherwise
        # This creates a balanced sign sequence with low discrepancy
        s = np.where(halton_vals < 0.5, 1.0, -1.0)
        
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        improved = True
        passes = 0
        max_passes = 50
        local_start = time.time()
        
        while improved and passes < max_passes and time.time() - local_start < 10.0 and time.time() < deadline:
            improved = False
            passes += 1
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                old_si = s[i]
                s[i] = -old_si
                
                A_new = _calc_acf(s)
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    current_val = val_new
                    A = A_new
                    improved = True
                else:
                    s[i] = old_si
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # If time remains, do random restarts with refinement
    while time.time() < deadline:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        local_start = time.time()
        improved = True
        passes = 0
        max_passes = 30
        
        while improved and passes < max_passes and time.time() - local_start < 5.0 and time.time() < deadline:
            improved = False
            passes += 1
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                old_si = s[i]
                s[i] = -old_si
                
                A_new = _calc_acf(s)
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    current_val = val_new
                    A = A_new
                    improved = True
                else:
                    s[i] = old_si
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()

    return best_seq
