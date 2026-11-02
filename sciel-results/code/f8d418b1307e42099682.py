"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(seq):
    """Compute full autocorrelation via FFT."""
    n = len(seq)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f = np.fft.fft(seq, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf

def _psl(seq):
    """Peak sidelobe level: max |acf[1:]|."""
    acf = _compute_acf(seq)
    return float(np.max(np.abs(acf[1:])))

def _update_acf_on_flip(acf, seq, i, n):
    """
    Compute the change in autocorrelation when flipping position i.
    Returns the delta array (length n) of changes to acf.
    """
    delta = np.zeros(n)
    si = seq[i]
    # For each lag k=1..n-1:
    # delta[k] = -2*si*seq[i+k] if i+k < n else 0
    #          - 2*seq[i-k]*si if i-k >= 0 else 0
    # Vectorize:
    # Term 1: i+k < n => k < n-i => k in [1, min(n-1, n-1-i)]
    # Term 2: i-k >= 0 => k <= i => k in [1, min(n-1, i)]
    k_max1 = min(n - 1, n - 1 - i)
    k_max2 = min(n - 1, i)
    
    if k_max1 >= 1:
        ks1 = np.arange(1, k_max1 + 1)
        delta[ks1] += -2 * si * seq[i + ks1]
    if k_max2 >= 1:
        ks2 = np.arange(1, k_max2 + 1)
        delta[ks2] += -2 * seq[i - ks2] * si
    
    return delta

def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound C3."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Phase 1: Random initialization with multiple restarts
    best_seq = None
    best_score = float('inf')
    
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        seq = np.random.choice([-1.0, 1.0], size=N)
        
        # Compute initial acf
        acf = _compute_acf(seq)
        current_score = float(np.max(np.abs(acf[1:])))
        
        if current_score < best_score:
            best_score = current_score
            best_seq = seq.copy()
        
        # Phase 2: Iterative local search with incremental acf updates
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 10:
            if no_improve > max_no_improve:
                break
            
            # Find the flip that gives the best improvement
            best_improvement = 0.0
            best_flip_idx = -1
            best_new_score = current_score
            
            # Sample a subset of positions to flip to save time
            # Use all positions for thoroughness, but we can subsample if needed
            for j in range(N):
                if time.time() > deadline - 10:
                    break
                
                # Compute delta acf
                delta = _update_acf_on_flip(acf, seq, j, N)
                new_acf = acf + delta
                new_score = float(np.max(np.abs(new_acf[1:])))
                
                if new_score < best_new_score - 1e-12:
                    best_new_score = new_score
                    best_flip_idx = j
                    best_improvement = current_score - new_score
            
            if best_flip_idx != -1:
                # Apply the flip
                seq[best_flip_idx] = -seq[best_flip_idx]
                acf = acf + _update_acf_on_flip(acf, seq, best_flip_idx, N)
                # Recompute acf exactly to avoid drift
                acf = _compute_acf(seq)
                current_score = float(np.max(np.abs(acf[1:])))
                no_improve = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_seq = seq.copy()
            else:
                no_improve += 1
    
    if best_seq is None:
        best_seq = np.ones(N)
    
    return best_seq
# EVOLVE-BLOCK-END
