"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_off_peak_autocorr(seq):
    """Compute the maximum absolute off-peak autocorrelation."""
    n = len(seq)
    if n < 2:
        return 0.0
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f = np.fft.fft(seq, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return float(np.max(np.abs(acf[1:])))

def _compute_psl(seq):
    """Compute peak sidelobe level (max |acf[1:]|)."""
    return _max_off_peak_autocorr(seq)

def _try_flip_pair(seq, i, j, deadline):
    """Try flipping two positions simultaneously."""
    seq[i] = -seq[i]
    seq[j] = -seq[j]
    score = _compute_psl(seq)
    seq[i] = -seq[i]
    seq[j] = -seq[j]
    return score

def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound C3."""
    t_start = time.time()
    deadline = t_start + 140  # Safety margin before 150s limit
    
    rng = np.random.default_rng(12345)
    
    best_seq = np.ones(N, dtype=np.float64)
    best_score = _compute_psl(best_seq)
    
    # Phase 1: Multiple random initializations with local search
    n_attempts = 20
    
    for attempt in range(n_attempts):
        if time.time() > deadline - 30:
            break
        
        # Initialize with random ±1
        seq = rng.choice([-1.0, 1.0], size=N)
        cur_score = _compute_psl(seq)
        
        if cur_score < best_score:
            best_score = cur_score
            best_seq = seq.copy()
        
        # Simulated annealing
        T = 50.0  # Initial temperature
        T_min = 1.0
        cooling_rate = 0.95
        
        while T > T_min:
            if time.time() > deadline - 30:
                break
            
            # Try single flip
            i = rng.integers(0, N)
            seq[i] = -seq[i]
            new_score = _compute_psl(seq)
            
            delta = new_score - cur_score
            if delta < 0 or rng.random() < np.exp(-delta / T):
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_seq = seq.copy()
            else:
                seq[i] = -seq[i]
            
            # Occasionally try double flip
            if rng.random() < 0.3:
                j = rng.integers(0, N)
                if j != i:
                    seq[i] = -seq[i]
                    seq[j] = -seq[j]
                    new_score2 = _compute_psl(seq)
                    delta2 = new_score2 - cur_score
                    if delta2 < 0 or rng.random() < np.exp(-delta2 / T):
                        cur_score = new_score2
                        if cur_score < best_score:
                            best_score = cur_score
                            best_seq = seq.copy()
                    else:
                        seq[i] = -seq[i]
                        seq[j] = -seq[j]
            
            T *= cooling_rate
    
    # Phase 2: Intensive local search on best sequence
    seq = best_seq.copy()
    cur_score = best_score
    
    # First pass: single flips, multiple passes until no improvement
    for pass_num in range(5):
        if time.time() > deadline - 10:
            break
        
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 10:
                break
            
            seq[i] = -seq[i]
            new_score = _compute_psl(seq)
            
            if new_score < cur_score:
                cur_score = new_score
                improved = True
            else:
                seq[i] = -seq[i]
        
        if not improved:
            break
    
    if cur_score < best_score:
        best_score = cur_score
        best_seq = seq.copy()
    
    # Phase 3: Double-flip local search
    if time.time() < deadline - 15:
        seq = best_seq.copy()
        cur_score = best_score
        
        for pass_num in range(3):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try pairs of flips
            indices = rng.permutation(N)
            for i_idx in range(0, N - 1, 2):
                if time.time() > deadline - 10:
                    break
                
                i = indices[i_idx]
                j = indices[i_idx + 1]
                
                new_score = _try_flip_pair(seq, i, j, deadline)
                if new_score < cur_score:
                    seq[i] = -seq[i]
                    seq[j] = -seq[j]
                    cur_score = new_score
                    improved = True
            
            if not improved:
                break
        
        if cur_score < best_score:
            best_score = cur_score
            best_seq = seq.copy()
    
    return best_seq
# EVOLVE-BLOCK-END
