import numpy as np
import time

N = 400
DEADLINE = 145.0  # safety margin before 150s kill

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound (max absolute autocorrelation) for a signed step function.
    Uses FFT for linear autocorrelation.
    """
    f = np.asarray(signed_step, dtype=np.float64)
    # Linear autocorrelation via FFT with zero-padding to 2*N
    fft_f = np.fft.fft(f, n=2 * N)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    return np.max(np.abs(autoconv))

def _compute_c3_vectorized_batch(seqs, n=2*N):
    """
    Compute C3 for a batch of sequences. seqs is (B, N) array.
    Returns (B,) array of C3 values.
    """
    # FFT along last axis
    fft_seqs = np.fft.fft(seqs, n=n, axis=-1)
    # Power spectrum
    power = fft_seqs * np.conj(fft_seqs)
    # Inverse FFT
    autoconvs = np.fft.ifft(power, axis=-1).real
    # Max absolute value over all lags for each sequence
    return np.max(np.abs(autoconvs), axis=-1)

def construct():
    start_time = time.time()
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Initialize with a balanced random sequence
    init_seq = np.random.choice([-1, 1], size=N)
    # Ensure exact balance: 200 ones, 200 minus ones
    c_ones = np.sum(init_seq == 1)
    if c_ones != N // 2:
        idx_ones = np.where(init_seq == 1)[0]
        idx_minus = np.where(init_seq == -1)[0]
        if c_ones > N // 2:
            for i in range(c_ones - N // 2):
                init_seq[idx_ones[i]] = -1
        else:
            for i in range(N // 2 - c_ones):
                init_seq[idx_minus[i]] = 1
    
    best_seq = init_seq.copy()
    best_cost = _compute_c3(best_seq)
    current_seq = best_seq.copy()
    current_cost = best_cost
    
    # Simulated Annealing with multi-flip moves
    T_start = 0.1
    T_end = 1e-6
    max_iters = 5_000_000  # will be limited by time
    
    cool_rate = (T_end / T_start) ** (1.0 / max_iters)
    
    # Pre-allocate for batch evaluation
    batch_size = 8
    
    for i in range(max_iters):
        # Time check
        if time.time() - start_time > DEADLINE:
            break
        
        # Temperature
        T = T_start * (cool_rate ** i)
        if T < 1e-10:
            T = 1e-10
        
        # Multi-flip move: flip 1-3 random indices
        n_flips = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        flip_indices = np.random.choice(N, size=n_flips, replace=False)
        
        old_vals = current_seq[flip_indices].copy()
        current_seq[flip_indices] = -current_seq[flip_indices]
        
        new_cost = _compute_c3(current_seq)
        
        delta = new_cost - current_cost
        
        if delta < 0:
            # Accept
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_seq = current_seq.copy()
        else:
            # Accept with probability exp(-delta/T)
            if delta / T < 50:  # avoid overflow
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_cost = new_cost
                else:
                    # Reject, revert
                    current_seq[flip_indices] = old_vals
            else:
                # Reject, revert
                current_seq[flip_indices] = old_vals
        
        # Periodic restart from best if stuck
        if i % 100000 == 0 and i > 0:
            # Check if we've made progress recently
            if current_cost - best_cost > 0.1 * best_cost:
                # Restart from best with perturbation
                current_seq = best_seq.copy()
                current_cost = best_cost
                # Perturb a few random indices
                n_perturb = np.random.randint(5, 15)
                perturb_indices = np.random.choice(N, size=n_perturb, replace=False)
                current_seq[perturb_indices] = -current_seq[perturb_indices]
                current_cost = _compute_c3(current_seq)
    
    return best_seq
