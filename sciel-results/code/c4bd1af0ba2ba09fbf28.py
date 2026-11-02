"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_bound(f):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    The score is (1/N^2) * sum_k |R(k)| where R is the linear autocorrelation.
    For all-ones this gives 1.0. We want to minimize this.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    R = np.fft.irfft(F * np.conj(F), n=n_fft)
    l1_norm = np.abs(R[0]) + 2.0 * np.sum(np.abs(R[1:N]))
    return l1_norm / (N * N)


def _van_der_corput_sequence(n, base=2):
    """Generate the first n values of the van der Corput sequence (radical inverse).
    
    The van der Corput sequence is a low-discrepancy sequence that distributes
    points uniformly in [0,1). We use it to determine sign flips: if the value
    is below 0.5, sign is +1; otherwise -1. This creates a pattern with
    near-uniform distribution of sign changes across all frequency scales.
    """
    seq = np.zeros(n)
    for i in range(n):
        # Compute radical inverse for base=2 (binary)
        v = 0.0
        div = 1.0
        x = i
        while x > 0:
            div *= base
            v += (x % base) / div
            x //= base
        seq[i] = v
    return seq


def _construct_vdc_signs(n):
    """Construct sign sequence from van der Corput sequence."""
    vdc = _van_der_corput_sequence(n)
    # Map to signs: +1 if vdc < 0.5, -1 otherwise
    signs = np.where(vdc < 0.5, 1.0, -1.0)
    return signs


def _block_flip_move(f, block_size, rng):
    """Perform a block flip move: flip a contiguous block of signs."""
    f_new = f.copy()
    start = rng.integers(0, N - block_size + 1)
    f_new[start:start + block_size] *= -1.0
    return f_new


def _simulated_annealing(f_init, time_budget, rng):
    """Simulated annealing with block-flip moves to minimize C3 bound."""
    f = f_init.copy()
    current_val = _compute_c3_bound(f)
    best_f = f.copy()
    best_val = current_val
    
    # Temperature schedule: start high, decay exponentially
    T_start = 0.5
    T_end = 0.001
    total_steps = 20000
    
    for step in range(total_steps):
        if time.time() > time_budget:
            break
        
        # Temperature: linear decay
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        # Choose block size: mix of small and large blocks
        if rng.random() < 0.3:
            block_size = rng.integers(2, 8)
        elif rng.random() < 0.6:
            block_size = rng.integers(8, 32)
        else:
            block_size = rng.integers(32, 80)
        
        f_new = _block_flip_move(f, block_size, rng)
        new_val = _compute_c3_bound(f_new)
        
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            f = f_new
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_f = f.copy()
        
        # Every 500 steps, try a single-bit flip for fine-tuning
        if step % 500 == 499:
            idx = rng.integers(N)
            f_single = f.copy()
            f_single[idx] *= -1.0
            val_single = _compute_c3_bound(f_single)
            if val_single < current_val:
                f = f_single
                current_val = val_single
                if current_val < best_val:
                    best_val = current_val
                    best_f = f.copy()
    
    return best_f, best_val


def _pairwise_swap_optimize(f, time_budget, rng):
    """Local optimization via pairwise swaps (flip two bits)."""
    f = f.copy()
    current_val = _compute_c3_bound(f)
    best_f = f.copy()
    best_val = current_val
    improved = True
    
    while improved and time.time() < time_budget:
        improved = False
        # Try random pairs
        for _ in range(200):
            if time.time() > time_budget:
                break
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            f_new = f.copy()
            f_new[i] *= -1.0
            f_new[j] *= -1.0
            new_val = _compute_c3_bound(f_new)
            if new_val < current_val - 1e-15:
                f = f_new
                current_val = new_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_f = f.copy()
    
    return best_f, best_val


def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Strategy: Use van der Corput sequence as initial guess, then optimize
    # with simulated annealing using block flips, then fine-tune with pairwise swaps.
    
    # Generate multiple diverse initial candidates
    candidates = []
    
    # 1. Van der Corput based initial
    vdc_signs = _construct_vdc_signs(N)
    candidates.append(vdc_signs)
    
    # 2. Random initial
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 3. Modified VDC with different base
    vdc3 = _van_der_corput_sequence(N, base=3)
    candidates.append(np.where(vdc3 < 0.5, 1.0, -1.0))
    
    # 4. Thue-Morse-like sequence (complementary)
    tm = np.zeros(N)
    for i in range(N):
        # Count trailing 1s in binary representation
        x = i
        cnt = 0
        while x & 1:
            cnt += 1
            x >>= 1
        tm[i] = 1.0 if cnt % 2 == 0 else -1.0
    candidates.append(tm)
    
    # Evaluate all candidates
    best_f = None
    best_val = np.inf
    for cand in candidates:
        val = _compute_c3_bound(cand)
        if val < best_val:
            best_val = val
            best_f = cand.copy()
    
    # Run simulated annealing from the best candidate
    sa_time = time_limit * 0.5
    best_f, best_val = _simulated_annealing(best_f, start_time + sa_time, rng)
    
    # Fine-tune with pairwise swaps
    swap_time = time_limit * 0.3
    best_f, best_val = _pairwise_swap_optimize(best_f, start_time + swap_time, rng)
    
    # If we still have time, do another round of SA from a different start
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        # Try SA from a random restart
        f_rand = rng.choice([-1.0, 1.0], size=N)
        val_rand = _compute_c3_bound(f_rand)
        if val_rand < best_val * 1.1:  # Only if promising
            f2, val2 = _simulated_annealing(f_rand, start_time + time_limit, rng)
            if val2 < best_val:
                best_f, best_val = f2, val2
    
    # Final single-bit optimization pass
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        f = best_f.copy()
        cur_val = best_val
        for i in range(N):
            if time.time() > start_time + time_limit:
                break
            f[i] *= -1.0
            v = _compute_c3_bound(f)
            if v < cur_val - 1e-15:
                cur_val = v
            else:
                f[i] *= -1.0
        if cur_val < best_val:
            best_f, best_val = f, cur_val
    
    return best_f
# EVOLVE-BLOCK-END
