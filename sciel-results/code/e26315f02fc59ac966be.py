import numpy as np
import time

N = 400
DEADLINE = 118.0
M = 4 * N  # fine grid size = 1600
dx = 0.5 / M  # domain length is 0.5
REP = M // N  # repeat factor = 4


def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the step function.
    Uses FFT on a fine grid for efficiency.
    """
    f_fine = np.repeat(signed_step.astype(np.float64), REP)
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
    autoconv *= dx
    return np.max(np.abs(autoconv))


def _make_fast_evaluator(signed_step):
    """
    Create a fast evaluator that uses the FFT of the base signal
    and analytically computes the effect of single flips.
    
    For a flip at position i: f_new = f - 2*f[i]*e_i
    autoconv_new = autoconv_old - 2*f[i]*(f * e_i + e_i * f) + 4*f[i]^2*(e_i * e_i)
    Since f[i] = ±1, f[i]^2 = 1
    (f * e_i)(t) = f(t-i)  (shifted copy of f)
    (e_i * e_i)(t) = delta at t=0 (a spike of height dx at origin)
    
    Actually, let me think more carefully.
    autoconv_f(t) = sum_k f[k] * f[k + (t/dx)] * dx  (discrete approximation)
    
    With FFT: autoconv = ifft(|fft(f)|^2) * dx
    
    If we flip position i: f_new[k] = f[k] - 2*f[i]*delta(k-i)
    fft(f_new) = fft(f) - 2*f[i]*exp(-2*pi*1*i*k/(2M))
    
    This is getting complex. Let me use a different approach:
    Precompute the autoconv, and for each flip, recompute only the affected parts.
    
    Actually, the simplest fast approach: just use the full FFT but with
    a pre-allocated array and minimize overhead.
    """
    f_fine_base = np.repeat(signed_step.astype(np.float64), REP)
    fft_base = np.fft.rfft(f_fine_base, n=2 * M)
    
    def fast_c3(step):
        """Compute C3 for a given step array."""
        f_fine = np.repeat(step.astype(np.float64), REP)
        fft_f = np.fft.rfft(f_fine, n=2 * M)
        autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
        autoconv *= dx
        return np.max(np.abs(autoconv))
    
    def fast_c3_from_fft(fft_f):
        """Compute C3 from an FFT."""
        autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
        autoconv *= dx
        return np.max(np.abs(autoconv))
    
    return fast_c3, fast_c3_from_fft, fft_base, f_fine_base


def _compute_c3_from_fft(fft_f):
    """Compute C3 directly from FFT."""
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
    autoconv *= dx
    return np.max(np.abs(autoconv))


def _sa_search(initial_step, deadline, seed, max_time=None):
    """
    Run simulated annealing with efficient evaluation.
    Uses full FFT recomputation but with minimal overhead.
    """
    if max_time is None:
        max_time = deadline - 3.0
    
    rng = np.random.RandomState(seed)
    
    current_step = initial_step.copy().astype(np.int8)
    current_c3 = _compute_c3(current_step)
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # SA parameters
    T = 0.005
    T_min = 1e-8
    alpha = 0.99995
    
    no_improve = 0
    patience = 3000
    
    # Pre-allocate
    f_fine_buf = np.empty(M, dtype=np.float64)
    
    while time.time() < max_time:
        # Choose move: single flip (80%) or multi-flip 2-3 (20%)
        r = rng.random()
        if r < 0.8:
            idx = rng.randint(N)
            old_val = current_step[idx]
            current_step[idx] = -old_val
            
            # Fast evaluation
            f_fine_buf[:] = np.repeat(current_step, REP)
            fft_f = np.fft.rfft(f_fine_buf, n=2 * M)
            autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
            autoconv *= dx
            new_c3 = np.max(np.abs(autoconv))
            
            delta = new_c3 - current_c3
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_c3 = new_c3
                if current_c3 < best_c3:
                    best_c3 = current_c3
                    best_step = current_step.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_step[idx] = old_val
                no_improve += 1
        else:
            n_flips = rng.randint(2, 4)
            idxs = rng.choice(N, size=n_flips, replace=False)
            old_vals = current_step[idxs].copy()
            current_step[idxs] = -old_vals
            
            f_fine_buf[:] = np.repeat(current_step, REP)
            fft_f = np.fft.rfft(f_fine_buf, n=2 * M)
            autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
            autoconv *= dx
            new_c3 = np.max(np.abs(autoconv))
            
            delta = new_c3 - current_c3
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_c3 = new_c3
                if current_c3 < best_c3:
                    best_c3 = current_c3
                    best_step = current_step.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_step[idxs] = old_vals
                no_improve += 1
        
        T = max(T_min, T * alpha)
        
        if no_improve >= patience:
            break
    
    return best_step, best_c3


def _greedy_polish(step, c3, deadline, max_passes=10):
    """
    Greedy single-flip polish to find local optimum.
    Multiple passes for thoroughness.
    """
    current_step = step.copy().astype(np.int8)
    current_c3 = c3
    
    f_fine_buf = np.empty(M, dtype=np.float64)
    
    for _pass in range(max_passes):
        if time.time() > deadline - 2.0:
            break
        improved = False
        for idx in range(N):
            if time.time() > deadline - 2.0:
                return current_step, current_c3
            
            old_val = current_step[idx]
            current_step[idx] = -old_val
            
            f_fine_buf[:] = np.repeat(current_step, REP)
            fft_f = np.fft.rfft(f_fine_buf, n=2 * M)
            autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
            autoconv *= dx
            new_c3 = np.max(np.abs(autoconv))
            
            if new_c3 < current_c3:
                current_c3 = new_c3
                improved = True
            else:
                current_step[idx] = old_val
        
        if not improved:
            break
    
    return current_step, current_c3


def _greedy_polish_fast(step, c3, deadline):
    """
    Faster greedy polish using a different strategy:
    Evaluate all possible flips in a batch-like manner.
    """
    current_step = step.copy().astype(np.int8)
    current_c3 = c3
    
    # Precompute base fine grid and FFT
    f_fine_base = np.repeat(current_step, REP).astype(np.float64)
    fft_base = np.fft.rfft(f_fine_base, n=2 * M)
    
    improved = True
    passes = 0
    max_passes = 20
    
    while improved and time.time() < deadline - 2.0 and passes < max_passes:
        improved = False
        passes += 1
        
        for idx in range(N):
            if time.time() > deadline - 2.0:
                return current_step, current_c3
            
            old_val = current_step[idx]
            current_step[idx] = -old_val
            
            f_fine = np.repeat(current_step, REP).astype(np.float64)
            fft_f = np.fft.rfft(f_fine, n=2 * M)
            autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
            autoconv *= dx
            new_c3 = np.max(np.abs(autoconv))
            
            if new_c3 < current_c3:
                current_c3 = new_c3
                improved = True
                # Update base
                f_fine_base = f_fine.copy()
                fft_base = fft_f.copy()
            else:
                current_step[idx] = old_val
    
    return current_step, current_c3


def _generate_initializations(deadline):
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. All ones (seed)
    inits.append(np.ones(N, dtype=np.int8))
    
    # 2. Alternating pattern
    alt = np.ones(N, dtype=np.int8)
    alt[1::2] = -1
    inits.append(alt)
    
    # 3. Random with different seeds
    for seed in [42, 123, 456, 789, 1024]:
        if time.time() > deadline - 30.0:
            break
        rng = np.random.RandomState(seed)
        init = rng.choice([-1, 1], size=N).astype(np.int8)
        inits.append(init)
    
    # 4. Structured: blocks of +1 and -1
    rng = np.random.RandomState(777)
    n_blocks = rng.randint(4, 12)
    block_sizes = rng.multinomial(N, np.ones(n_blocks) / n_blocks)
    init = np.empty(N, dtype=np.int8)
    pos = 0
    for i, bs in enumerate(block_sizes):
        init[pos:pos+bs] = 1 if i % 2 == 0 else -1
        pos += bs
    inits.append(init)
    
    # 5. Sine-like pattern
    x = np.linspace(0, 4 * np.pi, N)
    inits.append(np.sign(np.sin(x)).astype(np.int8))
    
    return inits


def construct():
    start_time = time.time()
    
    # Phase 1: Quick greedy polish from all-ones to get a baseline
    initial = np.ones(N, dtype=np.int8)
    best_step = initial.copy()
    best_c3 = _compute_c3(best_step)
    
    # Quick polish (limited time)
    if time.time() < DEADLINE - 20.0:
        best_step, best_c3 = _greedy_polish_fast(best_step, best_c3, DEADLINE - 15.0)
    
    # Phase 2: Multi-start SA with diverse initializations
    inits = _generate_initializations(DEADLINE - 10.0)
    
    # Allocate time for each SA run
    n_sa = len(inits)
    time_per_sa = (DEADLINE - 15.0 - time.time()) / max(n_sa, 1)
    
    for i, init in enumerate(inits):
        if time.time() > DEADLINE - 10.0:
            break
        
        sa_deadline = time.time() + max(time_per_sa, 5.0)
        if sa_deadline > DEADLINE - 10.0:
            sa_deadline = DEADLINE - 10.0
        
        step_sa, c3_sa = _sa_search(init, sa_deadline, seed=1000 + i * 100)
        
        if c3_sa < best_c3:
            best_step = step_sa
            best_c3 = c3_sa
    
    # Phase 3: Polish the best result found
    if time.time() < DEADLINE - 3.0:
        best_step, best_c3 = _greedy_polish_fast(best_step, best_c3, DEADLINE)
    
    # Phase 4: Final greedy polish with more passes
    if time.time() < DEADLINE - 1.0:
        best_step, best_c3 = _greedy_polish(best_step, best_c3, DEADLINE, max_passes=15)
    
    return best_step.astype(np.int8)
