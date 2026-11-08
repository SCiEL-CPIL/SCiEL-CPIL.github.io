import numpy as np
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the signed step function.
    """
    M = 2 * N  # fine grid for accuracy
    dx = 0.5 / M
    
    # Build fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    f_fine = signed_step.astype(np.float64)[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def _structured_initialization():
    """Generate multiple structured candidates and return the best."""
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N, dtype=np.int8))
    
    # Alternating
    candidates.append(np.tile(np.array([1, -1], dtype=np.int8), N // 2))
    
    # Random
    np.random.seed(42)
    for _ in range(10):
        candidates.append(np.random.choice([-1, 1], size=N).astype(np.int8))
    
    best = None
    best_c3 = np.inf
    for c in candidates:
        c3 = _compute_c3(c)
        if c3 < best_c3:
            best_c3 = c3
            best = c.copy()
    
    return best, best_c3

def construct():
    start_time = time.time()
    
    # Get a good structured start
    best_step, best_c3 = _structured_initialization()
    
    # Precompute fine grid mapping
    M = 2 * N
    dx = 0.5 / M
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    
    # Build fine grid signal
    current_step = best_step.copy()
    f_fine = current_step.astype(np.float64)[i_indices]
    
    # Compute initial autoconvolution
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    current_c3 = np.max(np.abs(autoconv))
    
    # For each coarse index i, the fine positions are 2i and 2i+1 (since M=2N)
    # Verify: i_indices[m] = m * N // M = m * N // (2N) = m // 2
    # So fine position m maps to coarse index m // 2
    # Thus coarse index i maps to fine positions 2*i and 2*i+1
    
    # Simulated annealing parameters
    T_start = 0.05
    T_end = 0.001
    max_iter = 500000  # Much more iterations possible with O(M) updates
    
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
    
    for iteration in range(max_iter):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Propose a move: flip 1 random position
        flip_idx = np.random.randint(N)
        old_val = current_step[flip_idx]
        new_val = -old_val
        d = float(new_val - old_val)  # = -2 * old_val
        
        # Fine positions for this coarse index
        m1 = 2 * flip_idx
        m2 = 2 * flip_idx + 1
        
        # Update autoconvolution:
        # new_autoconv[t] = old_autoconv[t] + d * (f[(m1+t)%M] + f[(m2+t)%M] + f[(m1-t)%M] + f[(m2-t)%M]) + d^2 * (delta(t) + delta(t-(m2-m1)) + delta(t+(m2-m1)))
        # Wait, let me be more careful.
        # autoconv[t] = sum_m f[m] * f[(m+t) % M] * dx
        # Change: f[m] -> f[m] + d*1[m in {m1,m2}]
        # new_autoconv[t] = sum_m (f[m] + d*1[m in {m1,m2}]) * (f[(m+t)%M] + d*1[(m+t)%M in {m1,m2}]) * dx
        # = old_autoconv[t] + d * sum_{m in {m1,m2}} f[(m+t)%M] * dx + d * sum_{m: (m+t)%M in {m1,m2}} f[m] * dx + d^2 * (1[t in {0, m2-m1, m1-m2}]) * dx
        
        # First cross term: d * (f[(m1+t)%M] + f[(m2+t)%M]) * dx
        # Second cross term: d * (f[(m1-t)%M] + f[(m2-t)%M]) * dx  [since (m+t)%M = m1 means m = (m1-t)%M]
        # Third term: d^2 * dx * (number of pairs)
        # Pairs (m_a, m_b) where m_a in {m1,m2} and (m_a+t)%M in {m1,m2}
        # i.e., t such that m_a + t ≡ m_b (mod M) for m_a,m_b in {m1,m2}
        # t ≡ m_b - m_a (mod M)
        # Possible t values: 0, m2-m1=1, m1-m2=M-1, and if m1=m2 then just 0
        # Since m2 = m1+1: t in {0, 1, M-1}
        
        # Compute the shift
        shift = np.roll(f_fine, -t) if we had t... but we need all t at once
        
        # Vectorized approach:
        # cross1[t] = f[(m1+t)%M] + f[(m2+t)%M] for all t
        # = np.roll(f_fine, -m1) + np.roll(f_fine, -m2)  [since f[(m+t)%M] = roll(f, -m)[t]]
        
        cross1 = np.roll(f_fine, -m1) + np.roll(f_fine, -m2)
        cross2 = np.roll(f_fine, m1) + np.roll(f_fine, m2)
        # Wait, f[(m-t)%M] = np.roll(f_fine, m)[t]
        
        new_autoconv = autoconv + d * dx * (cross1 + cross2)
        
        # Add d^2 * dx at lags 0, 1, M-1
        d2 = d * d
        new_autoconv[0] += d2 * dx
        new_autoconv[1] += d2 * dx
        new_autoconv[M-1] += d2 * dx
        
        new_c3 = np.max(np.abs(new_autoconv))
        
        delta = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            # Accept the move
            current_step[flip_idx] = new_val
            f_fine[m1] = float(new_val)
            f_fine[m2] = float(new_val)
            autoconv = new_autoconv
            current_c3 = new_c3
            
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        
        T *= cooling_rate
    
    return best_step
