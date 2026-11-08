import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin before 120s kill

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
    """
    Generate multiple structured candidates and return the best.
    Kept for interface compatibility, but construct() may use its own init.
    """
    candidates = []
    candidates.append(np.ones(N, dtype=np.int8))
    candidates.append(np.tile(np.array([1, -1], dtype=np.int8), N // 2))
    
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
    
    # 1. Initialization
    # Use _structured_initialization to get a decent start
    init_step, init_c3 = _structured_initialization()
    
    # Also try a few random ones to see if we can beat the structured one quickly
    rng = np.random.RandomState(42)
    for _ in range(5):
        r_step = rng.choice([-1, 1], size=N).astype(np.int8)
        r_c3 = _compute_c3(r_step)
        if r_c3 < init_c3:
            init_step = r_step
            init_c3 = r_c3
            
    current_step = init_step.copy()
    current_c3 = init_c3
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # 2. Precomputation for fast updates
    M = 2 * N
    dx = 0.5 / M
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    
    # Map coarse index to fine indices
    coarse_to_fine = [[] for _ in range(N)]
    for m in range(M):
        coarse_to_fine[i_indices[m]].append(m)
    coarse_to_fine_arrays = [np.array(coarse_to_fine[i], dtype=np.int64) for i in range(N)]
    
    # Precompute G_i (FFT of unit pulse in bin i) and S_i (Autoconv of unit pulse)
    # G_i is shape (2*M,). We need N of them.
    # Memory: N * 2*M * 16 bytes (complex128) = 400 * 800 * 16 = 5.12 MB. OK.
    G_fft = np.zeros((N, 2 * M), dtype=np.complex128)
    S_self = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        delta_f = np.zeros(M, dtype=np.float64)
        delta_f[coarse_to_fine_arrays[i]] = 1.0
        G_fft[i] = np.fft.fft(delta_f, n=2 * M)
        # S_i = IFFT(G_i * conj(G_i)).real * dx
        S_self[i] = np.fft.ifft(G_fft[i] * np.conj(G_fft[i])).real * dx
        
    # Current Fine Grid and FFT
    f_fine = current_step.astype(np.float64)[i_indices]
    F_old = np.fft.fft(f_fine, n=2 * M)
    C_old = np.fft.ifft(F_old * np.conj(F_old)).real * dx
    current_c3 = np.max(np.abs(C_old))
    
    # 3. Simulated Annealing
    T_start = 0.5
    T_end = 0.001
    # Total time budget for SA
    sa_end_time = start_time + DEADLINE - 5.0 # Leave 5s for safety
    
    while time.time() < sa_end_time:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / DEADLINE)
        T = T_start * (T_end / T_start) ** progress
        
        i = rng.randint(N)
        old_val = current_step[i]
        new_val = -old_val
        
        # Delta value in time domain is (new_val - old_val) = -2 * old_val
        delta_val = -2.0 * old_val
        
        # G_i is FFT of unit pulse. 
        # Delta F = delta_val * G_i
        G_i = G_fft[i]
        
        # Cross term: 2 * IFFT(F_old * Delta F) = 2 * delta_val * IFFT(F_old * G_i)
        # Note: F_old and G_i are complex.
        # IFFT(F_old * G_i) is complex? 
        # F_old is FFT of real signal. G_i is FFT of real signal.
        # Product is not necessarily Hermitian, so IFFT is complex.
        # However, the autoconvolution is real.
        # Let's check: 
        # C_new = IFFT((F + dG)^2) = IFFT(F^2 + 2dFG + d^2 G^2).
        # IFFT(F^2) is real (autoconv).
        # IFFT(d^2 G^2) is real (autoconv of delta).
        # IFFT(2d F G). 
        # Is IFFT(F G) real?
        # F = FFT(f), G = FFT(g).
        # IFFT(F G) = f * g (circular convolution)? No, IFFT(FFT(f) FFT(g)) = circular convolution of f and g.
        # f and g are real. Circular convolution of real signals is real.
        # So yes, it is real. We can take .real.
        
        cross_term = 2.0 * delta_val * np.fft.ifft(F_old * G_i).real * dx
        self_term = (delta_val * delta_val) * S_self[i]
        
        new_C = C_old + cross_term + self_term
        new_c3 = np.max(np.abs(new_C))
        
        delta_c3 = new_c3 - current_c3
        
        if delta_c3 < 0 or (T > 0 and rng.random() < np.exp(-delta_c3 / T)):
            # Accept
            current_step[i] = new_val
            F_old += delta_val * G_i
            C_old = new_C
            current_c3 = new_c3
            
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
        else:
            # Reject
            pass
            
    return best_step
