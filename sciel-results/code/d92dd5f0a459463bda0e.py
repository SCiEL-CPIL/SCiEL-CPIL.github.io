"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_norm_accurate(coeffs):
    """
    Computes the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling (FFT) and golden section refinement for accuracy.
    Returns max |P(z)|.
    """
    n = len(coeffs)
    # Dense sampling using FFT
    M = 8192
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:n] = coeffs
    P = M * np.fft.ifft(c_padded)
    mags = np.abs(P)
    
    # Find top K peaks
    K = 50
    if M < K:
        idx = np.arange(M)
    else:
        idx = np.argsort(mags)[-K:][::-1]
    
    max_val = 0.0
    # Golden section refinement
    w = 2 * np.pi / M
    gr = (5 ** 0.5 - 1) / 2
    
    # Precompute coeffs for Horner
    for i in idx:
        theta_center = 2 * np.pi * i / M
        a = theta_center - w
        b = theta_center + w
        
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            p_val = 0.0
            for c in reversed(coeffs):
                p_val = p_val * z_val + c
            return np.abs(p_val)
        
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        
        fc = f(c_pt)
        fd = f(d_pt)
        
        for _ in range(50):
            if fc < fd:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = f(d_pt)
            else:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = f(c_pt)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def _score_accurate(coeffs):
    """
    Computes the score: max |P(z)| / sqrt(24)
    """
    n = len(coeffs)
    norm = _eval_norm_accurate(coeffs)
    return norm / np.sqrt(n)

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing the score.
    Uses vectorized local search with spectral synthesis initialization.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute DFT matrix for fast evaluation on a fixed grid
    M_grid = 16384
    theta_grid = 2 * np.pi * np.arange(M_grid) / M_grid
    # E[i, k] = exp(i * theta_i * k)
    E = np.exp(1j * np.outer(theta_grid, np.arange(N)))
    
    def get_supnorm_fast(coeffs):
        """Fast sup norm via dense grid."""
        P = E @ coeffs
        return float(np.max(np.abs(P)))
    
    def local_search_vectorized(current_c, time_limit):
        """
        Fast local search using vectorized 1-flip and 2-flip evaluation.
        """
        current_c = current_c.copy()
        P_curr = E @ current_c
        current_val = float(np.max(np.abs(P_curr)))
        best_c = current_c.copy()
        best_val = current_val
        
        # Precompute basis vectors for flips: delta_i = 2 * c_i * E[:, i]
        # If we flip c_i to -c_i, P changes by -2 * c_i * E[:, i]
        # new_P = P_curr - 2 * current_c[i] * E[:, i]
        
        max_iters = 1000
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break
            
            improved = False
            
            # 1-flip search
            best_flip_val = current_val
            best_flip_idx = -1
            for i in range(N):
                # P_new = P_curr - 2 * current_c[i] * E[:, i]
                P_new = P_curr - 2.0 * current_c[i] * E[:, i]
                val = float(np.max(np.abs(P_new)))
                if val < best_flip_val - 1e-12:
                    best_flip_val = val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                i = best_flip_idx
                current_c[i] *= -1
                P_curr = P_curr - 2.0 * current_c[i] * E[:, i] # Update P
                current_val = best_flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                improved = True
                continue
            
            # 2-flip search
            best_2flip_val = current_val
            best_2flip = None
            for i in range(N - 1):
                for j in range(i + 1, N):
                    # P_new = P_curr - 2*c_i*E_i - 2*c_j*E_j
                    P_new = P_curr - 2.0 * current_c[i] * E[:, i] - 2.0 * current_c[j] * E[:, j]
                    val = float(np.max(np.abs(P_new)))
                    if val < best_2flip_val - 1e-12:
                        best_2flip_val = val
                        best_2flip = (i, j)
            
            if best_2flip is not None:
                i, j = best_2flip
                current_c[i] *= -1
                current_c[j] *= -1
                P_curr = P_curr - 2.0 * current_c[i] * E[:, i] - 2.0 * current_c[j] * E[:, j]
                current_val = best_2flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                improved = True
                continue
            
            # 3-flip search (every few iterations to escape local minima)
            if iteration % 5 == 0:
                best_3flip_val = current_val
                best_3flip = None
                for i in range(N - 2):
                    for j in range(i + 1, N - 1):
                        for k in range(j + 1, N):
                            P_new = P_curr - 2.0 * current_c[i] * E[:, i] - 2.0 * current_c[j] * E[:, j] - 2.0 * current_c[k] * E[:, k]
                            val = float(np.max(np.abs(P_new)))
                            if val < best_3flip_val - 1e-12:
                                best_3flip_val = val
                                best_3flip = (i, j, k)
                
                if best_3flip is not None:
                    i, j, k = best_3flip
                    current_c[i] *= -1
                    current_c[j] *= -1
                    current_c[k] *= -1
                    P_curr = P_curr - 2.0 * current_c[i] * E[:, i] - 2.0 * current_c[j] * E[:, j] - 2.0 * current_c[k] * E[:, k]
                    current_val = best_3flip_val
                    if current_val < best_val - 1e-12:
                        best_c = current_c.copy()
                        best_val = current_val
                    improved = True
                    continue
            
            if not improved:
                break
        
        return best_c, best_val

    # --- Initialization ---
    initial_candidates = []
    
    # 1. Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    initial_candidates.append(np.array(p[:N], dtype=float))
    
    # 2. Spectral synthesis
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 60:
            break
        # Random phase
        phases = np.random.uniform(0, 2 * np.pi, size=N)
        spectrum = np.exp(1j * phases)
        c_complex = np.fft.ifft(spectrum)
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        c_bin[c_bin == 0] = 1
        initial_candidates.append(c_bin)
        
        # Tapered magnitude
        taper = np.cos(np.pi * np.arange(N) / (N - 1)) ** 2
        spectrum2 = taper * np.exp(1j * phases)
        c_complex2 = np.fft.ifft(spectrum2)
        c_real2 = np.real(c_complex2)
        c_bin2 = np.sign(c_real2)
        c_bin2[c_bin2 == 0] = 1
        initial_candidates.append(c_bin2)
        
        # Gaussian taper
        centers = np.random.uniform(0, N, size=2)
        widths = np.random.uniform(2, 6, size=2)
        taper3 = np.zeros(N)
        for c, w in zip(centers, widths):
            taper3 += np.exp(-((np.arange(N) - c) ** 2) / (2 * w ** 2))
        spectrum3 = taper3 * np.exp(1j * phases)
        c_complex3 = np.fft.ifft(spectrum3)
        c_real3 = np.real(c_complex3)
        c_bin3 = np.sign(c_real3)
        c_bin3[c_bin3 == 0] = 1
        initial_candidates.append(c_bin3)
    
    # 3. Random starts
    for _ in range(20):
        if time.time() > deadline - 60:
            break
        initial_candidates.append(np.random.choice([-1, 1], size=N).astype(float))

    best_coeffs = None
    best_fast_score = float('inf')
    
    # Phase 1: Local search on all initial candidates
    for c in initial_candidates:
        if time.time() > deadline - 30:
            break
        c_local, score_local = local_search_vectorized(c, deadline - 30)
        if score_local < best_fast_score:
            best_fast_score = score_local
            best_coeffs = c_local.copy()
    
    # Phase 2: Iterated Local Search (ILS)
    # Perturb the best solution and run local search again
    current_best = best_coeffs.copy()
    current_best_score = best_fast_score
    
    while time.time() < deadline - 10:
        # Perturbation: flip 3-5 random bits
        num_flips = np.random.randint(3, 6)
        indices = np.random.choice(N, num_flips, replace=False)
        perturbed = current_best.copy()
        for i in indices:
            perturbed[i] *= -1
        
        # Local search from perturbed state
        c_local, score_local = local_search_vectorized(perturbed, deadline - 10)
        
        if score_local < current_best_score:
            current_best_score = score_local
            current_best = c_local.copy()
            best_coeffs = c_local.copy()
            best_fast_score = score_local
        else:
            # If no improvement, we might stay with current_best or try a different perturbation
            # Standard ILS keeps the best found so far
            pass
            
    # Final accurate evaluation
    final_score = _score_accurate(best_coeffs)
    
    return [int(c) for c in best_coeffs]

# EVOLVE-BLOCK-END
