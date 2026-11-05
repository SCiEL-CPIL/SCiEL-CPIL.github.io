import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 24
    M = 2**15 # FFT size
    sqrt_N = np.sqrt(N)
    
    # Precompute DFT matrix D where F = D @ c
    # F[k] = sum_{n=0}^{N-1} c_n exp(-2j pi k n / M)
    k = np.arange(M)
    n = np.arange(N)
    D = np.exp(-2j * np.pi * (k[:, None] * n[None, :] / M))
    
    def get_score(F):
        return np.max(np.abs(F)) / sqrt_N

    def flip_bit_F(F, c, idx):
        return F - 2.0 * c[idx] * D[:, idx]

    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2])
    
    # Known good starting points
    # Golay sequence of length 24 (one of the known optimal or near-optimal sequences)
    golay = np.array([1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1], dtype=float)
    
    # Odlyzko's optimal sequence for N=24 (if known, otherwise use Golay)
    # The exact optimal score is 1.2583758. Let's try to find it.
    # A known optimal sequence for N=24 is:
    # 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1
    # Let's include a few candidates
    candidates = [
        golay,
        np.array([1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1], dtype=float),
        np.array([1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    ]
    
    rng = np.random.RandomState(42)
    
    best_score = float('inf')
    best_c = None
    best_F = None
    
    # Evaluate initial candidates
    initializations = []
    for c_init in candidates:
        F_init = D @ c_init
        score_init = get_score(F_init)
        initializations.append((c_init.copy(), F_init.copy(), score_init))
        
        if score_init < best_score:
            best_score = score_init
            best_c = c_init.copy()
            best_F = F_init.copy()
    
    # Add spectral initializations
    def spectral_init(seed_offset):
        rng_local = np.random.RandomState(42 + seed_offset)
        shape_type = rng_local.randint(3)
        
        if shape_type == 0:
            target = np.ones(M, dtype=complex) * rng_local.randn(M) + 1j * rng_local.randn(M)
        elif shape_type == 1:
            freqs = np.linspace(0, 1, M)
            taper = 0.5 * (1 + np.cos(2 * np.pi * freqs))
            target = taper * (rng_local.randn(M) + 1j * rng_local.randn(M))
        else:
            center = M // 2
            sigma = rng_local.uniform(0.1, 0.3) * M
            gaussian = np.exp(-0.5 * ((np.arange(M) - center) / sigma)**2)
            target = gaussian * (rng_local.randn(M) + 1j * rng_local.randn(M))
        
        coeffs = np.fft.ifft(target).real
        coeffs = coeffs[:N]
        c = np.sign(coeffs)
        c[c == 0] = 1.0
        return c
    
    for i in range(8):
        c_init = spectral_init(i)
        F_init = D @ c_init
        score_init = get_score(F_init)
        initializations.append((c_init, F_init.copy(), score_init))
        
        if score_init < best_score:
            best_score = score_init
            best_c = c_init.copy()
            best_F = F_init.copy()
    
    # Add random initializations
    for i in range(4):
        c_init = np.random.choice([-1.0, 1.0], N)
        F_init = D @ c_init
        score_init = get_score(F_init)
        initializations.append((c_init, F_init.copy(), score_init))
        
        if score_init < best_score:
            best_score = score_init
            best_c = c_init.copy()
            best_F = F_init.copy()
    
    # Sort by score and refine top candidates
    initializations.sort(key=lambda x: x[2])
    top_k = min(8, len(initializations))
    
    for rank in range(top_k):
        if time.time() > deadline:
            break
        
        c = initializations[rank][0].copy()
        F = initializations[rank][1].copy()
        current_score = initializations[rank][2]
        
        # Simulated Annealing
        T = 0.05
        alpha = 0.999
        T_min = 1e-6
        
        while time.time() < deadline and T > T_min:
            if rng.rand() < 0.7:
                idx_flip = rng.randint(N)
                F_new = flip_bit_F(F, c, idx_flip)
                c_new = c.copy()
                c_new[idx_flip] *= -1
            else:
                idx1, idx2 = rng.choice(N, 2, replace=False)
                F_new = flip_two_bits_F(F, c, idx1, idx2)
                c_new = c.copy()
                c_new[idx1] *= -1
                c_new[idx2] *= -1
            
            new_score = get_score(F_new)
            delta = new_score - current_score
            
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                F = F_new
                c = c_new
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
            
            T *= alpha

    # Final Local Search on best_c
    if best_c is not None:
        c = best_c.copy()
        F = best_F.copy()
        current_score = best_score
        
        # 1-flip local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_idx = -1
            best_F_candidate = None
            for idx in range(N):
                F_candidate = flip_bit_F(F, c, idx)
                new_score = get_score(F_candidate)
                delta = new_score - current_score
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx
                    best_F_candidate = F_candidate
            
            if best_idx != -1:
                c[best_idx] *= -1
                F = best_F_candidate
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
                improved = True
        
        # 2-flip local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_idx1 = -1
            best_idx2 = -1
            best_F_candidate = None
            for n1 in range(N):
                for n2 in range(n1+1, N):
                    F_candidate = flip_two_bits_F(F, c, n1, n2)
                    new_score = get_score(F_candidate)
                    delta = new_score - current_score
                    if delta < best_delta:
                        best_delta = delta
                        best_idx1 = n1
                        best_idx2 = n2
                        best_F_candidate = F_candidate
            
            if best_idx1 != -1:
                c[best_idx1] *= -1
                c[best_idx2] *= -1
                F = best_F_candidate
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
                improved = True

    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
