"""Rewritten program: spectral synthesis + vectorized local search + SA polish."""
import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Precompute DFT matrix
    M_grid = 16384
    theta_grid = 2 * np.pi * np.arange(M_grid) / M_grid
    E = np.exp(1j * np.outer(theta_grid, np.arange(N)))
    basis = E.T  # (N, M_grid)

    def get_supnorm_fast(coeffs):
        P = E @ coeffs
        return float(np.max(np.abs(P)))

    def get_supnorm_accurate(coeffs):
        """Accurate sup norm with golden-section refinement of top peaks."""
        M = 16384
        c_padded = np.zeros(M, dtype=complex)
        c_padded[:N] = coeffs
        P = M * np.fft.ifft(c_padded)
        mags = np.abs(P)
        
        K = 100
        idx = np.argsort(mags)[-K:][::-1]
        theta = 2 * np.pi * np.arange(M) / M
        
        max_val = 0.0
        for i in idx:
            w = 2 * np.pi / M
            a = theta[i] - w
            b = theta[i] + w
            gr = (5 ** 0.5 - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def f(theta_val):
                z_val = np.exp(1j * theta_val)
                p_val = 0.0
                for coeff in reversed(coeffs):
                    p_val = p_val * z_val + coeff
                return np.abs(p_val)
            
            fc = f(c)
            fd = f(d)
            for _ in range(60):
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(c)
            peak_val = max(fc, fd)
            if peak_val > max_val:
                max_val = peak_val
        return max_val

    def local_search(current_c, time_limit):
        """Vectorized 1-flip and 2-flip local search."""
        current_c = current_c.copy()
        current_val = get_supnorm_fast(current_c)
        best_c = current_c.copy()
        best_val = current_val
        
        P_curr = E @ current_c
        single_deltas = [2 * current_c[i] * basis[i] for i in range(N)]
        
        max_iters = 2000
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break
            
            improved = False
            
            # 1-flip search
            best_flip_val = current_val
            best_flip_idx = -1
            for i in range(N):
                P_new = P_curr - single_deltas[i]
                val = np.max(np.abs(P_new))
                if val < best_flip_val - 1e-12:
                    best_flip_val = val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                i = best_flip_idx
                current_c[i] *= -1
                P_curr = E @ current_c
                current_val = best_flip_val
                single_deltas[i] = 2 * current_c[i] * basis[i]
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                improved = True
            
            if not improved:
                # 2-flip search
                best_2flip_val = current_val
                best_2flip = None
                for i in range(N - 1):
                    for j in range(i + 1, N):
                        P_new = P_curr - single_deltas[i] - single_deltas[j]
                        val = np.max(np.abs(P_new))
                        if val < best_2flip_val - 1e-12:
                            best_2flip_val = val
                            best_2flip = (i, j)
                
                if best_2flip is not None:
                    i, j = best_2flip
                    current_c[i] *= -1
                    current_c[j] *= -1
                    P_curr = E @ current_c
                    current_val = best_2flip_val
                    single_deltas[i] = 2 * current_c[i] * basis[i]
                    single_deltas[j] = 2 * current_c[j] * basis[j]
                    if current_val < best_val - 1e-12:
                        best_c = current_c.copy()
                        best_val = current_val
                    improved = True
            
            if not improved:
                break
        
        return best_c, best_val

    # --- Spectral Synthesis Initialization ---
    initial_candidates = []
    
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p_new = p + q
        q_new = p + [-c for c in q]
        p, q = p_new, q_new
    initial_candidates.append(np.array(p[:N], dtype=float))
    
    # 2. Spectral synthesis with various target spectra
    for trial in range(40):
        if time.time() > deadline - 30:
            break
        
        # Different spectral shapes
        t = np.arange(N)
        if trial % 4 == 0:
            # Flat spectrum with random phases
            phases = np.random.uniform(0, 2 * np.pi, size=N)
            spectrum = np.exp(1j * phases)
        elif trial % 4 == 1:
            # Cosine taper
            taper = np.cos(np.pi * t / (N - 1)) ** 2
            phases = np.random.uniform(0, 2 * np.pi, size=N)
            spectrum = taper * np.exp(1j * phases)
        elif trial % 4 == 2:
            # Gaussian taper
            center = N / 2
            taper = np.exp(-((t - center) ** 2) / (2 * (N / 4) ** 2))
            phases = np.random.uniform(0, 2 * np.pi, size=N)
            spectrum = taper * np.exp(1j * phases)
        else:
            # Smooth random spectrum
            phases = np.cumsum(np.random.normal(0, 0.5, size=N))
            phases = phases - np.mean(phases)
            spectrum = np.exp(1j * phases)
        
        c_complex = np.fft.ifft(spectrum)
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        c_bin[c_bin == 0] = 1
        initial_candidates.append(c_bin)
    
    # 3. Random starts
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        initial_candidates.append(c_rand)
    
    # Run local search on all candidates
    best_overall_c = None
    best_overall_val = float('inf')
    
    for c0 in initial_candidates:
        if time.time() > deadline - 15:
            break
        remaining = deadline - 15 - time.time()
        alloc = max(1.0, min(3.0, remaining / max(1, len(initial_candidates))))
        c_best, v_best = local_search(c0, time.time() + alloc)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best
    
    # Simulated annealing
    T = 0.5
    cooling_rate = 0.998
    min_T = 1e-7
    
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    triples = [(i, j, k) for i in range(N - 2) for j in range(i + 1, N - 1) for k in range(j + 1, N)]
    
    last_local_search = time.time()
    
    while T > min_T and time.time() < deadline - 20:
        r = np.random.rand()
        if r < 0.6:
            i, j = pairs[np.random.randint(len(pairs))]
            best_overall_c[i] *= -1
            best_overall_c[j] *= -1
            new_score = get_supnorm_fast(best_overall_c)
            delta = new_score - best_overall_val
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_overall_val = new_score
            else:
                best_overall_c[i] *= -1
                best_overall_c[j] *= -1
        elif r < 0.85:
            i = np.random.randint(N)
            best_overall_c[i] *= -1
            new_score = get_supnorm_fast(best_overall_c)
            delta = new_score - best_overall_val
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_overall_val = new_score
            else:
                best_overall_c[i] *= -1
        else:
            i, j, k = triples[np.random.randint(len(triples))]
            best_overall_c[i] *= -1
            best_overall_c[j] *= -1
            best_overall_c[k] *= -1
            new_score = get_supnorm_fast(best_overall_c)
            delta = new_score - best_overall_val
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_overall_val = new_score
            else:
                best_overall_c[i] *= -1
                best_overall_c[j] *= -1
                best_overall_c[k] *= -1
        
        T *= cooling_rate
        
        # Periodic local search every ~3 seconds
        if time.time() - last_local_search > 3.0 and time.time() < deadline - 25:
            best_overall_c, best_overall_val = local_search(
                best_overall_c, time.time() + 2.0)
            last_local_search = time.time()
    
    # Final local search
    best_overall_c, best_overall_val = local_search(best_overall_c, deadline - 10)
    
    # Try a few more from perturbed versions
    for _ in range(8):
        if time.time() > deadline - 8:
            break
        perturbed = best_overall_c.copy()
        for _ in range(np.random.randint(2, 5)):
            idx = np.random.randint(N)
            perturbed[idx] *= -1
        c_best, v_best = local_search(perturbed, time.time() + 1.5)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best
    
    # Final local search
    best_overall_c, best_overall_val = local_search(best_overall_c, deadline - 5)
    
    return [int(c) for c in best_overall_c]
