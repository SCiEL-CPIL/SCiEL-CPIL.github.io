import numpy as np
import time
import itertools

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    # h*h has length 2N-1
    # Zero pad to power of 2 for FFT efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = -np.inf
    
    # Strategy:
    # Instead of searching for sparse supports (Sidon sets), we parameterize the full vector h
    # as a non-negative combination of a small set of basis functions.
    # h = B @ c, where c >= 0 and B is an MxK matrix of basis vectors.
    # We use DCT-like bases (cosines) and Fourier modes.
    # The goal is to find a configuration where the autoconvolution is as "flat" as possible,
    # maximizing the ratio of energy (L2^2) to peak autoconvolution.
    #
    # Objective: Maximize f(c) = ||B c||_2^2 / (||B c||_1 * ||(B c)*(B c)||_inf)
    # Note: The ratio is scale invariant. We can fix ||h||_1 = 1 or just normalize c.
    # Let's fix sum(h) = 1.
    
    # Generate basis sets
    # Basis 1: Cosines (DCT-I like)
    # Basis 2: Sines (DCT-II like)
    # Basis 3: Fourier modes (real and imaginary parts)
    # Basis 4: Polynomials (Legendre)
    
    def generate_bases():
        bases = []
        
        # DCT-I basis: cos(pi * k * n / (N-1))
        n = np.arange(N)
        for K in [2, 3, 4, 5, 6, 8, 10]:
            B = np.zeros((N, K))
            for k in range(K):
                # Use frequencies that are integer multiples to ensure orthogonality-ish properties
                # For DCT-I, frequencies are 0, 1, ..., K-1
                B[:, k] = np.cos(np.pi * k * n / (N - 1))
            bases.append(B)
            
        # Fourier modes: cos(2 pi k n / N), sin(2 pi k n / N)
        for K in [2, 3, 4, 5, 6]:
            B = np.zeros((N, 2*K))
            for k in range(K):
                B[:, 2*k] = np.cos(2 * np.pi * k * n / N)
                B[:, 2*k+1] = np.sin(2 * np.pi * k * n / N)
            bases.append(B)
            
        # Low-frequency dominant basis
        # h_i ~ 1 + eps * cos(2 pi n / N)
        # This is effectively K=2 Fourier
        
        # "Bump" basis: Gaussian-like bumps at different centers
        # Might not be orthogonal but could help with local adjustments
        # We'll stick to global orthogonal bases for now as per strategy
        
        return bases

    all_bases = generate_bases()
    
    # Optimization for a fixed basis B
    # We want to maximize R(c) = ||B c||_2^2 / (||B c||_1 * ||h*h||_inf) subject to c >= 0.
    # Since R is scale invariant, we can constrain sum(c) = 1 (or sum(Bc)=1).
    # Let's use sum(c) = 1 for simplicity in the simplex.
    # Actually, sum(h) = 1 is more natural for the L1 norm in the denominator.
    # Let's just optimize c on the simplex sum(c)=1, c>=0.
    
    def optimize_coefficients(B, max_iter=2000, tol=1e-8):
        K = B.shape[1]
        # Initialize c
        c = np.ones(K) / K
        
        best_c = c.copy()
        best_score = -np.inf
        
        # Precompute B^T B and B^T 1? 
        # h = B c
        # L1 = sum(|B c|) -- since we want h>=0, we must ensure B c >= 0.
        # This is a constraint! c must be in the cone {c | B c >= 0}.
        # This is tricky. Simplex projection doesn't guarantee B c >= 0.
        
        # Alternative: Parameterize h directly? No, that's high dim.
        # Alternative: Use a soft constraint or just clip h?
        # If we clip h = max(B c, 0), the mapping is non-smooth.
        
        # Better strategy:
        # 1. Sample random c >= 0.
        # 2. Compute h = B c.
        # 3. If h has negative values, reject or project?
        #    Projection onto {c | B c >= 0} is a QP.
        #    Maybe we can choose bases that are non-negative?
        #    Cosines are not non-negative.
        
        # Let's try a different parameterization:
        # h = D @ a, where D is a matrix of non-negative basis functions.
        # e.g., h_i = sum_k a_k * exp(-beta_k * (i - mu_k)^2)
        # Or simply: h is a polynomial of degree K-1 with positive coefficients?
        # Or: h_i = sum_{k=0}^{K-1} c_k * i^k ?
        
        # Let's restart the basis idea with NON-NEGATIVE bases.
        # Basis: Powers of i (monomials) i^k for k=0..K-1.
        # Basis: B-splines?
        # Basis: Gaussian bumps?
        
        # Let's try Monomials: h_i = c_0 + c_1 i + c_2 i^2 ...
        # This ensures smoothness.
        
        pass

    # Refined Strategy:
    # Use a set of non-negative basis functions.
    # 1. Monomials: i^k
    # 2. Gaussian bumps: exp(-(i-mu)^2 / sigma^2)
    # 3. Cosine bumps: (1 + cos(2 pi (i-mu)/N)) / 2  (always non-negative)
    
    def generate_nonnegative_bases():
        bases = []
        n = np.arange(N)
        
        # Monomials
        for K in [2, 3, 4, 5, 6, 8]:
            B = np.zeros((N, K))
            for k in range(K):
                B[:, k] = (n / (N - 1))**k  # Normalized to [0,1]
            bases.append(B)
            
        # Cosine bumps (Gabor-like)
        # Center at various positions, various widths
        centers = [N//4, N//2, 3*N//4]
        widths = [5, 10, 20]
        
        for K in [2, 3, 4]:
            # Select K random centers/widths? Or fixed grid?
            # Let's use a grid of centers
            B = np.zeros((N, 0))
            cols = []
            for mu in centers:
                for sig in widths:
                    col = np.exp(-((n - mu)**2) / (2 * sig**2))
                    cols.append(col)
            B = np.column_stack(cols)
            bases.append(B)
            
        # Mixed: Constant + Cosine bumps
        B_const = np.ones((N, 1))
        for K in [2, 3, 4]:
             # Add K cosine bumps to constant
             cols = [B_const[:,0]]
             for mu in np.linspace(N//4, 3*N//4, K):
                 col = 0.5 * (1 + np.cos(2 * np.pi * (n - mu) / N))
                 cols.append(col)
             B = np.column_stack(cols)
             bases.append(B)
             
        return bases

    all_bases = generate_nonnegative_bases()
    
    # Optimization loop
    for B in all_bases:
        if time.time() > deadline:
            break
            
        K = B.shape[1]
        
        # We want to maximize R(c) = ||B c||_2^2 / (||B c||_1 * ||h*h||_inf)
        # subject to c >= 0.
        # Scale invariance allows us to fix sum(c) = 1.
        
        # Use a simple coordinate ascent / hill climbing on the simplex.
        # Start with uniform c.
        c = np.ones(K) / K
        
        best_c = c.copy()
        best_score = -np.inf
        
        # Precompute B^T B for L2 norm?
        # ||B c||_2^2 = c^T (B^T B) c
        BTB = B.T @ B
        
        # L1 norm is sum(|B c|). Since B >= 0 and c >= 0, B c >= 0.
        # So ||B c||_1 = sum(B c) = c^T (B^T 1).
        B_ones = B.T @ np.ones(N)
        
        # Autoconvolution peak is the hard part.
        
        def get_score(c):
            h = B @ c
            l1 = np.sum(h)
            if l1 < 1e-10: return -np.inf
            l2_sq = np.sum(h*h)
            # FFT for conv
            n_fft = 1 << (2 * N - 1).bit_length()
            H = np.fft.rfft(h, n=n_fft)
            hh = np.fft.irfft(H * H, n=n_fft)
            hh_max = np.max(hh)
            if hh_max < 1e-10: return -np.inf
            return l2_sq / (l1 * hh_max)
        
        # Optimization
        T = 0.1
        for it in range(1000):
            if time.time() > deadline:
                break
                
            # Perturb c
            c_new = c.copy()
            # Random direction in simplex
            # c_new += noise, then project to simplex
            # Simple projection: subtract min, normalize? No, that's for Euclidean distance to simplex.
            # For c >= 0, sum(c)=1:
            # Just add noise, clip to 0, normalize.
            
            c_new += rng.normal(0, T, size=K)
            c_new = np.maximum(c_new, 0)
            s = np.sum(c_new)
            if s < 1e-10:
                continue
            c_new /= s
            
            score = get_score(c_new)
            
            if score > best_score:
                best_score = score
                best_c = c_new.copy()
                
            # Accept/Reject
            # If score is better, accept.
            # If worse, accept with prob exp((score - current_score)/T)
            # We need current_score
            if it > 0:
                pass # We didn't store current score in the loop variable properly for SA
            # Let's just do greedy ascent with occasional restarts
            
            if score >= best_score - 1e-5: # Accept if close to best or better
                c = c_new
            else:
                # With some probability, move towards best
                if rng.random() < 0.1:
                    c = best_c + rng.normal(0, T*0.5, size=K)
                    c = np.maximum(c, 0)
                    s = np.sum(c)
                    if s > 0: c /= s
                else:
                    # Keep current c, maybe reduce T
                    T *= 0.95
        
        # After local search, compute final h
        h_candidate = B @ best_c
        # Ensure non-negative
        h_candidate = np.maximum(h_candidate, 0)
        
        # Normalize? The ratio is scale invariant, so it doesn't matter, 
        # but let's return the raw heights.
        
        final_score = get_score(best_c)
        if final_score > best_score_global if 'best_score_global' in locals() else False:
             pass
            
        # Wait, I used best_score locally. Let's track global.
        
        if best_score > (best_score_global if 'best_score_global' in locals() else -np.inf):
            # Update global best
            if 'best_h' not in locals() or best_score > best_score_global:
                best_score_global = best_score
                best_h = h_candidate.copy()
            else:
                best_score_global = best_score
                best_h = h_candidate.copy()
        elif 'best_score_global' not in locals():
            best_score_global = best_score
            best_h = h_candidate.copy()
            
        # Simpler global tracking
        # Let's refactor the loop to be cleaner.
        
    # The code above got messy with local/global variables. Let's rewrite the main loop cleanly.
    
    # RESTART CLEAN IMPLEMENTATION
    
    best_h_global = None
    best_score_global = -np.inf
    
    for B in all_bases:
        if time.time() > deadline:
            break
            
        K = B.shape[1]
        
        # Local optimization
        c = np.ones(K) / K
        best_c_local = c.copy()
        best_score_local = -np.inf
        T = 0.1
        
        def eval_c(c_vec):
            h = B @ c_vec
            l1 = np.sum(h)
            if l1 < 1e-10: return -np.inf
            l2_sq = np.sum(h*h)
            n_fft = 1 << (2 * N - 1).bit_length()
            H = np.fft.rfft(h, n=n_fft)
            hh = np.fft.irfft(H * H, n=n_fft)
            hh_max = np.max(hh)
            if hh_max < 1e-10: return -np.inf
            return l2_sq / (l1 * hh_max)
        
        for it in range(2000):
            if time.time() > deadline:
                break
                
            # Perturb
            c_new = c + rng.normal(0, T, size=K)
            c_new = np.maximum(c_new, 0)
            s = np.sum(c_new)
            if s < 1e-10: continue
            c_new /= s
            
            score = eval_c(c_new)
            
            if score > best_score_local:
                best_score_local = score
                best_c_local = c_new.copy()
            
            # Greedy acceptance with noise
            # If score is close to best, accept.
            if score > best_score_local - 1e-6:
                c = c_new
            else:
                # Move towards best
                c = best_c_local + rng.normal(0, T*0.1, size=K)
                c = np.maximum(c, 0)
                s = np.sum(c)
                if s > 0: c /= s
            
            # Anneal
            T *= 0.99
            
        # End local optimization
        h_candidate = B @ best_c_local
        h_candidate = np.maximum(h_candidate, 0)
        
        # Verify score with original compute_ratio to be safe
        final_score = compute_ratio(h_candidate)
        
        if final_score > best_score_global:
            best_score_global = final_score
            best_h_global = h_candidate.copy()
            
    # Fallback
    if best_h_global is None:
        best_h_global = np.ones(N) / N
        
    return list(best_h_global)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
