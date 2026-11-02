import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
h = 0.5 / N

def compute_score(a):
    """
    Compute the normalized maximum autoconvolution.
    Score = max(autoconv) / (h^2 * (sum(a))^2)
    This is scale-invariant.
    """
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    max_conv = np.max(conv)
    norm = h**2 * (np.sum(a))**2
    if norm == 0:
        return float('inf')
    return max_conv / norm

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill

    # 1. Baseline: Uniform distribution
    a_uniform = np.ones(N)
    score_uniform = compute_score(a_uniform)
    best_a = a_uniform.copy()
    best_score = score_uniform

    # 2. Try Two-Block Disjoint Rectangles (Inspiration 2 strategy)
    # This structure often performs well in autocorrelation peak minimization
    # by creating a "dip" in the central autocorrelation peak.
    # We optimize the gap size and block height/width.
    # Since the score is scale-invariant, we can fix the sum of a to 1.
    # Or just optimize the shape.
    
    def _construct_two_block(gap_ratio, block_height_ratio):
        """
        gap_ratio: fraction of the domain that is the central gap (0 to 1).
        block_height_ratio: relative height of blocks vs the gap (not used if gap is 0, but here gap is 0 height).
        Actually, let's define it as:
        a[i] = H for i in [0, W] and [N-W, N], 0 otherwise.
        We need to determine W and H.
        If we fix sum(a) = 1, then H * (2W) = 1 => H = 1/(2W).
        So we just need to optimize W (width of one block in number of intervals).
        """
        # gap_ratio from 0 to 0.9 means gap is 0 to 0.9 of the total width.
        # Total width N. Gap width G = gap_ratio * N.
        # Block width W = (N - G) / 2.
        G = int(gap_ratio * N)
        W = (N - G) // 2
        if W <= 0:
            return None, float('inf')
        
        a = np.zeros(N)
        a[:W] = 1.0
        a[N-W:] = 1.0
        # Normalize to sum=1 for consistent comparison? 
        # The score is scale invariant, so we don't strictly need to normalize, 
        # but it helps with numerical stability.
        s = np.sum(a)
        if s == 0: return None, float('inf')
        a = a / s
        sc = compute_score(a)
        return a, sc

    # Grid search for two-block
    best_two_block_a = None
    best_two_block_score = float('inf')
    
    # Search over gap sizes
    # A uniform function corresponds to gap=0 (if we interpret it that way, but uniform has no gap).
    # Actually, uniform is a single block of width N/2 on both sides? No, uniform is all 1s.
    # Two-block implies a gap.
    # Let's search gap from 0.1 to 0.8 of the domain.
    gap_ratios = np.linspace(0.05, 0.8, 50)
    
    for gr in gap_ratios:
        a, sc = _construct_two_block(gr, 1.0)
        if a is not None and sc < best_two_block_score:
            best_two_block_score = sc
            best_two_block_a = a.copy()
            if sc < best_score:
                best_score = sc
                best_a = a.copy()

    # 3. Spectral/IDFT Strategy (as per directive)
    # Represent f as |IDFT(C)| where C has unit modulus phases? 
    # Or optimize the amplitudes of the spectrum to minimize the time-domain peak.
    # This is effectively minimizing the L-infinity norm of a signal given its L2 norm (fixed by Parseval).
    # Minimize max_k (f*k) / (integral f)^2.
    
    # Let's try a local optimization starting from the best found so far.
    # We can use gradient descent or scipy minimize on the vector of heights a.
    # Since a must be non-negative, we can optimize over log(a + eps) or use constraints.
    # Better: optimize over unbounded vector z, and let a = exp(z).
    # This ensures a > 0.
    
    # Initial guess: log(best_a + 1e-9)
    # Objective: compute_score(exp(z))
    
    def _objective(z):
        if time.time() > deadline - 5:
            return float('inf')
        a = np.exp(z)
        # Scale invariance: we can fix sum(a) = 1 to prevent drift?
        # Actually, the score is invariant to scaling. 
        # But exp(z) might drift to huge/small values.
        # Let's normalize a to sum=1 inside the objective to stabilize.
        s = np.sum(a)
        if s == 0: return float('inf')
        a_norm = a / s
        return compute_score(a_norm)

    # Run local optimization from a few starting points
    # 1. From best two-block
    # 2. From uniform
    # 3. From a random smooth function
    
    starts = []
    if best_two_block_a is not None:
        starts.append(np.log(best_two_block_a + 1e-9))
    starts.append(np.log(a_uniform + 1e-9))
    
    # Add a random smooth start
    # Smooth function: low pass filter of random noise
    random_a = np.random.rand(N)
    # Simple smoothing
    kernel = np.ones(10)/10
    random_a = np.convolve(random_a, kernel, mode='same')
    random_a = random_a / np.sum(random_a)
    starts.append(np.log(random_a + 1e-9))

    for z0 in starts:
        if time.time() > deadline - 10:
            break
        try:
            # Use L-BFGS-B or COBYLA? L-BFGS-B is fast.
            # We need to be careful with the number of iterations.
            res = minimize(_objective, z0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success and res.fun < best_score:
                a_opt = np.exp(res.x)
                a_opt = a_opt / np.sum(a_opt)
                sc = compute_score(a_opt)
                if sc < best_score:
                    best_score = sc
                    best_a = a_opt
        except Exception as e:
            pass

    # 4. Differential Evolution for global search (if time permits)
    # This is expensive, so we only do it if we have > 30s left.
    if time.time() < deadline - 40:
        # Reduce dimensionality? 600 dims is a lot for DE.
        # Let's try DE on a lower-dimensional parameterization:
        # e.g., a as a linear combination of a few basis functions (Gaussians).
        
        def _basis_score(params):
            if time.time() > deadline - 10:
                return float('inf')
            # params: [mean1, sigma1, amp1, mean2, sigma2, amp2, mean3, sigma3, amp3]
            # 3 Gaussians
            x = np.linspace(-0.25, 0.25, N)
            a = np.zeros(N)
            for i in range(3):
                m = params[i*3]
                s = params[i*3+1]
                amp = params[i*3+2]
                a += amp * np.exp(-0.5 * ((x - m) / s)**2)
            a = np.abs(a) # Ensure non-negative
            s = np.sum(a)
            if s == 0: return float('inf')
            a = a / s
            return compute_score(a)

        # Bounds for parameters
        # means in [-0.25, 0.25]
        # sigmas in [0.01, 0.25]
        # amps in [0, 1]
        bounds = [(-0.25, 0.25), (0.01, 0.25), (0, 1)] * 3
        
        # Run DE with small population
        # To save time, we limit the number of iterations and population size.
        popsize = 5
        # DE default is popsize * len(bounds). 5 * 9 = 45 individuals.
        # Max iterations: let's try 20.
        # This might be too slow? 45*20 = 900 evaluations.
        # Each eval is O(N log N) ~ 6000 ops. 900 * 6000 = 5.4M ops. Should be fast.
        
        res_de = differential_evolution(_basis_score, bounds, maxiter=20, popsize=5, tol=1e-4, seed=42, polish=False)
        
        if res_de.success and res_de.fun < best_score:
            # Reconstruct a
            x = np.linspace(-0.25, 0.25, N)
            a = np.zeros(N)
            params = res_de.x
            for i in range(3):
                m = params[i*3]
                s = params[i*3+1]
                amp = params[i*3+2]
                a += amp * np.exp(-0.5 * ((x - m) / s)**2)
            a = np.abs(a)
            s = np.sum(a)
            if s > 0:
                a = a / s
                sc = compute_score(a)
                if sc < best_score:
                    best_score = sc
                    best_a = a

    return list(best_a)
