"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a signed step function with 400 intervals to minimize the
    upper bound on C3 (third autocorrelation inequality).
    
    The score is the upper bound on C3 (lower is better).
    We use a Legendre symbol sequence modulo a prime near 400, then refine
    with local search to minimize the maximum autocorrelation peak.
    """
    t0 = time.time()
    deadline = t0 + 115  # 115s safety margin before 120s kill
    
    # Start with a Legendre symbol sequence modulo p=401
    # Legendre symbol (n/p) for n=0..N-1
    p = 401  # prime close to 400
    x = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if i == 0:
            x[i] = 0.0  # Legendre symbol of 0 is 0, but we need ±1
            x[i] = 1.0  # assign +1 for index 0
        else:
            # Compute Legendre symbol (i/p)
            ls = pow(i, (p - 1) // 2, p)
            if ls == 1:
                x[i] = 1.0
            else:
                x[i] = -1.0
    
    def compute_score(x):
        """Compute the upper bound on C3 from the signed step function."""
        # Autoconvolution: (x * x)(k) = sum_i x[i] * x[i-k]
        # The autocorrelation is the full convolution
        # For a step function, the autoconvolution at lag k is sum over overlaps
        # 
        # The "upper bound on C3" likely relates to the maximum of the
        # autoconvolution (excluding the zero-lag peak) or a specific combination.
        # 
        # From the context of "third autocorrelation inequality", C3 is likely
        # related to the third moment or a bound involving autocorrelation values.
        # 
        # A common formulation: C3 = max over k != 0 of |autocorr(k)| / autocorr(0)
        # or C3 = max_k autocorr(k) (the peak sidelobe level)
        # 
        # Let's compute the full autocorrelation and take the maximum sidelobe.
        ac = np.correlate(x, x, mode='full')
        # ac has length 2*N-1, centered at N-1
        # autocorr(0) = ac[N-1] = N (since sum x[i]^2 = N for ±1 values)
        # Sidelobes are all other values
        center = N - 1
        sidelobes = np.concatenate([ac[:center], ac[center+1:]])
        return np.max(np.abs(sidelobes))
    
    # Initial score
    best_x = x.copy()
    best_score = compute_score(best_x)
    
    # Local search: flip individual bits to improve score
    def try_flip(x, idx):
        """Try flipping x[idx] and return new score if improved."""
        old_val = x[idx]
        x[idx] = -old_val
        s = compute_score(x)
        x[idx] = old_val
        return s
    
    # Iterative local search
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        for i in range(N):
            if time.time() > deadline:
                break
            new_score = try_flip(best_x, i)
            if new_score < best_score - 1e-12:
                best_x[i] = -best_x[i]
                best_score = new_score
                improved = True
    
    # Second pass: try pairs of flips if time permits
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(i + 1, min(i + 20, N)):
                    if time.time() > deadline:
                        break
                    old_i, old_j = best_x[i], best_x[j]
                    best_x[i] = -old_i
                    best_x[j] = -old_j
                    new_score = compute_score(best_x)
                    if new_score < best_score - 1e-12:
                        best_score = new_score
                        improved = True
                    else:
                        best_x[i] = old_i
                        best_x[j] = old_j
    
    return best_x
# EVOLVE-BLOCK-END
