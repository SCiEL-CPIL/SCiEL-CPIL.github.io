"""
Spectral Optimization for Erdos Minimum Overlap.

Strategy:
1. Represent h via its Fourier coefficients.
2. The overlap function O(s) = N/2 - autocorr(s).
3. We want to minimize max_s O(s).
4. Use a smooth approximation of the maximum: (sum(exp(beta * O)))^(1/beta) or top-k sum.
5. Optimize the Fourier coefficients using L-BFGS-B.
   - The constraint sum(h) = N/2 is naturally handled if we fix c_0 = N/4 (since h is real and sum is N/2, mean is 1/2, so c_0 in standard FFT definition is N/2 * 1/2? 
     Let's check: h = ifft(c). sum(h) = N * c_0 if c_0 is the DC component in the "1/N" convention?
     numpy.fft.ifft: x = 1/N * sum(X_k * exp(...)). So sum(x) = X_0.
     We need sum(h) = N/2. So X_0 = N/2.
     Since h is real, X_0 is real. We fix X_0 = N/2.
     We optimize X_k for k=1..N/2.
     h is real if X_{N-k} = conj(X_k).
     We can parameterize by real and imaginary parts of X_1..X_{N/2}.
     X_{N/2} is real for even N.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _max_overlap_fft(h):
    """Max over all circular shifts of sum(h[i]*(1-h[(i+s)%N]))."""
    F = np.fft.fft(h)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    overlaps = N / 2.0 - autocorr
    return float(np.max(overlaps))

def _get_overlap_from_coeffs(coeffs, N):
    """
    Given Fourier coefficients (excluding DC), construct h and return overlaps.
    coeffs: array of shape (N//2, 2) or similar? 
    Let's use a flat vector of real parameters.
    For k=1..N/2-1: X_k = a_k + i b_k.
    For k=N/2: X_{N/2} = a_{N/2} (real).
    Total params: 2*(N/2 - 1) + 1 = N - 1.
    
    We fix X_0 = N/2.
    """
    X = np.zeros(N, dtype=complex)
    X[0] = N / 2.0
    
    # Map params to X
    # params[0], params[1] -> X[1]
    # params[2], params[3] -> X[2]
    # ...
    # params[N-2], params[N-1] -> X[N-1]? No, we only go up to N/2.
    
    # Let's define the structure clearly.
    # k = 1 to N//2 - 1: X[k] = p[2*k-2] + 1j*p[2*k-1]
    # k = N//2: X[N//2] = p[N-2] (real)
    # The rest are conjugates.
    
    h = np.zeros(N)
    
    # This is slow if done in a loop. Let's vectorize.
    # Actually, for N=100, a loop is fine, but let's try to be efficient.
    
    # Simpler: just fill X and ifft.
    for k in range(1, N // 2):
        idx = 2 * (k - 1)
        X[k] = coeffs[idx] + 1j * coeffs[idx + 1]
        X[N - k] = np.conj(X[k])
    
    if N % 2 == 0:
        X[N // 2] = coeffs[N - 2]
        # X[N - N//2] is X[N//2], so it's the same index.
        # Wait, for N=100, N//2 = 50.
        # X[50] is real.
        # The loop goes 1..49.
        # coeffs indices 0..97.
        # k=49: idx=96. X[49] = c[96] + i c[97]. X[51] = conj.
        # X[50] = c[98]? 
        # Total coeffs needed: 2*(N/2 - 1) + 1 = N - 1.
        # So coeffs has length N-1.
        # Index for X[N//2] is 2*(N/2 - 1) = N - 2.
        X[N // 2] = coeffs[N - 2]
    
    h = np.fft.ifft(X).real
    # Ensure h is in [0,1]? 
    # The problem requires h in [0,1]. 
    # If we optimize in Fourier space, h might go out of [0,1].
    # We can add a penalty or clip. Clipping breaks the Fourier relationship.
    # Better: add a penalty term to the objective for violations.
    
    F = np.fft.fft(h)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    overlaps = N / 2.0 - autocorr
    
    return overlaps, h

def _objective(params, beta=100.0):
    """
    Objective function for L-BFGS-B.
    params: Fourier coefficients (real and imaginary parts).
    """
    overlaps, h = _get_overlap_from_coeffs(params, N)
    
    # Penalty for h out of [0,1]
    penalty = 0.0
    if np.min(h) < 0 or np.max(h) > 1:
        penalty = np.sum(np.clip(h, None, 0)**2) + np.sum(np.clip(h - 1, None, 0)**2) * 100.0
    
    # Smooth max approximation
    # Use log-sum-exp for numerical stability
    m = np.max(overlaps)
    if m > 0:
        lse = m + np.log(np.sum(np.exp(overlaps - m))) / beta
    else:
        lse = 0.0
    
    return lse + penalty

def construct():
    deadline = time.time() + 115.0
    
    # Initial guess: h = 0.5 everywhere
    # This corresponds to X_0 = N/2, X_k = 0 for k>0.
    # So initial params are all zeros.
    n_params = N - 1
    x0 = np.zeros(n_params)
    
    best_score = 0.25 # Score for h=0.5
    best_h = np.full(N, 0.5)
    
    # We can try multiple starting points or betas.
    # Let's start with a small beta (smoother) and increase.
    
    # To escape local minima, we can add some random noise to initial Fourier coeffs.
    rng = np.random.default_rng(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Random initialization in Fourier space
        # Small amplitudes for higher frequencies
        x_try = rng.normal(0, 0.1, size=n_params)
        
        # Optimize
        try:
            res = minimize(
                _objective,
                x_try,
                args=(100.0,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
        except Exception:
            continue
            
        if time.time() > deadline:
            break
        
        # Evaluate result
        overlaps, h_candidate = _get_overlap_from_coeffs(res.x, N)
        
        # Check constraints
        if np.min(h_candidate) < -1e-6 or np.max(h_candidate) > 1 + 1e-6:
            # Clip to [0,1] and renormalize? 
            # Clipping might break sum= N/2.
            # Let's just skip if badly violated, or try to fix.
            # For now, if it's close, we can clip and renormalize.
            if np.min(h_candidate) < -0.1 or np.max(h_candidate) > 1.1:
                continue
            h_candidate = np.clip(h_candidate, 0, 1)
            # Renormalize to sum N/2
            s = h_candidate.sum()
            if s > 1e-6:
                h_candidate = h_candidate * (N / 2.0) / s
                # Re-clip
                h_candidate = np.clip(h_candidate, 0, 1)
                # Fix sum exactly
                diff = N / 2.0 - h_candidate.sum()
                if abs(diff) > 1e-9:
                    # Distribute to elements not at bounds
                    idxs = np.where((h_candidate > 1e-5) & (h_candidate < 1 - 1e-5))[0]
                    if len(idxs) > 0:
                        adj = diff / len(idxs)
                        h_candidate[idxs] += adj
                        h_candidate = np.clip(h_candidate, 0, 1)
        
        sc = _max_overlap_fft(h_candidate)
        
        if sc < best_score:
            best_score = sc
            best_h = h_candidate.copy()
            
        # If we found a good one, refine with higher beta
        if best_score < 0.24:
             # Refine with sharper beta
             try:
                 res2 = minimize(
                    _objective,
                    res.x,
                    args=(500.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12}
                 )
                 overlaps2, h_refined = _get_overlap_from_coeffs(res2.x, N)
                 if np.min(h_refined) >= -1e-6 and np.max(h_refined) <= 1 + 1e-6:
                     sc2 = _max_overlap_fft(h_refined)
                     if sc2 < best_score:
                         best_score = sc2
                         best_h = h_refined.copy()
             except Exception:
                 pass

    # Final validation
    if abs(best_h.sum() - N / 2.0) > 1e-8:
        # Emergency fix
        best_h = np.clip(best_h, 0, 1)
        s = best_h.sum()
        if s > 1e-6:
            best_h = best_h * (N / 2.0) / s
            best_h = np.clip(best_h, 0, 1)
            diff = N / 2.0 - best_h.sum()
            idxs = np.where((best_h > 1e-5) & (best_h < 1 - 1e-5))[0]
            if len(idxs) > 0:
                adj = diff / len(idxs)
                best_h[idxs] += adj
                best_h = np.clip(best_h, 0, 1)

    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_max_overlap_fft(h)}")
    print(f"Sum: {h.sum()}")
    print(f"Min: {h.min()}, Max: {h.max()}")
