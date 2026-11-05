"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    """
    Construct step heights h of length N=400 to minimize:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    We use convex optimization over the autoconvolution structure.
    Key insight: by symmetry and normalization, we can fix sum(h)=N (or any nonzero)
    and minimize max(convolve(h,h)).
    
    We parameterize h in the Fourier domain to enforce that the autoconvolution
    is a valid Toeplitz/convolution structure, then optimize.
    
    Approach: 
    - Normalize so that sum(h) = 1 (scaling invariance: value is scale-invariant)
    - Minimize max(autoconv) subject to sum(h)=1
    - Use projected gradient descent or scipy.optimize
    
    Actually, let's think more carefully. The value is:
        V = |2N * max_k (h*h)[k] / (sum h)^2|
    
    This is scale-invariant: if we scale h by c, both numerator (c^2) and denominator (c^2) scale the same.
    So we can fix sum(h) = 1 without loss of generality (as long as sum(h) != 0).
    
    Then V = |2N * max_k (h*h)[k]|
    
    We want to minimize this. Note that (h*h)[0] = sum(h_i^2) >= (sum h)^2 / N = 1/N by Cauchy-Schwarz.
    So max_k (h*h)[k] >= (h*h)[0] >= 1/N, giving V >= 2N * 1/N = 2.
    
    Wait, but the seed gives 2.0 with h = ones. Let's verify:
    h = ones(400), sum(h) = 400, convolve(h,h) has entries [400, 399, 398, ..., 1, ..., 398, 399, 400]
    max = 400, so V = |2*400*400 / 400^2| = |2*400*400/160000| = |320000/160000| = 2.0. Yes.
    
    So the lower bound is 2.0? But can we do better?
    
    Actually wait: (h*h)[0] = sum h_i^2. If sum h = 1, then sum h_i^2 >= 1/N.
    So max(autoconv) >= sum h_i^2 >= 1/N.
    Thus V >= 2N * (1/N) = 2.
    
    So 2.0 is a lower bound! The seed achieves it. Can we achieve less than 2?
    
    Hmm, but the problem says "lower is better" and the seed gives 2.0. If 2.0 is a theoretical lower bound,
    then we can't do better. But maybe I'm missing something.
    
    Let me re-examine. The autoconvolution (h*h)[k] for k=0,...,2N-2 is:
    (h*h)[k] = sum_{i+j=k} h_i h_j for k < N (roughly, with appropriate indexing)
    
    Actually np.convolve(h, h) computes the full convolution. For k from 0 to 2N-2:
    (h*h)[k] = sum_{i=0}^{min(k, N-1)} h[i] * h[k-i] where indices are valid.
    
    At k=0: (h*h)[0] = h[0]*h[0] = h[0]^2... no wait.
    
    Let me be precise. np.convolve(h, h) where h has length N:
    Result has length 2N-1.
    Result[k] = sum_{i=max(0,k-N+1)}^{min(k, N-1)} h[i] * h[k-i]
    
    So Result[0] = h[0]*h[0]
    Result[1] = h[0]*h[1] + h[1]*h[0] = 2*h[0]*h[1]
    ...
    Result[N-1] = sum_{i=0}^{N-1} h[i]*h[N-1-i]
    Result[N] = sum_{i=1}^{N-1} h[i]*h[N-i]
    ...
    Result[2N-2] = h[N-1]*h[N-1]
    
    So the maximum of the autoconvolution is at least max(h[i]^2) for the diagonal-like terms, but also includes cross terms.
    
    With h = ones(N): 
    Result[k] = N - |k - (N-1)| for k=0..2N-2? Let's check:
    Result[0] = 1, Result[1] = 2, ..., Result[N-1] = N, Result[N] = N-1, ..., Result[2N-2] = 1.
    So max = N = 400. Sum(h) = 400. V = 2*400*400/400^2 = 2.
    
    Now, can we get V < 2? We need max(autoconv) * 2N / (sum h)^2 < 2.
    With sum h = 1 (normalized), we need max(autoconv) < 1/N = 1/400 = 0.0025.
    
    But (autoconv)[0] = h[0]^2, (autoconv)[2N-2] = h[N-1]^2.
    And (autoconv)[N-1] = sum h[i]*h[N-1-i].
    
    By Cauchy-Schwarz: (sum h[i]*h[N-1-i])^2 <= (sum h[i]^2)(sum h[N-1-i]^2) = (sum h[i]^2)^2.
    So |(autoconv)[N-1]| <= sum h[i]^2.
    
    Also, sum h[i]^2 >= (sum h)^2 / N = 1/N.
    
    So max(autoconv) >= h[0]^2, h[N-1]^2, and |autoconv[N-1]| <= sum h[i]^2 but autoconv[N-1] could be negative.
    
    The maximum (not absolute) of autoconv entries... we need max over signed entries.
    
    If we can make all autoconv entries small in magnitude, but the maximum (largest signed value) small...
    
    Actually, we just need the largest entry (most positive) to be small. Entries can be negative.
    
    So we want to minimize max_k (h*h)[k] subject to sum h = 1.
    
    We know (h*h)[0] = h[0]^2 >= 0 and (h*h)[2N-2] = h[N-1]^2 >= 0.
    So max(autoconv) >= max(h[0]^2, h[N-1]^2, ..., and other positive entries).
    
    To minimize max(autoconv), we'd want all h[i] small, but sum h = 1 forces some to be positive.
    
    If we spread the mass evenly: h[i] = 1/N for all i, then autoconv[k] = (N-|k-(N-1)|)/N^2, max = 1/N.
    Then V = 2N * (1/N) / 1^2 = 2. Same as before.
    
    What if we use alternating signs? h[i] = (-1)^i / sqrt(N)? Then sum h might be 0 or small.
    
    Let me try: h[i] = (-1)^i. Sum = 0 (for even N). Not allowed.
    h[i] = (-1)^i + eps? Sum = eps (for N even). 
    autoconv of (-1)^i: (h*h)[k] = sum (-1)^i (-1)^{k-i} = sum (-1)^k = N*(-1)^k for k < N.
    So max = N. V = 2N*N/eps^2 which is huge.
    
    Let me try a different approach. What if we use a smooth function?
    h[i] = exp(-alpha * i^2 / N^2) normalized to sum 1?
    
    Actually, let me just do numerical optimization. Fix sum(h) = 1, minimize max(autoconv(h)).
    
    I'll use scipy.optimize with a smooth approximation of max (soft max) or just use the fact that
    we can do projected gradient descent.
    
    Let me try: minimize f(h) = max_k (h*h)[k] subject to sum(h) = 1.
    
    Use scipy.optimize.minimize with SLSQP or similar, but 400 variables might be slow.
    
    Alternative: use Fourier domain. The autoconvolution in Fourier domain is |H|^2 where H is the DFT.
    But max in time domain is hard to express in frequency.
    
    Let me just try a simple numerical optimization with scipy.
    
    """
    # Normalize: fix sum(h) = 1
    # Minimize max(autoconv(h))
    
    # Use a smooth approximation: soft_max(x, t) = (1/t) * log(sum(exp(t*x)))
    # Or just use the actual max and rely on scipy's handling.
    
    # Let's try with scipy.optimize.minimize using the 'SLSQP' method
    # Objective: max of autoconv(h)
    # Constraint: sum(h) = 1
    
    def objective(h):
        ac = np.convolve(h, h)
        return np.max(ac)
    
    def constraint_sum(h):
        return np.sum(h) - 1.0
    
    # Initial guess: uniform
    h0 = np.ones(N) / N
    
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    # Try to minimize
    try:
        result = minimize(objective, h0, method='SLSQP', constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-10})
        h_opt = result.x
        # Ensure sum is exactly 1
        h_opt = h_opt / np.sum(h_opt)
        val = abs(2 * N * np.max(np.convolve(h_opt, h_opt)) / np.sum(h_opt)**2)
        if val < 2.0:
            return h_opt
    except Exception:
        pass
    
    # If optimization didn't help, return seed
    return np.ones(N)
