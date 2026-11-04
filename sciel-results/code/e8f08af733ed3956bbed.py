"""Second autocorrelation inequality: maximize ratio for 50 nonnegative step heights."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
H = 1.0 / N  # step height in the interval [-1/4, 1/4]

def _ratio(h):
    """Compute the ratio: (L2 norm)^2 / (L1 norm * sup norm of autoconvolution).
    
    For a step function with heights h_i on intervals of width H:
    - L2 norm squared = sum(h_i^2) * H
    - L1 norm = sum(h_i) * H
    - autoconvolution is a piecewise linear function; its sup norm is max over the convolution
    
    The autoconvolution of a step function f(x) = sum h_i * 1_{[x_i, x_i+H)}(x)
    is (f*f)(t) = integral f(x) f(t-x) dx.
    
    For step functions on a grid, the autoconvolution at the grid points (centers or edges)
    can be computed. The supremum of the autoconvolution occurs at the center of the 
    convolution support. For a step function, the autoconvolution is a piecewise linear 
    function that peaks at t where the overlap is maximized.
    
    Actually, let me think more carefully. The autoconvolution of a step function with 
    heights h_0, ..., h_{N-1} on intervals [i*H, (i+1)*H) for i=0,...,N-1 (shifted to 
    [-1/4, 1/4]) is:
    
    (f*f)(t) = sum_i sum_j h_i h_j * length of intersection of [i*H, (i+1)*H) and [(t-(j+1)*H), t-j*H)
    
    This is a piecewise linear function. Its maximum occurs at one of the "corner" points 
    where t equals i*H for some integer i (shifted appropriately).
    
    Let's use the standard discrete approximation: if we sample the autoconvolution at 
    the midpoints of the intervals, we get the discrete autocorrelation. But the true 
    supremum of the continuous autoconvolution is larger.
    
    Actually, for a step function, the autoconvolution is a "tent-like" piecewise linear 
    function. The maximum of the continuous autoconvolution is the maximum over all 
    positions t of the integral. This maximum is achieved at t = k*H for some integer k, 
    and at those points, (f*f)(k*H) = H * sum_i h_i * h_{i+k} (with appropriate indexing).
    
    Wait, let me be more precise. Let f(x) = h_i for x in [i*H, (i+1)*H), i = 0,...,N-1.
    Then (f*f)(t) = integral f(x) f(t-x) dx.
    
    For t = m*H (integer multiple of H), the overlap of [i*H, (i+1)*H) with [(t-(j+1)*H), t-j*H) = [(m-j-1)*H, (m-j)*H) is:
    - Full overlap (length H) if the intervals align exactly, which happens when i = m-j-1... 
    Actually this is getting complicated. Let me just compute it numerically.
    
    The autoconvolution at t = m*H: the contribution from pair (i,j) is h_i * h_j * 
    length([i*H,(i+1)*H) ∩ [(m-j-1)*H, (m-j)*H)).
    
    The second interval is [(m-j-1)*H, (m-j)*H). The first is [i*H, (i+1)*H).
    These overlap with length H if i = m-j-1, i.e., j = m-1-i.
    
    So (f*f)(m*H) = H * sum_i h_i * h_{m-1-i} where the index m-1-i must be in [0, N-1].
    
    But the supremum might be higher at non-integer multiples of H. However, for a 
    nonnegative step function, the autoconvolution is a concave piecewise linear function 
    in each "diagonal strip", and its maximum over the whole domain is actually achieved 
    at t = (N-1)*H/2 + ... hmm.
    
    Let me just compute the autoconvolution on a fine grid and take the max.
    """
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return 0.0
    
    l2_sq = np.sum(h ** 2) * H
    l1 = np.sum(h) * H
    
    if l1 <= 0:
        return 0.0
    
    # Compute autoconvolution on a fine grid
    # The support of f is [0, N*H) (shifted), so f*f has support [0, 2*N*H)
    # Let's compute on a grid with resolution H/4 for accuracy
    n_grid = 4 * N
    dt = 2.0 * N * H / n_grid
    t = np.arange(n_grid) * dt
    
    # f on the grid
    # f(x) = h_i for x in [i*H, (i+1)*H)
    # For x in [0, N*H), i = floor(x/H)
    x = t  # t ranges from 0 to 2*N*H
    f_vals = np.zeros_like(t)
    for i in range(N):
        mask = (x >= i * H) & (x < (i + 1) * H)
        f_vals[mask] = h[i]
    
    # Autoconvolution: (f*f)(t) = integral f(x) f(t-x) dx
    # Use numerical integration: sum f(x_k) f(t - x_k) dx
    # We need to evaluate f at t - x_k for each k
    # This is O(N^2 * n_grid) which might be slow. Let's use FFT-based convolution.
    
    # Pad f to avoid circular convolution
    f_pad = np.zeros(2 * n_grid)
    f_pad[:n_grid] = f_vals
    
    # Convolution via FFT
    conv = np.fft.irfft(np.fft.rfft(f_pad) ** 2, n=2 * n_grid) * dt
    
    # The autoconvolution is conv, but we only care about t in [0, 2*N*H)
    sup_conv = np.max(conv[:n_grid])
    
    if sup_conv <= 0:
        return 0.0
    
    return l2_sq / (l1 * sup_conv)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Strategy: concentrate mass. Try various spike profiles and optimize.
    
    # Initial candidates:
    candidates = []
    
    # Single spike
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # Two adjacent spikes
    for pos in range(N - 1):
        h = np.zeros(N)
        h[pos] = 1.0
        h[pos + 1] = 1.0
        candidates.append(h)
    
    # Gaussian-like profiles
    for center in [N // 2, N // 2 - 1, N // 2 + 1]:
        for width in [1, 2, 3, 5, 8]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width) ** 2)
            candidates.append(h)
    
    # Triangular profiles
    for center in [N // 2, N // 2 - 1, N // 2 + 1]:
        for width in [5, 10, 15, 20, 25]:
            x = np.arange(N)
            h = np.maximum(0, 1 - np.abs(x - center) / width)
            candidates.append(h)
    
    # Evaluate initial candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization using Nelder-Mead
    def neg_ratio(h_flat):
        h = np.maximum(0, h_flat)  # enforce nonnegativity
        return -_ratio(h)
    
    # Optimize from the best initial candidate
    if time.time() < deadline:
        res = minimize(neg_ratio, best_h, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < -best_score:
            best_h = np.maximum(0, res.x)
            best_score = -res.fun
    
    # Try optimizing from several other starting points
    for h0 in candidates[:10]:
        if time.time() > deadline:
            break
        # Normalize to have max 1
        h0_norm = h0 / np.max(h0) if np.max(h0) > 0 else h0
        res = minimize(neg_ratio, h0_norm, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < -best_score:
            best_h = np.maximum(0, res.x)
            best_score = -res.fun
    
    # Try some random restarts
    rng = np.random.RandomState(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        h_rand = rng.random(N)
        # Make it sparse-ish
        h_rand[h_rand < 0.5] = 0
        h_rand = h_rand / (np.max(h_rand) + 1e-10)
        res = minimize(neg_ratio, h_rand, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < -best_score:
            best_h = np.maximum(0, res.x)
            best_score = -res.fun
    
    # Final refinement
    if time.time() < deadline:
        res = minimize(neg_ratio, best_h, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < -best_score:
            best_h = np.maximum(0, res.x)
            best_score = -res.fun
    
    # Ensure nonnegative
    best_h = np.maximum(0, best_h)
    
    return list(best_h)


if __name__ == "__main__":
    result = construct()
    print("Score:", _ratio(result))
    print("Heights:", result)
