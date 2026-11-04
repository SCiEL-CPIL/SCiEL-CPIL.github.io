import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

N = 50
w = 1.0 / N

def _autoconv_peak(h):
    """
    Compute the supremum norm of the autoconvolution of a piecewise-constant
    function defined by heights h on N equal intervals over [-1/4, 1/4].
    """
    # Autocorrelation of h: u_k = sum_{i+j=k} h_i h_j
    # h is length N. u will be length 2N-1.
    # Using np.convolve(h, h[::-1], mode='full') gives correlation?
    # np.correlate(h, h, mode='full') gives c_k = sum_n h[n] h[n+k].
    # We want sum_{i+j=k} h_i h_j.
    # Let's just use the direct sum for small N or FFT.
    # For N=50, direct sum is 2500 ops. Very fast.
    
    u = np.zeros(2*N - 1)
    for i in range(N):
        for j in range(N):
            u[i+j] += h[i] * h[j]
            
    # Kernel K_m = max(0, 1 - |m|) for m in range(-(2N-2), 2N-1)?
    # The index m in V_m = sum_k u_k K_{m-k} goes from 0 to 2N-2.
    # k goes from 0 to 2N-2.
    # m-k goes from -(2N-2) to 2N-2.
    # K is defined for integer shifts.
    # K_d = max(0, 1 - |d|).
    
    # We can construct the kernel for the convolution.
    # We want V = convolve(u, K, mode='valid' or 'same'?)
    # V_m = sum_k u_k K_{m-k}.
    # This is the convolution of u and K.
    # Length of u is 2N-1.
    # We need V for m=0..2N-2.
    # The "full" convolution of u (len 2N-1) and K (len 2N-1, indices -N+1..N-1?)
    # Let's define K array explicitly.
    # d ranges from -(2N-2) to 2N-2.
    # But K_d is 0 for |d| >= 1? No, K_d = max(0, 1-|d|).
    # K_0 = 1.
    # K_1 = 0.
    # K_{-1} = 0.
    # Wait, 1 - |1| = 0.
    # So the kernel is just a delta function?
    # Let's re-read the triangle function definition.
    # Triangle function $\text{tri}(t) = \max(0, 1-|t|)$.
    # Support is $[-1, 1]$.
    # $t = \frac{x - (x_i+x_j)}{w}$.
    # If $x = x_p + x_q$, then $t = \frac{x_p+x_q - (x_i+x_j)}{w} = \frac{(p-i)w + (q-j)w}{w} = (p-i) + (q-j)$.
    # Let $d = (p-i) + (q-j)$.
    # $\text{tri}(d) = \max(0, 1 - |d|)$.
    # If $d=0$, value is 1.
    # If $d=1$, value is 0.
    # If $d=-1$, value is 0.
    # So the kernel is indeed non-zero ONLY at $d=0$.
    # This implies that the autoconvolution at the centers $x_p+x_q$ receives contribution ONLY from pairs $(i,j)$ such that $i+j = p+q$.
    # And the value is $\sum_{i+j=p+q} h_i h_j \cdot 1 \cdot w$.
    # So the peak is simply $w \cdot \max_{m} \sum_{i+j=m} h_i h_j$.
    # This is $w \cdot \max_{m} u_m$.
    
    # Is this correct?
    # The triangle function has support $[-1, 1]$.
    # The distance between adjacent centers $x_i$ is $w$.
    # So the distance between adjacent sum centers $x_i+x_j$ is $w$.
    # The triangle from pair $(i,j)$ has support $[x_i+x_j-w, x_i+x_j+w]$.
    # It touches the adjacent sum centers $x_{i+1}+x_j$ and $x_i+x_{j+1}$ at 0.
    # So yes, at the grid points of the sums, the value is determined ONLY by the pairs that sum to that exact index.
    # The "tail" of the triangle from $(i,j)$ does not reach the next grid point $(i+1)+j$ because the distance is exactly $w$, and the triangle goes to 0 at distance $w$.
    
    # So, $\|\text{autoconv}\|_\infty = w \cdot \max_{m=0}^{2N-2} \sum_{i+j=m} h_i h_j$.
    
    max_u = np.max(u)
    return w * max_u

def construct():
    # Objective function to minimize: - ratio
    def objective(h_flat):
        h = np.maximum(0, h_flat)
        l1_sum = np.sum(h)
        if l1_sum < 1e-12:
            return 1e6
        l2_sq_sum = np.sum(h**2)
        
        # Compute peak
        # u_m = sum_{i+j=m} h_i h_j
        # We can compute this efficiently
        # But for N=50, simple loop is fine inside objective?
        # Or precompute? No, h changes.
        
        # Fast way:
        # u = np.convolve(h, h, mode='full') is not correct for sum_{i+j=m}.
        # np.convolve(h, h[::-1], mode='full')?
        # Let's just use the loop or np.einsum if needed.
        # Actually, we can use the fact that u is the autocorrelation.
        # But since we just need the max, and N is small:
        
        u = np.zeros(2*N - 1)
        # Vectorized update?
        # h_i h_j added to u[i+j]
        # This is an outer product summed along diagonals.
        # We can use scipy.ndimage or just a loop.
        # Loop is 2500 iterations.
        for i in range(N):
            u[i:i+N] += h[i] * h
            
        peak_val = w * np.max(u)
        
        if peak_val < 1e-12:
            return 1e6
            
        ratio = (l2_sq_sum * w) / (l1_sum * w * peak_val)
        # Wait, ratio = (w * sum h^2) / (w * sum h * w * max_u) = sum h^2 / (w * sum h * max_u)
        # My previous derivation:
        # R = (w * sum h^2) / ( (w * sum h) * (w * max_u) ) = sum h^2 / (w * sum h * max_u)
        
        return -ratio

    # Initial guess
    h0 = np.zeros(N)
    h0[N // 2] = 1.0
    
    bounds = [(0, None)] * N
    
    # Use trust-constr
    try:
        res = minimize(objective, h0, method='trust-constr', bounds=bounds,
                       options={'maxiter': 1000, 'gtol': 1e-8})
        h_opt = np.maximum(0, res.x)
    except Exception:
        # Fallback to L-BFGS-B if trust-constr fails or is too slow
        try:
            res = minimize(objective, h0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-8})
            h_opt = np.maximum(0, res.x)
        except Exception:
            h_opt = h0
            
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        h_opt = h0
        
    return h_opt
