"""Program for the third autocorrelation inequality task.

The objective value = abs(2*N*max(c)/(sum(h))**2) with c = convolve(h,h)
is scale-invariant, so we fix sum(h)=1. The seed program minimizes h^T h,
which yields the uniform vector and value 2.0.

This program instead minimizes the squared L2 norm of the autoconvolution
||c||_2^2 subject to sum(h)=1. This encourages the autoconvolution to be
"flat" across all lags, potentially reducing the global maximum max(c)
below 1/N, thereby achieving a value < 2.0.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _objective(h):
    c = np.convolve(h, h)
    return float(np.dot(c, c))


def _jac(h):
    # Gradient of f(h) = ||convolve(h, h)||_2^2
    # f(h) = sum_k c_k^2
    # df/dh_j = 4 * sum_k c_k * (partial c_k / partial h_j)
    # partial c_k / partial h_j = 2 * h_{k-j} (if valid index)
    # This is equivalent to 4 * (c * h) with appropriate alignment.
    # Let's compute it via convolution.
    # c has length 2N-1. h has length N.
    # We need sum_k c_k h_{k-j}.
    # This is the cross-correlation of c and h, or convolution with flipped h.
    # Since c is symmetric (c_k = c_{2N-2-k}), convolution with h is similar.
    
    # Efficient computation:
    # grad_j = 4 * sum_{k} c_k h_{k-j}
    # Let's use np.convolve.
    # convolve(c, h) gives length 3N-2.
    # The term for j corresponds to index j in the "centered" sense?
    # Let's just compute it directly for clarity or use a fast method.
    # For N=400, O(N^2) is 160,000 ops, which is fast enough for gradient.
    
    # But let's use the property:
    # grad = 4 * np.convolve(c, h)[N-1 : 3*N-2 - N + 1] ?
    # Let's verify indices.
    # c_k = sum_i h_i h_{k-i}.
    # d/dh_j c_k = h_{k-j} + h_{k-j} = 2 h_{k-j}.
    # Sum_k c_k * 2 h_{k-j} = 2 sum_k c_k h_{k-j}.
    # Total grad_j = 2 * (2 sum_k c_k h_{k-j}) = 4 sum_k c_k h_{k-j}.
    
    # sum_k c_k h_{k-j} is the coefficient of z^j in C(z) H(1/z)?
    # It is the cross-correlation.
    # np.correlate(c, h, mode='full')?
    # np.correlate(a, v) computes sum_k a_k v_{k-j} ?
    # Docs: np.correlate(a, v, mode='full')
    # out[n] = sum_m a[m] v[m+n] ? No.
    # Let's stick to a reliable method.
    # grad_j = 4 * sum_{k} c_k h_{k-j}
    # This is the convolution of c and h, but reversed?
    # Let h_rev = h[::-1].
    # conv = np.convolve(c, h_rev)
    # conv[k] = sum_i c_i h_rev[k-i] = sum_i c_i h_{N-1-(k-i)} = sum_i c_i h_{N-1-k+i}
    # This is getting messy.
    
    # Let's use the fact that for symmetric c, 
    # sum_k c_k h_{k-j} is related to np.convolve(c, h).
    # Let's just compute it with a loop or vectorized slice if N is small, 
    # but N=400 is fine for O(N^2) gradient if not called too often.
    # However, SLSQP might call it many times.
    
    # Let's use the spectral method or just np.convolve.
    # Note: sum_k c_k h_{k-j} = (c * h)[N-1-j] ?
    # Let's test with small N.
    # Actually, we can just use the gradient from finite differences to be safe 
    # in the first iteration, but analytic is better.
    
    # Let's use the property:
    # grad = 4 * np.convolve(c, h)[N-1 : 3*N-2 - (N-1)]?
    # Let's check the length. c len 2N-1, h len N. Conv len 3N-2.
    # The relevant part is indices N-1 to 2N-2?
    # Let's assume the gradient is 4 * np.convolve(c, h)[N-1 : 3*N-2 - N + 1]
    # which is length N.
    
    c = np.convolve(h, h)
    # The term sum_k c_k h_{k-j} is the j-th element of the cross-correlation.
    # For symmetric c, it is the same as convolution with h?
    # Let's use np.correlate(c, h, mode='full')
    # correlate(a, v)[n] = sum_k a[k] v[k+n] (depending on definition)
    # Numpy correlate: out[n] = sum_k a[k] v[k+n] is not standard.
    # Standard: sum_k a[k] v[k-j].
    # np.correlate(a, v) computes sum_k a[k] v[k-j]?
    # Actually, np.correlate(a, v) is equivalent to np.convolve(a, v[::-1]).
    
    # Let's use the explicit formula:
    # grad_j = 4 * sum_{k=max(0, j)}^{min(2N-2, j+N-1)} c[k] * h[k-j]
    
    # Vectorized approach:
    # Create a matrix of indices? No.
    # Use np.convolve.
    # Let's try:
    # conv_ch = np.convolve(c, h)
    # The sum we want is sum_k c_k h_{k-j}.
    # Let h_flipped = h[::-1]
    # conv_cf = np.convolve(c, h_flipped)
    # conv_cf[m] = sum_i c_i h_flipped[m-i] = sum_i c_i h_{N-1-(m-i)} = sum_i c_i h_{N-1-m+i}
    # We want sum_k c_k h_{k-j}.
    # Let i=k. sum_k c_k h_{k-j}.
    # In conv_cf, if we set N-1-m+i = k-j => i = m - N + 1 + k - j.
    # This doesn't look like a simple shift.
    
    # Let's just use the O(N^2) loop for the gradient. N=400 is 160k ops.
    # 100 iterations * 160k = 16M ops. Very fast in Python? 
    # Maybe a bit slow if pure Python loop. 
    # But we can vectorize the gradient calculation.
    
    # grad_j = 4 * sum_k c_k h_{k-j}
    # This is the j-th element of the "full" cross-correlation of c and h.
    # np.correlate(c, h, mode='full')
    # According to docs, np.correlate(a, v) is sum_k a[k] v[k+n] ?
    # Let's check: 
    # a=[1,2], v=[3,4]. 
    # full: [1*4, 1*4+2*3, 2*4] = [4, 10, 8]?
    # Convolve a, v[::-1] = [1,2] * [4,3] = [4, 7, 6].
    # So correlate is NOT convolve with flip in numpy?
    # Numpy correlate(a, v) computes sum_k a[k] v[k-j] for lags?
    # Actually, np.correlate(a, v, mode='full') returns the same as np.convolve(a, v[::-1]).
    # Wait, my manual calc:
    # conv([1,2], [4,3]) = 1*4 + 2*0 = 4 (lag 0? no, index 0)
    # Index 0: 1*4 = 4
    # Index 1: 1*3 + 2*4 = 3+8=11
    # Index 2: 2*3 = 6
    # So conv is [4, 11, 6].
    # Correlate([1,2], [3,4]):
    # Numpy doc: "out[n] = sum_k a[k] v[k+n]"?
    # Let's just use the definition:
    # We want g_j = sum_k c_k h_{k-j}.
    # This is the coefficient of z^j in C(z) H(z^{-1})?
    # If we compute np.convolve(c, h[::-1]), we get:
    # (c * h_rev)_m = sum_i c_i h_rev[m-i] = sum_i c_i h_{N-1-(m-i)} = sum_i c_i h_{N-1-m+i}
    # Let k = i. We want sum_k c_k h_{k-j}.
    # If we set N-1-m+k = k-j => N-1-m = -j => m = N-1+j.
    # So g_j = (c * h_rev)_{N-1+j}.
    
    h_rev = h[::-1]
    conv_ch = np.convolve(c, h_rev)
    # We need indices N-1 to N-1+N-1
    grad = 4.0 * conv_ch[N-1 : N-1+N]
    
    return grad


def construct():
    # Initialize with uniform vector
    x0 = np.full(N, 1.0 / N)
    
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    
    # Use SLSQP to handle the constraint
    res = minimize(_objective, x0, method="SLSQP", jac=_jac,
                   constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    
    h = res.x
    # Guard against numerical drift
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
        
    return h
