"""Rewritten program using a low-dimensional Fourier basis with smooth log-sum-exp surrogate."""
N = 400

import numpy as np
import time


def _full_autoconvolve_from_fourier(h):
    """
    Compute the full autoconvolution of h via FFT.
    h has length N.
    Full convolution length is 2N - 1.
    We zero-pad to length 2N for FFT.
    """
    # Zero pad h to length 2N
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    # FFT
    H = np.fft.fft(h_pad)
    # Convolution in frequency domain
    conv_freq = H * H
    # IFFT
    conv = np.fft.ifft(conv_freq).real
    # The full convolution is the first 2N-1 elements
    # Note: np.convolve(h, h, mode='full') gives length 2N-1
    # The FFT method gives length 2N, with the last element being 0 (since h_pad has N zeros at end)
    # Actually, let's verify.
    # h_pad = [h0, h1, ..., h_{N-1}, 0, ..., 0] (length 2N)
    # Convolution of h_pad with h_pad is circular convolution of length 2N.
    # Since h_pad is zero for indices >= N, the linear convolution is the same as circular convolution for the first 2N-1 elements?
    # Let's check with a small example.
    # h = [1, 2], N=2. h_pad = [1, 2, 0, 0].
    # np.convolve([1,2], [1,2]) = [1, 4, 4].
    # FFT method:
    # H = fft([1,2,0,0]) = [3, 1-1j, -1, 1+1j]
    # H*H = [9, (1-1j)^2, 1, (1+1j)^2] = [9, -2j, 1, 2j]
    # ifft([9, -2j, 1, 2j]) = [3, 2, 1, 2]? No.
    # ifft([9, -2j, 1, 2j]):
    # x[0] = (9 + (-2j) + 1 + 2j)/4 = 10/4 = 2.5
    # x[1] = (9 + (-2j)*(-1j) + 1*(-1) + 2j*(-1j))/4 = (9 + (-2) - 1 + 2)/4 = 8/4 = 2
    # x[2] = (9 + (-2j)*(-1) + 1*(-1) + 2j*(1))/4 = (9 + 2j - 1 + 2j)/4 = (8 + 4j)/4 = 2 + j -> real part 2
    # x[3] = (9 + (-2j)*(1j) + 1*(-1) + 2j*(-1))/4 = (9 + 2 - 1 - 2)/4 = 8/4 = 2
    # So ifft gives [2.5, 2, 2, 2].
    # But np.convolve gives [1, 4, 4].
    # The issue is that np.convolve is linear convolution, while FFT gives circular convolution.
    # To get linear convolution via FFT, we need to zero-pad to at least length len(h1) + len(h2) - 1.
    # Here len(h1)=len(h2)=N, so we need length 2N-1.
    # If we pad to 2N, the circular convolution will wrap around.
    # The element at index 2N-1 in the circular convolution will be the sum of the linear convolution at index 2N-1 and index 0.
    # Since we only care about the max, and the wrap-around might affect the value, we should be careful.
    # However, for h = ones(N), the autoconvolution is [1, 2, ..., N, ..., 2, 1].
    # The max is N at the center.
    # In the FFT method with pad 2N:
    # h_pad = [1, ..., 1, 0, ..., 0].
    # The circular convolution will have the peak at index N-1 (value N) and also some contribution at index 0?
    # Let's re-evaluate.
    # Actually, a simpler way is to just use np.convolve directly if N is small enough, or use the correct FFT padding.
    # The correct padding for linear convolution of two length-N arrays is 2N-1.
    # But FFT requires power of 2 for efficiency, so we pad to the next power of 2 >= 2N-1.
    # For N=400, 2N-1 = 799. Next power of 2 is 1024.
    
    size = 1
    while size < 2 * N - 1:
        size *= 2
    
    h_pad = np.zeros(size)
    h_pad[:N] = h
    H = np.fft.fft(h_pad)
    conv_freq = H * H
    conv = np.fft.ifft(conv_freq).real
    # The linear convolution is the first 2N-1 elements of conv
    return conv[:2 * N - 1]


def _logsumexp(x, axis=None, b=None):
    """
    Stable log-sum-exp.
    """
    m = np.max(x, axis=axis, keepdims=True)
    # Handle case where all x are -inf
    if np.all(np.isneginf(m)):
        return np.full(np.shape(m), -np.inf)
    # Subtract max to avoid overflow
    x_shifted = x - m
    s = np.sum(np.exp(x_shifted), axis=axis, keepdims=True)
    out = np.log(s) + m
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _objective(params, K, beta):
    """
    Compute the objective value using the Fourier basis and log-sum-exp surrogate.
    params: array of shape (2K + 1,) containing real and imaginary parts of Fourier coefficients.
    params[0] is the real part of the DC component (which is real).
    params[1] is the imaginary part of the DC component (should be 0, but we allow it).
    params[2:2K+2] are the real and imaginary parts of the positive frequency components.
    
    We reconstruct h from the Fourier coefficients.
    Then compute the autoconvolution and the surrogate max.
    """
    # Reconstruct h
    # h[n] = sum_{k=-K}^{K} c_k exp(i 2 pi k n / N)
    # We use the FFT convention.
    # Let's construct the full spectrum C of length N.
    C = np.zeros(N, dtype=complex)
    # DC component
    C[0] = params[0] + 1j * params[1]
    # Positive frequencies
    for k in range(1, K + 1):
        C[k] = params[2 * k] + 1j * params[2 * k + 1]
        # Negative frequencies (conjugate symmetry for real h)
        C[N - k] = np.conj(C[k])
    
    # Inverse FFT to get h
    h = np.fft.ifft(C).real * N  # Scale by N because ifft divides by N
    
    # Compute sum(h)
    sum_h = np.sum(h)
    if abs(sum_h) < 1e-10:
        return 1e10  # Large penalty
    
    # Compute autoconvolution
    conv = _full_autoconvolve_from_fourier(h)
    
    # Compute the value: abs(2 * N * max(conv) / sum_h^2)
    # We use a soft-max surrogate for max(conv)
    # max(conv) approx (1/beta) * log(sum(exp(beta * conv)))
    # To avoid overflow, we use log-sum-exp
    # conv can have negative values, so we need to be careful.
    # The log-sum-exp formula is: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # We want max(conv).
    # Let's use: max(conv) approx (1/beta) * logsumexp(beta * conv)
    # This is a smooth approximation.
    
    # For numerical stability, we scale conv before applying logsumexp
    # The values of conv can be large.
    # Let's use a fixed beta, or adapt it.
    
    # Compute logsumexp(beta * conv)
    # beta * conv can be large.
    # We use the stable logsumexp function.
    
    # To avoid underflow/overflow in exp, we rely on the stable logsumexp.
    # However, if beta * conv is very large, exp might overflow.
    # The stable logsumexp handles this by subtracting the max.
    
    # Let's compute the surrogate max
    # surrogate_max = (1/beta) * logsumexp(beta * conv)
    
    # We need to be careful with the scaling.
    # Let's try a moderate beta.
    
    # Compute the value
    surrogate_max = (1.0 / beta) * _logsumexp(beta * conv)
    
    value = abs(2.0 * N * surrogate_max / (sum_h ** 2))
    
    return value


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Parameters
    K = 20  # Number of positive frequency components to include
    beta = 10.0  # Inverse temperature for log-sum-exp
    
    # Initial guess: all ones in time domain -> only DC component non-zero
    # h = ones(N) -> C[0] = N, C[k] = 0 for k != 0
    # params[0] = N, params[1] = 0, params[2:] = 0
    params = np.zeros(2 * K + 2)
    params[0] = N  # DC component
    
    # Use L-BFGS-B to minimize the objective
    # We need to provide gradients for better performance.
    # For now, let's try without gradients and see if it works.
    # If it's too slow, we'll add gradients.
    
    # Let's try a simple gradient-free method first, or a coarse grid search.
    # Actually, L-BFGS-B with numerical gradients might be slow.
    # Let's try a random restart with a few iterations.
    
    best_params = params.copy()
    best_value = _objective(params, K, beta)
    
    # Multi-start with random initializations
    np.random.seed(42)
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        if i == 0:
            p = params.copy()
        else:
            p = np.random.randn(2 * K + 2) * 0.1
            p[0] = N  # Keep DC component large
        
        # Optimize
        from scipy.optimize import minimize
        
        try:
            res = minimize(_objective, p, args=(K, beta), method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-6})
            if res.fun < best_value:
                best_value = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Reconstruct h from best_params
    C = np.zeros(N, dtype=complex)
    C[0] = best_params[0] + 1j * best_params[1]
    for k in range(1, K + 1):
        C[k] = best_params[2 * k] + 1j * best_params[2 * k + 1]
        C[N - k] = np.conj(C[k])
    
    h = np.fft.ifft(C).real * N
    
    # Final polish: use the exact max instead of the surrogate
    # And refine with a few more iterations if time permits
    
    if time.time() < deadline - 10.0:
        # Try to improve with exact max
        def _objective_exact(p, K):
            C = np.zeros(N, dtype=complex)
            C[0] = p[0] + 1j * p[1]
            for k in range(1, K + 1):
                C[k] = p[2 * k] + 1j * p[2 * k + 1]
                C[N - k] = np.conj(C[k])
            h = np.fft.ifft(C).real * N
            sum_h = np.sum(h)
            if abs(sum_h) < 1e-10:
                return 1e10
            conv = _full_autoconvolve_from_fourier(h)
            value = abs(2.0 * N * np.max(conv) / (sum_h ** 2))
            return value
        
        from scipy.optimize import minimize
        try:
            res = minimize(_objective_exact, best_params, args=(K,), method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-8})
            if res.fun < best_value:
                best_value = res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Reconstruct h again
        C = np.zeros(N, dtype=complex)
        C[0] = best_params[0] + 1j * best_params[1]
        for k in range(1, K + 1):
            C[k] = best_params[2 * k] + 1j * best_params[2 * k + 1]
            C[N - k] = np.conj(C[k])
        h = np.fft.ifft(C).real * N
    
    # Ensure h is finite and sum(h) != 0
    if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-10:
        return np.ones(N)
    
    return h


# Test
if __name__ == '__main__':
    h = construct()
    conv = np.convolve(h, h)
    value = abs(2 * N * np.max(conv) / np.sum(h) ** 2)
    print(f"Value: {value}")
    print(f"Score: {1.0 / value}")
