import numpy as np
import time

N = 400

def _autoconv_lin(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    conv = np.fft.irfft(X * X.conj(), size)
    return conv[:2 * n - 1]

def _metric_offdiag(x):
    """Max |c_k| for k != 0, divided by (sum x)^2."""
    c = _autoconv_lin(x)
    n = len(x)
    # Off-diagonal lags
    # c[0] is lag 0
    # c[1..n-1] are positive lags
    # c[n..2n-2] are negative lags (c[n+k] corresponds to lag -k? No)
    # Linear conv of x with x:
    # index 0: lag 0
    # index 1: lag 1
    # ...
    # index n-1: lag n-1
    # index n: lag -(n-1) ?? 
    # Let's check:
    # x = [1, 2]
    # conv = [1, 4, 4]
    # lag 0: 1*1 + 2*2 = 5? No.
    # np.convolve([1,2], [1,2]) = [1, 4, 4].
    # c[0] = 1*1 = 1 (lag 1? No, standard conv index 0 is x[0]x[0])
    # Wait, np.convolve does linear convolution.
    # c[k] = sum_i x[i] x[i+k] is cross-correlation.
    # Autoconvolution is sum_i x[i] x[i+k].
    # np.convolve(x, x) gives sum_i x[i] x[j-i].
    # This is different from autocorrelation.
    # The task says "autoconvolution".
    # Usually in signal processing, "autocorrelation" is R[k] = sum x[n] x[n+k].
    # "Autoconvolution" is (x*x)[k] = sum x[n] x[k-n].
    # The seed code uses:
    # X = np.fft.rfft(x)
    # conv = np.fft.irfft(X * X.conj())
    # This computes the **Autocorrelation** (circular, then linear extracted).
    # X * conj(X) is the power spectrum, IDFT is autocorrelation.
    # So the seed minimizes the peak of the **Autocorrelation**.
    # The task text says "autoconvolution".
    # However, the seed code is the ground truth for the interface.
    # I will stick to the seed's definition: Autocorrelation.
    
    # For autocorrelation R[k] = sum x[n] x[n+k].
    # R[0] = sum x^2 = N (for +/- 1).
    # R[k] for k>0.
    # The seed extracts:
    # off = np.concatenate([c[1:n], c[n+1:2*n-1]])
    # This assumes the FFT output is arranged as [R[0], R[1], ..., R[N-1], R[-(N-1)], ..., R[-1]]?
    # np.fft.irfft of X*conj(X) gives the circular autocorrelation.
    # For linear, we need to be careful.
    # If we zero-pad to 2N-1, the circular autocorrelation of the zero-padded signal
    # is the linear autocorrelation.
    # The output of irfft is length size.
    # The values are R[0], R[1], ..., R[N-1], R[-(N-1)]...
    # Actually, for real input, the autocorrelation is symmetric.
    # R[k] = R[-k].
    # The FFT result puts positive lags first, then negative?
    # No, irfft returns the time-domain sequence.
    # If we treat the zero-padded signal as circular, the "wrap around" is avoided.
    # The output c[0] is R[0].
    # c[1] is R[1].
    # ...
    # c[N-1] is R[N-1].
    # c[N] is R[-(N-1)]? 
    # For a signal of length N, zero padded to 2N-1.
    # The autocorrelation lags range from -(N-1) to N-1.
    # The output array from irfft has length 2N-1.
    # Indices 0 to N-1 correspond to lags 0 to N-1.
    # Indices N to 2N-2 correspond to lags -(N-1) to -1.
    # So c[1:N] are positive lags.
    # c[N:2N-1] are negative lags.
    # Since R[k] = R[-k], we only need to check positive lags.
    # The seed checks both, which is redundant but safe.
    
    off = np.concatenate([c[1:n], c[n+1:2*n-1]])
    max_abs = np.max(np.abs(off)) if len(off) > 0 else 0.0
    s = np.sum(x)
    if abs(s) < 1e-12:
        return max_abs
    return max_abs / (s * s)

def construct():
    deadline = time.time() + 115.0
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_score = _metric_offdiag(best_x)
    
    def _local_search(x, time_budget):
        cur = x.copy().astype(float)
        cur_score = _metric_offdiag(cur)
        t_end = time.time() + time_budget
        improved = True
        while improved and time.time() < t_end:
            improved = False
            # Try flipping each bit
            for i in range(N):
                if time.time() > t_end:
                    break
                cur[i] = -cur[i]
                new_score = _metric_offdiag(cur)
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                else:
                    cur[i] = -cur[i]
        return cur, cur_score

    # Strategy: Multi-start with different biases
    biases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for bias in biases:
        if time.time() > deadline - 10:
            break
        
        # Generate random sequence with given fraction of -1s
        num_negs = int(N * bias)
        indices = rng.choice(N, size=num_negs, replace=False)
        x = np.ones(N)
        x[indices] = -1
        
        # Local search
        x_opt, score_opt = _local_search(x, 5)
        
        if score_opt < best_score:
            best_score = score_opt
            best_x = x_opt
    
    # Final refinement on the best found
    if time.time() < deadline - 5:
        x_final, score_final = _local_search(best_x, deadline - time.time() - 5)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final
            
    return best_x.astype(float)
