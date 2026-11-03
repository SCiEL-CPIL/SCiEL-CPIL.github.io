import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 145  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for speed
    n_fft = 2 * N
    f_hat = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
    return np.max(conv) / (s * s)

def gradient_soft_max_conv(f, beta):
    """
    Compute the gradient of the soft-max of the autoconvolution of f.
    Returns the gradient vector and the soft-max value.
    """
    n_fft = 2 * N
    f_hat = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
    
    # Soft max weights
    c_max = np.max(conv)
    if c_max < 1e-15:
        return np.zeros_like(f), 0.0
    
    # Numerical stability
    shifted = conv - c_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        w = np.zeros_like(conv)
    else:
        w = exp_shifted / sum_exp
        
    soft_max_val = c_max + np.log(sum_exp) / beta
    
    # Gradient computation
    # G_j = sum_k w_k (f_{j-k} + f_{j+k}) = 2 * (w * f)_j
    # Since w is symmetric (conv is symmetric), we can use circular convolution
    # We need to pad w to size n_fft
    w_padded = np.zeros(n_fft)
    w_padded[:len(w)] = w
    # Note: conv is length 2N-1. w is length 2N-1.
    # The indices in conv are 0..2N-2 corresponding to lags -(N-1)..(N-1)
    # For circular convolution via FFT, we need to map lags to indices correctly.
    # Standard FFT convolution computes sum_k w_k f_{j-k} (circular).
    # Our w_k corresponds to lag k.
    # In FFT, index 0 is lag 0, index 1 is lag 1, ..., index N-1 is lag N-1,
    # index N is lag -N+1 (or N+1?), etc.
    # Actually, np.fft.irfft returns linear convolution if zero-padded.
    # But we want the gradient w.r.t f_j which involves f_{j+k} and f_{j-k}.
    # This is a correlation/convolution structure.
    
    # Let's compute G via FFT of w and f.
    # We need (w * f)_j = sum_k w_k f_{j-k}.
    # This is standard linear convolution if we treat w and f as finite sequences.
    # However, the relationship is local.
    # Let's just use the property that G = 2 * ifft(fft(w_padded) * fft(f_padded))
    # We must be careful with the indexing of w.
    # conv[k] for k=0..2N-2.
    # Lag k = index k.
    # Lag -k = index 2N-1-k? No.
    # np.convolve(f, f, 'full') gives index i = i_f1 + i_f2.
    # Lag = i_f2 - i_f1? No.
    # Let's stick to the direct definition if N=600 is small enough for O(N^2) gradient?
    # N=600, N^2 = 360,000. This is very fast in Python/Numpy if vectorized?
    # No, 360k ops is nothing. But we do this in a loop.
    # Let's try to vectorize the gradient calculation directly.
    
    # G_j = sum_k w_k (f_{j-k} + f_{j+k})
    # We can compute this by looping over k where w_k is significant?
    # Or use FFT.
    
    # Let's use FFT properly.
    # w is defined on lags.
    # Let's construct a vector w_fft_ready of length n_fft.
    # w_fft_ready[0] = w[0] (lag 0)
    # w_fft_ready[k] = w[k] for k=1..N-1 (lag k)
    # w_fft_ready[n_fft-k] = w[k] for k=1..N-1 (lag -k)
    # Wait, conv has length 2N-1.
    # Index 0: lag -(N-1)? No.
    # np.convolve([1,0], [1,0], 'full') -> [1, 0, 0]. Index 0 is 0*0.
    # If f is length N.
    # conv[k] = sum_{i=0}^{N-1} f[i] f[i+k-N+1]? No.
    # Standard: conv[k] = sum_i f[i] f[k-i].
    # If we want lag d = j - i.
    # Let's just use the direct O(N^2) computation for gradient to avoid FFT indexing pitfalls, 
    # since N=600 is small.
    
    # G_j = sum_k w_k (f_{j-k} + f_{j+k})
    # We can compute this using np.convolve if we expand w to the right size?
    # Actually, let's just do it with matrix multiplication or simple loops if needed, 
    # but N=600 allows for efficient vectorized ops.
    
    # Let's try:
    # G = 2 * (w * f) where * is circular convolution?
    # No, it's linear convolution truncated.
    
    # Let's use the fact that w is symmetric.
    # G_j = 2 * sum_{k=0}^{N-1} w_k f_{j-k} (ignoring out of bounds for a moment, assuming zero padding)
    # This is linear convolution of w and f.
    # w length 2N-1. f length N.
    # conv_w_f = np.convolve(w, f, 'full')
    # This will be length 3N-2.
    # We need the central part corresponding to j=0..N-1.
    # The term w_k f_{j-k} is non-zero when 0 <= j-k < N and 0 <= k < 2N-1.
    # This is getting messy.
    
    # Simplest robust way:
    # G = np.zeros(N)
    # for k in range(2*N-1):
    #     if w[k] > 1e-10:
    #         # Add w[k] * (f shifted by k and -k)
    #         # f_{j-k} is f shifted right by k (with zeros)
    #         # f_{j+k} is f shifted left by k (with zeros)
    #         pass
    
    # Let's use FFT with correct padding.
    # We want G_j = sum_k w_k f_{j-k} + sum_k w_k f_{j+k}
    # Let A = linear_convolution(w, f).
    # A_m = sum_k w_k f_{m-k}.
    # Then sum_k w_k f_{j-k} = A_j.
    # sum_k w_k f_{j+k} = sum_k w_k f_{j+k}. Let l = j+k. sum_l w_{l-j} f_l.
    # This is cross-correlation.
    
    # Let's just compute the gradient via FFT of the power spectrum derivative.
    # J = soft_max(C).
    # dJ/df = 2 * ifft( F_hat * conj(dC/dF_hat) )?
    # C = ifft(|F_hat|^2).
    # dC/df is tricky.
    
    # Let's go back to the O(N^2) loop but vectorized over j?
    # No, let's use the property:
    # G = 2 * ifft( fft(w_padded) * fft(f_padded) )
    # If we define w_padded correctly for circular convolution that mimics linear.
    
    # Let's use a simpler, reliable method:
    # Compute G by summing contributions.
    # Since N=600, we can afford a few N^2 operations.
    # G = np.zeros(N)
    # for k in range(2*N-1):
    #     wk = w[k]
    #     if wk < 1e-12: continue
    #     # f_{j-k}
    #     # j-k in [0, N-1] => j in [k, k+N-1]
    #     # Intersection with [0, N-1]
    #     start = max(0, k)
    #     end = min(N, k+N)
    #     # f indices: j-k
    #     # j range: start..end-1
    #     # f indices: start-k .. end-1-k
    #     # This is getting error prone.
    
    # Let's use scipy.signal.convolve?
    # Or just use the FFT method with a known correct mapping.
    
    # Mapping:
    # conv[k] from np.convolve(f, f, 'full') has length 2N-1.
    # k=0 corresponds to i1=0, i2=0.
    # Lag d = i2 - i1.
    # For a given k, the pairs (i1, i2) satisfy i1 + i2 = k.
    # This is not a single lag.
    # Wait, autoconvolution is usually defined as (f * f)(t) = int f(t-s) f(s) ds.
    # In discrete: C[k] = sum_i f[i] f[i+k].
    # This is cross-correlation, not standard convolution.
    # np.convolve(f, f, 'full') computes sum_i f[i] f[k-i].
    # These are different.
    # The task says "autoconvolution". Usually in signal processing, autocorrelation is used for "similarity".
    # But "autoconvolution" specifically means f * f.
    # If it's f * f, then C[k] = sum_i f[i] f[k-i].
    # Then dC[k]/df[j] = f[k-j] + f[j-k] (since f[k-j] comes from i=j term, and f[j-k] comes from i=k-j term? No).
    # C[k] = sum_i f[i] f[k-i].
    # dC[k]/df[j] = f[k-j] (from i=j) + f[j-k] (from k-i=j => i=k-j).
    # So dC[k]/df[j] = f[k-j] + f[j-k].
    # This is the same form as before, just with different index interpretation.
    
    # So G_j = sum_k w_k (f[k-j] + f[j-k]).
    # Since w is symmetric (C is symmetric for real f? Yes, C[k] = C[-k] if we extend).
    # And f is real.
    # This is 2 * sum_k w_k f[j-k] (if w is symmetric and we sum over all k).
    # So G = 2 * (w * f).
    
    # Let's implement G = 2 * np.convolve(w, f, 'full') and extract the middle part.
    # w has length 2N-1. f has length N.
    # conv_w_f has length 3N-2.
    # We need G_j for j=0..N-1.
    # The term f[j-k] is non-zero for 0 <= j-k < N => j <= k+N-1 and j >= k.
    # So for a fixed j, we sum over k.
    # In the convolution result A = w * f, A[m] = sum_k w[k] f[m-k].
    # We want sum_k w[k] f[j-k]. This is A[j].
    # So G_j = 2 * A[j].
    # We just need A[0..N-1].
    
    A = np.convolve(w, f, 'full')
    G = 2.0 * A[:N]
    
    return G, soft_max_val

def optimize_gradient(f_init, start_time, max_time, beta_init=10.0, beta_end=100.0):
    """
    Optimize f using projected gradient descent on soft-max autoconvolution.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-12:
        f = np.ones(N)
        s = N
    f = f * (N / s)
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    eta = 0.01
    beta = beta_init
    
    while time.time() < max_time:
        # Increase beta slowly
        beta = min(beta * 1.05, beta_end)
        
        G, soft_val = gradient_soft_max_conv(f, beta)
        
        # Normalize gradient?
        # The scale of G depends on beta and f.
        # Let's normalize G to have unit norm or similar.
        g_norm = np.linalg.norm(G)
        if g_norm < 1e-12:
            break
        G = G / g_norm
        
        # Try step
        f_new = f - eta * G
        f_new = np.maximum(f_new, 0.0)
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            eta *= 0.5
            continue
        f_new = f_new * (N / s_new)
        
        val_new = compute_score(f_new)
        
        if val_new < best_val:
            best_val = val_new
            best_f = f_new.copy()
            f = f_new
            # Increase eta slightly
            eta = min(eta * 1.1, 0.1)
        else:
            # Decrease eta
            eta *= 0.7
            if eta < 1e-6:
                # Try smaller beta or break
                if beta > beta_end * 0.9:
                    break
                beta = beta * 1.2
                eta = 0.01
    
    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Initial guesses
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 * 5))
    
    # 3. Random
    rng = np.random.RandomState(42)
    inits.append(rng.rand(N))
    
    # 4. Cosine bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    inits.append(0.5 * (1 + np.cos(2 * np.pi * x)))
    
    best_f = None
    best_score = np.inf
    
    # Allocate time for each initial guess
    # We have ~145s. 4 inits. ~30s each.
    time_per_init = (TIME_LIMIT - 5) / len(inits)
    
    for i, f_init in enumerate(inits):
        if time.time() > TIME_LIMIT - 5:
            break
        
        t_end = start_time + time_per_init
        f_opt, score = optimize_gradient(f_init, start_time, t_end)
        
        if score < best_score:
            best_score = score
            best_f = f_opt
    
    # Final polish on the best result with remaining time
    if time.time() < TIME_LIMIT - 2 and best_f is not None:
        f_polish, score_polish = optimize_gradient(best_f, start_time, TIME_LIMIT - 2, beta_init=50.0, beta_end=200.0)
        if score_polish < best_score:
            best_f = f_polish
            best_score = score_polish
            
    if best_f is None:
        return [1.0] * N
        
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
