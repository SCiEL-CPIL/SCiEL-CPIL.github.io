"""Program for minimizing the normalized maximum of autoconvolution of a step function."""
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    def linear_autoconv(h):
        # Linear convolution of h with h
        # Length is 2*N - 1
        C = np.convolve(h, h, mode='full')
        return C

    def optimize(h0, steps=2000, lr=0.01):
        h = h0.copy()
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s > 0:
            h /= s
        else:
            h = np.ones(N) / N
            
        for step in range(steps):
            C = linear_autoconv(h)
            max_val = np.max(C)
            # Find indices where C is close to max to smooth the gradient
            # Using a threshold helps with non-differentiability
            threshold = max_val * 0.99
            max_indices = np.where(C >= threshold)[0]
            
            # Compute gradient
            # G[j] = sum_{k in max_indices} dC[k]/dh[j]
            # dC[k]/dh[j] = 2 * h[k-j] if 0 <= k-j < N, else 0
            # We can compute this by summing shifted versions of h
            G = np.zeros(N)
            # For efficiency, we can use the fact that this is a correlation-like sum
            # But N=600 and max_indices might be small.
            # If max_indices is large, it's O(N * len(max_indices)).
            # Let's try to be efficient.
            # G[j] = 2 * sum_{k in max_indices} h[k-j]
            # This is 2 * (h * indicator_max_indices)[j] ? No.
            # Let's just loop if len(max_indices) is small.
            if len(max_indices) < 50:
                for k in max_indices:
                    # h[k-j] for j=0..N-1
                    # valid when 0 <= k-j < N => k-N+1 <= j <= k
                    start = max(0, k - N + 1)
                    end = min(N, k + 1)
                    # j ranges from start to end-1
                    # h[k-j]
                    # Let's create the slice
                    # indices of h: k-j
                    # j=start -> k-start
                    # j=end-1 -> k-(end-1)
                    # We need h[k-start : k-(end-1)+1 : -1]?
                    # Simpler:
                    # h_vals = h[k - np.arange(start, end)]
                    # G[start:end] += 2 * h_vals
                    j_vals = np.arange(start, end)
                    h_vals = h[k - j_vals]
                    G[start:end] += 2 * h_vals
            else:
                # Fallback: FFT based or just slower loop
                # For robustness, if max_indices is large, the function is very flat.
                # We can use the full convolution gradient
                # G = 2 * np.convolve(h, np.ones_like(C)[max_indices] / len(max_indices), mode='same')? 
                # No, let's stick to the loop for now, N=600 is small.
                for k in max_indices:
                    start = max(0, k - N + 1)
                    end = min(N, k + 1)
                    j_vals = np.arange(start, end)
                    h_vals = h[k - j_vals]
                    G[start:end] += 2 * h_vals

            # Update
            h -= lr * G
            h = np.clip(h, 0, None)
            s = np.sum(h)
            if s > 0:
                h /= s
            else:
                h = np.ones(N) / N

        return h

    # Initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 / (2 * 0.2**2))
    inits.append(gauss)
    
    # 3. Cosine bump
    cos = np.cos(np.pi * x / 2) ** 2
    inits.append(cos)
    
    # 4. Triangular
    tri = 1 - np.abs(x)
    inits.append(tri)

    best_h = None
    best_score = np.inf
    
    for h0 in inits:
        # Use adaptive learning rate or fixed
        h_opt = optimize(h0, steps=1000, lr=0.05)
        C = linear_autoconv(h_opt)
        score = np.max(C)
        if score < best_score:
            best_score = score
            best_h = h_opt
            
    return list(best_h)
# EVOLVE-BLOCK-END
