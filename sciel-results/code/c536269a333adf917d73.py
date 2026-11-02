import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 600
DEADLINE = 140.0  # Safety margin before 150s limit

def compute_score(h):
    """Compute normalized max of autoconvolution: max(auto_conv) / ((sum h)^2 / N)"""
    h = np.asarray(h, dtype=np.float64)
    total = np.sum(h)
    if total <= 0:
        return float('inf')
    H = np.fft.fft(h)
    auto_conv = np.fft.ifft(np.abs(H)**2).real
    return np.max(auto_conv) / (total * total / N)

def construct():
    """
    Returns a sequence of nonnegative step heights on equal intervals of [-1/4, 1/4].
    Uses a combination of structural initialization and local optimization.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()
    
    # Try various structural initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Two rectangles (disjoint)
    for split in [N//4, N//3, N//2, 2*N//3, 3*N//4]:
        h = np.zeros(N)
        h[:split//2] = 1.0
        h[N-split//2:] = 1.0
        initializations.append(h)
    
    # 3. Alternating blocks
    for block_size in [10, 20, 30, 50]:
        h = np.zeros(N)
        for i in range(0, N, 2*block_size):
            h[i:i+block_size] = 1.0
        initializations.append(h)
    
    # 4. Gaussian-like
    x = np.linspace(-1, 1, N)
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        h = np.exp(-x**2 / (2*sigma**2))
        initializations.append(h)
    
    # 5. Cosine window
    x = np.linspace(-1, 1, N)
    h = np.cos(np.pi * x / 2)
    h = np.maximum(h, 0)
    initializations.append(h)
    
    # 6. Random smooth
    rng = np.random.RandomState(42)
    for _ in range(5):
        h = rng.rand(N)
        # Smooth
        kernel = np.ones(10) / 10
        h = np.convolve(h, kernel, mode='same')
        h = np.maximum(h, 0)
        initializations.append(h)
    
    # Evaluate all initializations
    for h_init in initializations:
        if time_left() < 5:
            break
        h_init = np.maximum(h_init, 0)
        if np.sum(h_init) <= 0:
            continue
        # Normalize to sum = N for consistency
        h_init = h_init * (N / np.sum(h_init))
        score = compute_score(h_init)
        if score < best_score:
            best_score = score
            best_h = h_init.copy()
    
    # Local optimization from best initialization using gradient-free methods
    if time_left() > 20:
        # Use Nelder-Mead on a reduced set of parameters
        # We'll optimize in a low-dimensional parameterization
        # Use a Fourier basis: h[i] = a0 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N)
        # But ensure nonnegativity
        
        K = 20  # Number of Fourier modes
        def fourier_to_h(params):
            a0 = params[0]
            a = params[1:1+K]
            b = params[1+K:1+2*K]
            i = np.arange(N)
            h = np.full(N, a0)
            for k in range(1, K+1):
                h += a[k-1] * np.cos(2*np.pi*k*i/N)
                h += b[k-1] * np.sin(2*np.pi*k*i/N)
            h = np.maximum(h, 0)
            if np.sum(h) <= 0:
                return np.ones(N)
            return h * (N / np.sum(h))
        
        def objective(params):
            h = fourier_to_h(params)
            return compute_score(h)
        
        # Start from the best_h we have
        # Project best_h onto Fourier basis
        a0 = np.mean(best_h)
        a = np.zeros(K)
        b = np.zeros(K)
        i = np.arange(N)
        for k in range(1, K+1):
            a[k-1] = 2/N * np.sum(best_h * np.cos(2*np.pi*k*i/N))
            b[k-1] = 2/N * np.sum(best_h * np.sin(2*np.pi*k*i/N))
        
        x0 = np.concatenate([[a0], a, b])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_h = fourier_to_h(res.x)
        except Exception:
            pass
    
    # Try differential evolution on a smaller parameter space if time permits
    if time_left() > 30:
        # Optimize a mixture of a few basis functions
        def mixture_objective(params):
            # params: [w1, w2, w3, c1, c2, c3]
            # h = w1 * gaussian1 + w2 * gaussian2 + w3 * gaussian3
            w = params[:3]
            centers = params[3:6]
            x = np.linspace(-1, 1, N)
            h = np.zeros(N)
            for j in range(3):
                h += w[j] * np.exp(-(x - centers[j])**2 / 0.2)
            h = np.maximum(h, 0)
            if np.sum(h) <= 0:
                return 1e6
            h = h * (N / np.sum(h))
            return compute_score(h)
        
        bounds = [(0, 10)] * 3 + [(-1, 1)] * 3
        try:
            res = differential_evolution(mixture_objective, bounds, seed=42,
                                         maxiter=100, popsize=15, tol=1e-6,
                                         polish=True)
            if res.fun < best_score:
                best_score = res.fun
                h = mixture_objective(res.x)  # This returns score, need to recompute
                # Recompute h
                w = res.x[:3]
                centers = res.x[3:6]
                x = np.linspace(-1, 1, N)
                h = np.zeros(N)
                for j in range(3):
                    h += w[j] * np.exp(-(x - centers[j])**2 / 0.2)
                h = np.maximum(h, 0)
                if np.sum(h) > 0:
                    h = h * (N / np.sum(h))
                    best_h = h
        except Exception:
            pass
    
    # Final projection and cleanup
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) <= 0:
        best_h = np.ones(N)
    
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Best score: {compute_score(np.array(res))}")
