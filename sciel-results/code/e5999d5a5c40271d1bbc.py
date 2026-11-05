import numpy as np
import scipy.optimize as opt
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Number of harmonics
    # 40 harmonics -> 80 parameters (sine + cosine for each)
    K = 40
    
    t = np.arange(N)
    # Create basis matrix of shape (N, 2*K)
    # Columns: cos(2*pi*k*t/N), sin(2*pi*k*t/N) for k=1..K
    basis = np.zeros((N, 2 * K))
    for k in range(1, K + 1):
        basis[:, 2 * (k - 1)] = np.cos(2 * np.pi * k * t / N)
        basis[:, 2 * (k - 1) + 1] = np.sin(2 * np.pi * k * t / N)
    
    # We can also add a DC term? 
    # If we include a DC term (k=0), it's just a constant offset.
    # The sum S is dominated by the DC term.
    # If we don't include it, the sum S is determined by the interaction of harmonics.
    # It's safer to include a DC term to control the mean explicitly.
    # Let's add one extra column for the DC term (constant 1).
    # New shape (N, 2*K + 1)
    basis_full = np.hstack([basis, np.ones((N, 1))])
    
    n_params = basis_full.shape[1]
    
    def objective(theta):
        h = basis_full @ theta
        S = np.sum(h)
        if abs(S) < 1e-8:
            return 1e10
        # Compute autocorrelation
        # Using FFT for speed, though N=400 is small
        h_fft = np.fft.fft(h, 2 * N)
        h_conv = np.real(np.fft.ifft(h_fft * np.conj(h_fft)))[:N]
        # Note: np.convolve(h, h) gives full convolution length 2N-1.
        # The max is taken over the full convolution.
        # h_conv computed above is circular convolution of length 2N (padded).
        # The linear convolution is the first N elements of the circular convolution 
        # if padded to 2N-1? 
        # Let's check: 
        # Linear conv of h (len N) with h (len N) has length 2N-1.
        # Circular conv of h padded to 2N has length 2N.
        # The first 2N-1 elements of the circular conv (padded to 2N) match the linear conv?
        # Yes, if we zero pad h to length 2N-1 or 2N.
        # Let's use np.convolve to be safe and exact as per task description.
        # N=400 is small enough for np.convolve to be fast enough.
        h_conv_linear = np.convolve(h, h)
        
        # Soft max using Log-Sum-Exp
        beta = 10.0
        # Numerical stability for LSE
        max_val = np.max(h_conv_linear)
        lse = (1.0 / beta) * (max_val + np.log(np.sum(np.exp(beta * (h_conv_linear - max_val)))))
        
        val = (2.0 * N * lse) / (S * S)
        return val
    
    best_h = np.ones(N)
    best_val = np.inf
    
    # Multi-start
    rng = np.random.default_rng(42)
    
    for i in range(200): # Max restarts, limited by time
        if time.time() > deadline:
            break
            
        # Initialize theta
        # Small random values for harmonics, larger for DC to ensure S != 0
        theta0 = rng.normal(0, 0.1, size=n_params)
        theta0[-1] = 1.0 + rng.normal(0, 0.1) # DC term
        
        try:
            res = opt.minimize(objective, theta0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9})
            if res.success:
                h_candidate = basis_full @ res.x
                # Calculate true objective
                S = np.sum(h_candidate)
                if abs(S) < 1e-8:
                    continue
                h_conv_linear = np.convolve(h_candidate, h_candidate)
                true_val = (2.0 * N * np.max(h_conv_linear)) / (S * S)
                
                if true_val < best_val:
                    best_val = true_val
                    best_h = h_candidate
        except Exception:
            continue
            
    return best_h
