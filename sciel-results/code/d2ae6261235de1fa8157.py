import numpy as np

N = 400

def construct():
    # Initialize with random real vector
    np.random.seed(42)
    x = np.random.randn(N)
    
    # We want to maximize the denominator (sum x)^2, so let's start with positive bias
    x = np.abs(x)
    x = x / np.linalg.norm(x)
    
    M = 2 * N  # Zero-padding for linear autocorrelation
    
    best_x = x.copy()
    best_score = np.inf
    
    # ADMM-like parameters
    lambda_l1 = 0.01  # Sparsity penalty
    mu = 1.0          # Quadratic penalty parameter
    t = 1.0           # Auxiliary variable for max sidelobe
    
    for iteration in range(500):
        # 1. Compute Autocorrelation
        x_pad = np.zeros(M)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        P = np.abs(X)**2
        R = np.fft.ifft(P).real
        
        # Extract linear autocorrelation
        # R[0] is lag 0
        # R[1:N] are lags 1 to N-1
        # R[M-N+1:M] are lags -(N-1) to -1
        R_lag0 = R[0]
        R_sidelobes = np.concatenate([R[1:N], R[M-N+1:M]])
        
        # Calculate current score (max abs sidelobe / sum^2)
        sum_x = np.sum(x)
        if sum_x == 0:
            sum_x = 1e-9
        score = np.max(np.abs(R_sidelobes)) / (sum_x**2)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # 2. Update t (max sidelobe)
        t = np.max(np.abs(R_sidelobes))
        
        # 3. Update x
        # Minimize sum(|R_sidelobes| - t)^2 + lambda * ||x||_1
        # Approximation: Minimize energy of sidelobes (flattening spectrum) + L1 penalty
        # Flattening spectrum:
        P_mean = np.mean(P)
        # Sharpen the flattening to reduce peaks
        P_new = np.sqrt(P * P_mean)
        
        # Recover X
        X_new = np.sqrt(P_new) * np.exp(1j * np.angle(X))
        x_new = np.fft.ifft(X_new).real[:N]
        
        # Apply Proximal Operator for L1 norm (Soft Thresholding)
        # This enforces sparsity
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - lambda_l1, 0)
        
        # Ensure we don't lose the DC component entirely (keep sum large)
        # If sum becomes too small, the score blows up.
        # Add a small bias to maintain non-zero sum
        if np.sum(x_new) < 0.1:
             x_new = x_new + 0.1 * np.ones(N)
             
        x_new = x_new / np.linalg.norm(x_new)
        x = x_new
        
    return best_x
