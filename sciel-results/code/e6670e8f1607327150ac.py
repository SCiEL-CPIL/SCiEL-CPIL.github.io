import numpy as np

N = 400
M = 2 * N

def construct():
    np.random.seed(42)
    x = np.random.randn(N)
    x = x / np.linalg.norm(x)
    
    # ADMM parameters
    rho = 1.0
    lambda_l1 = 0.1  # Sparsity threshold
    max_iters = 300
    tol = 1e-6
    grad_steps = 5
    lr = 0.05
    
    # Dual variables
    u = np.zeros(M)
    v = np.zeros(N)
    
    # Primal variables
    z = x.copy()
    
    best_psl = np.inf
    best_x = x.copy()
    
    # Helper: Compute linear autocorrelation
    def autocorr(x):
        x_pad = np.zeros(M)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        P = np.abs(X)**2
        R = np.fft.ifft(P).real
        return R
        
    # Helper: Gradient of ||R(x) - target||^2 w.r.t x
    def grad_r(x, target):
        Rx = autocorr(x)
        diff = Rx - target
        x_pad = np.zeros(M)
        x_pad[:N] = x
        
        X = np.fft.fft(x_pad)
        D = np.fft.fft(diff)
        Grad_F = 2 * X * D
        Grad = np.fft.ifft(Grad_F).real[:N]
        
        return Grad

    for i in range(max_iters):
        # 1. Update y (spectral peak constraint)
        Rx = autocorr(x)
        t = np.max(np.abs(Rx))
        y = np.clip(Rx + u, -t, t)
        
        # 2. Update z (sparsity)
        z = np.sign(x + v) * np.maximum(np.abs(x + v) - lambda_l1/rho, 0)
        
        # 3. Update x
        target_r = y - u
        target_x = z - v
        
        # Perform gradient steps
        for _ in range(grad_steps):
            grad_r_val = grad_r(x, target_r)
            grad_x_val = x - target_x
            total_grad = grad_r_val + grad_x_val
            x = x - lr * total_grad
            norm_x = np.linalg.norm(x)
            if norm_x > 1e-10:
                x = x / norm_x
            else:
                x = np.random.randn(N)
                x = x / np.linalg.norm(x)
                
        # Update duals
        Rx_new = autocorr(x)
        u = u + Rx_new - y
        v = v + x - z
        
        # Check convergence
        current_psl = np.max(np.abs(Rx_new[1:])) / Rx_new[0]
        if current_psl < best_psl:
            best_psl = current_psl
            best_x = x.copy()
            
        if i > 50 and abs(current_psl - best_psl) < tol:
            break
            
    return best_x
