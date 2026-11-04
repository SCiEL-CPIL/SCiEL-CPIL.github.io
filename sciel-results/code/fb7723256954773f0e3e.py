import numpy as np
from scipy.optimize import minimize

N = 400
M = 2 * N

def construct():
    best_x = None
    best_val = np.inf
    
    np.random.seed(42)
    initializations = [
        np.hanning(N),
        np.ones(N),
        np.random.randn(N),
        np.abs(np.random.randn(N)),
        np.random.randn(N) * np.hanning(N),
    ]
    
    for x0 in initializations:
        x0 = x0.copy()
        s = np.sum(x0)
        if abs(s) < 1e-6:
            x0 += 0.1
            s = np.sum(x0)
        x0 = x0 / s
        
        tau = 0.1
        
        def objective(x):
            x_pad = np.zeros(M)
            x_pad[:N] = x
            X = np.fft.fft(x_pad)
            P = np.abs(X)**2
            R = np.fft.ifft(P).real
            
            R_sidelobes = np.concatenate([R[1:N], R[M-N+1:M]])
            abs_R = np.abs(R_sidelobes)
            
            max_abs = np.max(abs_R)
            if max_abs > 500:
                return 1e10
            
            shifted = abs_R - max_abs
            smooth_max = tau * (max_abs/tau + np.log(np.sum(np.exp(shifted/tau))))
            
            sum_x = np.sum(x)
            if abs(sum_x) < 1e-10:
                return 1e10
            
            return smooth_max - 2 * np.log(abs(sum_x))
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            
            x = result.x
            x_pad = np.zeros(M)
            x_pad[:N] = x
            X = np.fft.fft(x_pad)
            P = np.abs(X)**2
            R = np.fft.ifft(P).real
            R_sidelobes = np.concatenate([R[1:N], R[M-N+1:M]])
            sum_x = np.sum(x)
            if abs(sum_x) > 1e-10:
                score = np.max(np.abs(R_sidelobes)) / (sum_x**2)
                if score < best_val:
                    best_val = score
                    best_x = x.copy()
        except Exception:
            continue
    
    if best_x is None:
        best_x = np.hanning(N)
    
    return best_x
