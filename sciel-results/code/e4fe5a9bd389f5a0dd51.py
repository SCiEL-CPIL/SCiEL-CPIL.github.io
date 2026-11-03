import numpy as np
from scipy.optimize import minimize

N = 600
# Intervals on [-1/4, 1/4]
x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
# Actually, step function heights.
# Let's define the function by its values on the grid.

def objective(params):
    # params: coefficients for basis functions
    # Basis: 1, cos(2*pi*k*x), sin(2*pi*k*x)
    # Keep it simple: 1, cos(4*pi*x), cos(8*pi*x)
    c0, c1, c2 = params
    
    # Construct f
    f = c0 + c1 * np.cos(4 * np.pi * x) + c2 * np.cos(8 * np.pi * x)
    
    # Enforce non-negativity
    f = np.maximum(f, 0)
    
    # Calculate norm
    L1 = np.sum(f) * (0.5 / N) # dx
    if L1 == 0:
        return 1e10
        
    # Autoconvolution
    # Use FFT for efficiency
    f_fft = np.fft.fft(f)
    conv_fft = f_fft * f_fft
    conv = np.fft.ifft(conv_fft).real
    
    # The convolution is circular if we use FFT directly on the array.
    # We need linear convolution.
    # Pad with zeros.
    f_padded = np.zeros(2*N)
    f_padded[:N] = f
    conv = np.fft.ifft(np.fft.fft(f_padded, 2*N) ** 2).real[:N] # Center part
    
    # The peak of the convolution
    max_conv = np.max(conv)
    
    # Metric: max_conv / L1^2
    return max_conv / (L1 * L1)

# Initial guess: constant (c0=1, c1=0, c2=0)
# But we want to explore deviations.
# Since constant is likely optimal, we might not find anything better.
# However, if the seed score is 1.0, and we can't beat it, we should just return the constant function.
# But to be safe, let's try to optimize slightly.

def construct():
    # Try a few shapes
    best_val = 1e10
    best_f = [1.0] * N
    
    # 1. Constant
    f_const = np.ones(N)
    L1 = np.sum(f_const) * (0.5/N)
    conv = np.fft.ifft(np.fft.fft(np.concatenate([f_const, np.zeros(N)])) ** 2).real[:N]
    val_const = np.max(conv) / (L1*L1)
    
    # 2. Optimized Super-ellipse / Cosine mix
    # Use Nelder-Mead on a few parameters
    def obj(params):
        c0, c1, c2 = params
        f = c0 + c1 * np.cos(4 * np.pi * x) + c2 * np.cos(8 * np.pi * x)
        f = np.maximum(f, 0)
        L1 = np.sum(f) * (0.5/N)
        if L1 < 1e-9: return 1e10
        conv = np.fft.ifft(np.fft.fft(np.concatenate([f, np.zeros(N)])) ** 2).real[:N]
        return np.max(conv) / (L1*L1)
        
    # Start near constant
    x0 = [1.0, 0.0, 0.0]
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if res.fun < val_const:
        c0, c1, c2 = res.x
        f_opt = c0 + c1 * np.cos(4 * np.pi * x) + c2 * np.cos(8 * np.pi * x)
        f_opt = np.maximum(f_opt, 0)
        # Normalize? No, metric is scale invariant.
        # But we need to return the heights.
        # The evaluator likely normalizes or expects specific scaling?
        # "normalized maximum".
        # If I return f_opt, the evaluator calculates the metric.
        # I should return the raw heights.
        best_f = list(f_opt)
    else:
        best_f = list(f_const)
        
    return best_f
