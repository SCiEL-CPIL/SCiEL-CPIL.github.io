import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters: 3 RBF centers, 3 RBF widths, 1 linear slope, 1 linear intercept, 1 global offset
    c1, c2, c3, w1, w2, w3, m, b, offset = params[:9]
    
    # Ensure widths are positive and not too small to avoid numerical issues
    # Use softplus or exp to guarantee positivity, but since BFGS optimizes raw params,
    # we need to handle potential negative widths. 
    # A common trick: use exp(w) or abs(w) + eps. 
    # However, if the optimizer drives w to negative, exp(w) becomes small.
    # Let's use np.exp(w) for widths to ensure they are positive.
    # But wait, if w is very negative, exp(w) is ~0, leading to division by zero in exp(-((x-c)/w)^2).
    # Better: width = np.exp(w) + 1e-6.
    
    w1 = np.exp(w1) + 1e-6
    w2 = np.exp(w2) + 1e-6
    w3 = np.exp(w3) + 1e-6
    
    # Compute RBFs
    # Gaussian RBF: exp(-((x - center) / width)^2)
    rbf1 = np.exp(-((x - c1) / w1)**2)
    rbf2 = np.exp(-((x - c2) / w2)**2)
    rbf3 = np.exp(-((x - c3) / w3)**2)
    
    # Linear term
    linear = m * x + b
    
    # Combine
    return rbf1 + rbf2 + rbf3 + linear + offset
