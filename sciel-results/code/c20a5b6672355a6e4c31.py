import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Mixture Model with 3 components and a constant offset.
    y = A0 + sum_{i=1}^3 Ai * exp(-0.5 * ((x - ci) / wi)^2)
    wi = exp(vi) to ensure positive width.
    
    params:
    0: A0
    1: A1
    2: c1
    3: v1 (log width)
    4: A2
    5: c2
    6: v2
    7: A3
    8: c3
    9: v3
    """
    x = np.asarray(x, dtype=np.float64)
    
    A0 = params[0]
    
    A1, c1, v1 = params[1:4]
    A2, c2, v2 = params[4:7]
    A3, c3, v3 = params[7:10]
    
    w1 = np.exp(v1)
    w2 = np.exp(v2)
    w3 = np.exp(v3)
    
    # Avoid division by zero if w is extremely small (though exp(v) is always > 0)
    # If v is very negative, w is tiny. (x-c)/w is huge. exp(-huge) is 0.
    # This is fine.
    
    g1 = np.exp(-0.5 * ((x - c1) / w1) ** 2)
    g2 = np.exp(-0.5 * ((x - c2) / w2) ** 2)
    g3 = np.exp(-0.5 * ((x - c3) / w3) ** 2)
    
    y = A0 + A1 * g1 + A2 * g2 + A3 * g3
    
    return y
