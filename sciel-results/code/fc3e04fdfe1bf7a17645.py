import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    # Structural change: Hybrid basis combining local RBFs (for sharp features)
    # and global Chebyshev polynomials (for smooth trends).
    # This addresses the limitation of pure polynomial bases (Runge phenomenon, 
    # poor local approximation) and pure RBFs (poor extrapolation, conditioning).
    # 
    # Basis composition (10 params):
    # 1. Constant term (T0)
    # 2. Linear term (T1)
    # 3. Quadratic term (T2)
    # 4. Cubic term (T3)
    # 5. Quartic term (T4)
    # 6-10. Five RBFs centered at -1.5, -0.75, 0, 0.75, 1.5
    
    # Global Chebyshev polynomials (stable on [-1, 1], scaled to [-2, 2])
    # We use standard Chebyshev T_n(x/2) to keep values bounded.
    # T0(z) = 1
    # T1(z) = z
    # T2(z) = 2z^2 - 1
    # T3(z) = 4z^3 - 3z
    # T4(z) = 8z^4 - 8z^2 + 1
    z = x / 2.0  # Map [-2, 2] to [-1, 1]
    
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z**2 - 1.0
    T3 = 4.0 * z**3 - 3.0 * z
    T4 = 8.0 * z**4 - 8.0 * z**2 + 1.0
    
    # Local RBFs
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    sigma = 0.5  # Width parameter
    
    # Compute RBF values: exp(-((x - c)^2) / (2 * sigma^2))
    # Shape: (n_points, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Build basis matrix
    # Columns 0-4: Chebyshev polynomials
    # Columns 5-9: RBFs
    basis = np.column_stack([T0, T1, T2, T3, T4, rbf])
    
    # Linear combination
    return basis @ params
