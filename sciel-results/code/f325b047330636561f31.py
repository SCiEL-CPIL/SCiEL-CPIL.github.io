import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Robust symbolic regression basis: combine Chebyshev polynomials,
    # Fourier terms, and Gaussian RBFs to cover polynomial, oscillatory,
    # and localized non-linear behavior with a well-conditioned 10-column basis.
    
    n = x.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Scale to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # --- Chebyshev polynomials T_0 .. T_4 (5 terms) ---
    T0 = np.ones(n)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # --- Fourier basis (3 terms) ---
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    
    # --- Gaussian RBFs at two symmetric centers (2 terms) ---
    # Placed at ±1.0 (mid-range within [-2, 2]) to capture localized bumps
    rbf_1 = np.exp(-0.8 * (x - 1.0) ** 2)
    rbf_2 = np.exp(-0.8 * (x + 1.0) ** 2)
    
    # Build 10-column basis matrix
    basis = np.column_stack([
        T0, T1, T2, T3, T4,   # 0-4: Chebyshev
        sin_x, cos_x, sin_2x, # 5-7: Fourier
        rbf_1, rbf_2          # 8-9: RBFs
    ])
    
    # Linear combination
    pred = basis @ params
    
    # Safety: ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
