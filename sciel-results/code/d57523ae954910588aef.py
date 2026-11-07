import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Uses a multi-scale Gaussian RBF basis with centers distributed on a 3x3 grid
    across the domain and varying bandwidths to capture both local high-frequency
    features and global smooth trends. The model is strictly linear in the 10
    parameters to ensure BFGS converges reliably to the global optimum.
    
    Basis functions:
    1. Bias (global offset)
    2. RBF at (-1, -1) with narrow bandwidth
    3. RBF at (-1,  1) with narrow bandwidth
    4. RBF at ( 1, -1) with narrow bandwidth
    5. RBF at ( 1,  1) with narrow bandwidth
    6. RBF at (-0.5, -0.5) with medium bandwidth
    7. RBF at (-0.5,  0.5) with medium bandwidth
    8. RBF at ( 0.5, -0.5) with medium bandwidth
    9. RBF at ( 0.5,  0.5) with medium bandwidth
    10. RBF at (0, 0) with wide bandwidth (global trend)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Ensure 1D arrays
    x = x.ravel()
    y = y.ravel()

    # Precompute squared distances for each RBF center
    # Center 1: (-1, -1), bandwidth sigma=0.4 -> 1/(2*sigma^2) = 1/0.32 = 3.125
    dx1 = x + 1.0
    dy1 = y + 1.0
    r1_sq = dx1 * dx1 + dy1 * dy1
    b2 = np.exp(-3.125 * r1_sq)

    # Center 2: (-1, 1), bandwidth sigma=0.4
    dx2 = x + 1.0
    dy2 = y - 1.0
    r2_sq = dx2 * dx2 + dy2 * dy2
    b3 = np.exp(-3.125 * r2_sq)

    # Center 3: (1, -1), bandwidth sigma=0.4
    dx3 = x - 1.0
    dy3 = y + 1.0
    r3_sq = dx3 * dx3 + dy3 * dy3
    b4 = np.exp(-3.125 * r3_sq)

    # Center 4: (1, 1), bandwidth sigma=0.4
    dx4 = x - 1.0
    dy4 = y - 1.0
    r4_sq = dx4 * dx4 + dy4 * dy4
    b5 = np.exp(-3.125 * r4_sq)

    # Center 5: (-0.5, -0.5), bandwidth sigma=0.6 -> 1/(2*0.36) = 1/0.72 ≈ 1.3889
    dx5 = x + 0.5
    dy5 = y + 0.5
    r5_sq = dx5 * dx5 + dy5 * dy5
    b6 = np.exp(-1.3889 * r5_sq)

    # Center 6: (-0.5, 0.5), bandwidth sigma=0.6
    dx6 = x + 0.5
    dy6 = y - 0.5
    r6_sq = dx6 * dx6 + dy6 * dy6
    b7 = np.exp(-1.3889 * r6_sq)

    # Center 7: (0.5, -0.5), bandwidth sigma=0.6
    dx7 = x - 0.5
    dy7 = y + 0.5
    r7_sq = dx7 * dx7 + dy7 * dy7
    b8 = np.exp(-1.3889 * r7_sq)

    # Center 8: (0.5, 0.5), bandwidth sigma=0.6
    dx8 = x - 0.5
    dy8 = y - 0.5
    r8_sq = dx8 * dx8 + dy8 * dy8
    b9 = np.exp(-1.3889 * r8_sq)

    # Center 9: (0, 0), bandwidth sigma=1.0 -> 1/(2*1.0) = 0.5
    r9_sq = x * x + y * y
    b10 = np.exp(-0.5 * r9_sq)

    # Bias term
    b1 = np.ones_like(x)

    # Linear combination
    pred = (p[0] * b1 +
            p[1] * b2 +
            p[2] * b3 +
            p[3] * b4 +
            p[4] * b5 +
            p[5] * b6 +
            p[6] * b7 +
            p[7] * b8 +
            p[8] * b9 +
            p[9] * b10)

    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
