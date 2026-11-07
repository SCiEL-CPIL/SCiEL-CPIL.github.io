"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Gaussian RBF basis.
    
    Inputs:
        x: displacement
        v: velocity
        params: array of 10 fitted constants [p0, ..., p9]
    
    Basis terms:
        p0 * exp(-((x - cx0)^2 + (v - cv0)^2) / (2 * s0^2))
        p1 * exp(-((x - cx1)^2 + (v - cv1)^2) / (2 * s1^2))
        p2 * exp(-((x - cx2)^2 + (v - cv2)^2) / (2 * s2^2))
        p3 * exp(-((x - cx3)^2 + (v - cv3)^2) / (2 * s3^2))
        p4 * exp(-((x - cx4)^2 + (v - cv4)^2) / (2 * s4^2))
    
    Centers (cx, cv) and scales (s) are fixed to cover the input domain:
    x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    """
    # Fixed centers covering the input space (grid-like distribution)
    # x centers: -0.4, 0, 0.4
    # v centers: -0.1, 0.1
    # We need 5 terms. Let's use a 2x2 grid plus one in the middle, or just 5 specific points.
    # x: -0.3, 0.0, 0.3
    # v: -0.1, 0.1
    # That's 6 points. We have 5 params.
    # Let's use:
    # 1: (-0.3, -0.1)
    # 2: (0.0, -0.1)
    # 3: (0.3, -0.1)
    # 4: (-0.3, 0.1)
    # 5: (0.3, 0.1)
    # This leaves the center (0,0) uncovered by a specific center, but the RBFs from corners will cover it.
    # Alternatively, use a 1D grid in x and v combined? No, 2D RBFs are better for interaction.
    
    # Let's try a symmetric set of 5 points:
    # 1: (0, 0)
    # 2: (-0.4, 0)
    # 3: (0.4, 0)
    # 4: (0, -0.2)
    # 5: (0, 0.2)
    # This covers the axes well.
    
    cx = np.array([0.0, -0.4, 0.4, 0.0, 0.0])
    cv = np.array([0.0, 0.0, 0.0, -0.2, 0.2])
    s = np.array([0.5, 0.3, 0.3, 0.3, 0.3]) # Widths. 0.5 is broad for center, 0.3 for others.
    
    # Calculate squared distances
    dx = x - cx
    dv = v - cv
    
    # Compute RBF values
    # dist_sq = dx^2 + dv^2
    # rbf = exp(-dist_sq / (2 * s^2))
    
    # To avoid overflow/underflow issues and ensure differentiability, we compute directly.
    # x and v are arrays. cx, cv, s are scalars/arrays.
    # Broadcasting: x is (N,), cx is (5,). We need (N, 5).
    
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dv = v[:, np.newaxis] - cv[np.newaxis, :]
    
    dist_sq = dx**2 + dv**2
    s_sq = s[np.newaxis, :]**2
    
    # Prevent division by zero if s is 0, but s is fixed > 0.
    rbf_vals = np.exp(-dist_sq / (2 * s_sq))
    
    # Linear combination
    # params is (10,), but we only use 5 for the RBFs?
    # The task allows up to 10 params. We should use all 10 to maximize flexibility.
    # We can add 5 linear terms as well? Or use 10 RBFs?
    # 10 RBFs might be too many for 10000 points? No, it's fine.
    # But 10 RBFs with fixed centers might be overkill or underfit depending on centers.
    # Let's stick to 5 RBFs + 5 polynomial terms?
    # The current best (73.7) uses 10 polynomial terms.
    # A hybrid might be good: 5 RBFs + 5 Polynomials.
    # Or just 10 RBFs.
    
    # Let's use 10 RBFs to fully utilize the parameter space.
    # We need 10 centers.
    # Grid: 3x3 = 9 points + 1 center? Or 2x5?
    # x: -0.4, 0, 0.4
    # v: -0.2, 0, 0.2
    # That's 9 points. Add (0,0) again? No.
    # Let's use a 2x5 grid?
    # x: -0.4, -0.2, 0, 0.2, 0.4
    # v: -0.1, 0.1
    # That's 10 points.
    
    cx2 = np.array([-0.4, -0.2, 0.0, 0.2, 0.4, -0.4, -0.2, 0.0, 0.2, 0.4])
    cv2 = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    s2 = np.full(10, 0.3) # Uniform width
    
    dx2 = x[:, np.newaxis] - cx2[np.newaxis, :]
    dv2 = v[:, np.newaxis] - cv2[np.newaxis, :]
    
    dist_sq2 = dx2**2 + dv2**2
    s_sq2 = s2[np.newaxis, :]**2
    
    rbf_vals2 = np.exp(-dist_sq2 / (2 * s_sq2))
    
    # rbf_vals2 is (N, 10)
    # params is (10,)
    # result = sum(rbf_vals2 * params, axis=1)
    
    return np.sum(rbf_vals2 * params, axis=1)

# EVOLVE-BLOCK-END
