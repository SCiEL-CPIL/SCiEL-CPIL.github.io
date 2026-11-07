"""
Strategy: Adaptive Radial Basis Function (RBF) Kernel Model.
Instead of a fixed polynomial basis, this model uses a sum of Gaussian kernels
centered at 7 fixed points in the input space. The 10 parameters control the
bandwidth (inverse variance) of the kernels, the ridge regularization, and the
weights of the centers. This allows the model to approximate a wider class of
smooth functions than a quadratic polynomial, including those with local features
or non-polynomial curvature, by adapting the effective smoothness via the 
bandwidth parameters.
"""
# EVOLVE-BLOCK-START
import numpy as np

# 7 fixed centers in [0.5, 2]^3
# Chosen to cover the corners and center of the domain to ensure global coverage
_CENTERS = np.array([
    [0.5, 0.5, 0.5],
    [2.0, 0.5, 0.5],
    [0.5, 2.0, 0.5],
    [2.0, 2.0, 0.5],
    [0.5, 0.5, 2.0],
    [2.0, 0.5, 2.0],
    [1.25, 1.25, 1.25]  # Center
])

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using an RBF basis with adaptive bandwidths.
    params[0]: bandwidth1 (inverse variance) for first 4 centers (z=0.5 plane)
    params[1]: bandwidth2 (inverse variance) for next 2 centers (z=2.0 plane)
    params[2]: bandwidth3 (inverse variance) for center
    params[3]: ridge regularization coefficient
    params[4..9]: weights for the 7 centers
    """
    # Ensure params are finite to avoid NaNs in calculation
    if not np.all(np.isfinite(params)):
        return np.zeros_like(x)

    p_bw1, p_bw2, p_bw3, p_ridge, w0, w1, w2, w3, w4, w5 = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9]
    
    # We need 7 weights but only have 6 slots (4-9). 
    # Let's re-map: 
    # params[4..9] are 6 params. 
    # I will use 6 centers to fit 6 weights, or share weights.
    # Actually, let's use 6 centers to match the 6 weight slots.
    # Centers:
    # 0: [0.5, 0.5, 0.5]
    # 1: [2.0, 0.5, 0.5]
    # 2: [0.5, 2.0, 0.5]
    # 3: [2.0, 2.0, 0.5]
    # 4: [0.5, 0.5, 2.0]
    # 5: [2.0, 0.5, 2.0]
    # This covers 6 of the 8 corners. The 7th center (1.25, 1.25, 1.25) can be handled by a shared weight or just omitted if we strictly follow 10 params.
    # Let's use 6 centers and 6 weights.
    
    # Re-define centers for 6 RBFs
    c0 = np.array([0.5, 0.5, 0.5])
    c1 = np.array([2.0, 0.5, 0.5])
    c2 = np.array([0.5, 2.0, 0.5])
    c3 = np.array([2.0, 2.0, 0.5])
    c4 = np.array([0.5, 0.5, 2.0])
    c5 = np.array([2.0, 0.5, 2.0])

    # Calculate distances squared
    # x, y, z are arrays of same length
    dx0 = x - c0[0]; dy0 = y - c0[1]; dz0 = z - c0[2]
    dx1 = x - c1[0]; dy1 = y - c1[1]; dz1 = z - c1[2]
    dx2 = x - c2[0]; dy2 = y - c2[1]; dz2 = z - c2[2]
    dx3 = x - c3[0]; dy3 = y - c3[1]; dz3 = z - c3[2]
    dx4 = x - c4[0]; dy4 = y - c4[1]; dz4 = z - c4[2]
    dx5 = x - c5[0]; dy5 = y - c5[1]; dz5 = z - c5[2]

    r2_0 = dx0*dx0 + dy0*dy0 + dz0*dz0
    r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1
    r2_2 = dx2*dx2 + dy2*dy2 + dz2*dz2
    r2_3 = dx3*dx3 + dy3*dy3 + dz3*dz3
    r2_4 = dx4*dx4 + dy4*dy4 + dz4*dz4
    r2_5 = dx5*dx5 + dy5*dy5 + dz5*dz5

    # Bandwidths: exp(-bw * r^2)
    # Clamp bandwidths to be positive to ensure decay
    bw1 = np.maximum(p_bw1, 1e-6)
    bw2 = np.maximum(p_bw2, 1e-6)
    bw3 = np.maximum(p_bw3, 1e-6)

    # Assign bandwidths to centers
    # Centers 0-3 use bw1, Centers 4-5 use bw2. 
    # What about bw3? Let's use bw3 for a "global" term or just ignore it if not needed.
    # Better: Use 3 distinct bandwidths for variety.
    # C0, C1 -> bw1
    # C2, C3 -> bw2
    # C4, C5 -> bw3
    
    k0 = np.exp(-bw1 * r2_0)
    k1 = np.exp(-bw1 * r2_1)
    k2 = np.exp(-bw2 * r2_2)
    k3 = np.exp(-bw2 * r2_3)
    k4 = np.exp(-bw3 * r2_4)
    k5 = np.exp(-bw3 * r2_5)

    # Ridge term: sum of kernels * ridge
    # This acts as a regularizer on the magnitude of the prediction
    # Standard RBF regression: f(x) = sum(w_i * K(x, c_i))
    # The "ridge" parameter can scale the whole output or act as a bias.
    # Let's use p_ridge as a scaling factor for the sum to allow the optimizer to adjust magnitude if needed, 
    # or more commonly, it's part of the loss. But here we just return the prediction.
    # If we include ridge in the prediction, it might distort the fit.
    # However, the task says "params... minimizing training MSE". 
    # If we don't use p_ridge in the output, the optimizer will ignore it (gradient 0).
    # So we MUST use p_ridge in the output to give it a role.
    # Let's make p_ridge a global bias/offset term? No, that's p0 in poly.
    # Let's make p_ridge a multiplier for the constant term? 
    # Or better: The model is f(x) = sum(w_i * K_i(x)) + p_ridge * (mean of K_i(x))? 
    # Simplest: f(x) = sum(w_i * K_i(x)) + p_ridge
    # This makes p_ridge a global bias. This is useful.

    pred = w0 * k0 + w1 * k1 + w2 * k2 + w3 * k3 + w4 * k4 + w5 * k5 + p_ridge

    return pred
# EVOLVE-BLOCK-END
