"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local Basis Function Network (Gaussian RBF) in State Space (x, v).
    
    Strategy:
    - Instead of global polynomials/trig, use 5 Gaussian kernels centered in the (x, v) plane.
    - This allows the model to capture local nonlinearities and the specific shape of the 
      oscillator's trajectory (e.g., limit cycle) without requiring high-order global terms.
    - Inputs x and v are normalized to improve numerical conditioning for BFGS.
    - Centers and widths are parameterized via sigmoid/tanh to keep them within reasonable bounds
      relative to the normalized input space, ensuring stability when starting from params=1.
    - Time 't' is included via a simple harmonic drive term (sin/cos) as a separate additive component,
      assuming the primary nonlinearity is state-dependent.
      
    Parameters (length 10):
    p[0:4]  -> Centers for 4 kernels in x-dimension (normalized space)
    p[4:8]  -> Centers for 4 kernels in v-dimension (normalized space)
    p[8]    -> Width (sigma) for all kernels (shared for simplicity and stability)
    p[9]    -> Global scale/offset or drive amplitude (mapped to small range)
    """
    p = params
    
    # Normalization constants based on training ranges:
    # x in [-0.2056, 0.1749] -> mean ~ -0.015, range ~ 0.38
    # v in [-0.3391, 0.3624] -> mean ~ 0.011, range ~ 0.70
    # We normalize to roughly [-1, 1] to make the RBF centers (initialized via sigmoid of 1.0)
    # land in a sensible region.
    
    # x_norm = (x - mean_x) / scale_x
    # v_norm = (v - mean_v) / scale_v
    
    mean_x = -0.01535
    scale_x = 0.19
    mean_v = 0.01165
    scale_v = 0.35
    
    x_norm = (x - mean_x) / scale_x
    v_norm = (v - mean_v) / scale_v
    
    # We use 4 kernels to save parameters for width and drive.
    # Centers: map raw params to [-1.5, 1.5] using tanh.
    # Starting at 1.0, tanh(1) ~ 0.76, which is a good initial guess for centers.
    
    cx1 = 1.5 * np.tanh(p[0])
    cx2 = 1.5 * np.tanh(p[1])
    cx3 = 1.5 * np.tanh(p[2])
    cx4 = 1.5 * np.tanh(p[3])
    
    cv1 = 1.5 * np.tanh(p[4])
    cv2 = 1.5 * np.tanh(p[5])
    cv3 = 1.5 * np.tanh(p[6])
    cv4 = 1.5 * np.tanh(p[7])
    
    # Width: map to [0.2, 1.0] to ensure kernels are not too narrow (noisy) or too wide (linear).
    # p[8] starts at 1.0. Let's use a softplus-like or sigmoid mapping.
    # sigma = 0.2 + 0.8 * sigmoid(p[8])
    # sigmoid(1) ~ 0.73 -> sigma ~ 0.78. Good.
    sigma = 0.2 + 0.8 * (1.0 / (1.0 + np.exp(-p[8])))
    
    # Compute RBF activations for the 4 kernels.
    # diff_x1 = x_norm - cx1, etc.
    
    # Kernel 1
    d1_x = x_norm - cx1
    d1_v = v_norm - cv1
    rbf1 = np.exp(-(d1_x**2 + d1_v**2) / (2 * sigma**2))
    
    # Kernel 2
    d2_x = x_norm - cx2
    d2_v = v_norm - cv2
    rbf2 = np.exp(-(d2_x**2 + d2_v**2) / (2 * sigma**2))
    
    # Kernel 3
    d3_x = x_norm - cx3
    d3_v = v_norm - cv3
    rbf3 = np.exp(-(d3_x**2 + d3_v**2) / (2 * sigma**2))
    
    # Kernel 4
    d4_x = x_norm - cx4
    d4_v = v_norm - cv4
    rbf4 = np.exp(-(d4_x**2 + d4_v**2) / (2 * sigma**2))
    
    # We need weights for these kernels. We don't have enough params left for 4 weights + drive.
    # Let's restructure the 10 params:
    # p[0:8] -> 4 centers (2 each)
    # p[8] -> shared width
    # p[9] -> shared weight for all kernels? No, that's too restrictive.
    
    # Alternative: Use 3 kernels?
    # p[0:6] -> 3 centers (2 each)
    # p[6] -> width
    # p[7:10] -> 3 weights
    # This is better. Let's switch to 3 kernels.
    
    # Recalculate with 3 kernels
    cx1 = 1.5 * np.tanh(p[0])
    cx2 = 1.5 * np.tanh(p[1])
    cx3 = 1.5 * np.tanh(p[2])
    
    cv1 = 1.5 * np.tanh(p[3])
    cv2 = 1.5 * np.tanh(p[4])
    cv3 = 1.5 * np.tanh(p[5])
    
    sigma = 0.2 + 0.8 * (1.0 / (1.0 + np.exp(-p[6])))
    
    w1 = p[7]
    w2 = p[8]
    w3 = p[9]
    
    # Recompute RBFs with 3 kernels
    d1_x = x_norm - cx1
    d1_v = v_norm - cv1
    rbf1 = np.exp(-(d1_x**2 + d1_v**2) / (2 * sigma**2))
    
    d2_x = x_norm - cx2
    d2_v = v_norm - cv2
    rbf2 = np.exp(-(d2_x**2 + d2_v**2) / (2 * sigma**2))
    
    d3_x = x_norm - cx3
    d3_v = v_norm - cv3
    rbf3 = np.exp(-(d3_x**2 + d3_v**2) / (2 * sigma**2))
    
    # Combine RBF outputs
    y_rbf = w1 * rbf1 + w2 * rbf2 + w3 * rbf3
    
    # Add a time-dependent drive term?
    # We have no params left for sin/cos(t) coefficients.
    # However, the oscillator is likely driven. If we ignore t, we assume autonomous or that t is just a label.
    # Given the task says "nonlinear oscillator" and inputs include t, ignoring t is risky if it's driven.
    # But we have 0 params left.
    # Let's check if we can squeeze in a fixed-time term.
    # The prompt says "up to 10 fitted constants".
    # If we use 3 kernels (6 centers + 1 width + 3 weights = 10 params), we have no room for t.
    # Is t important? If it's a driven oscillator, yes.
    # If it's an autonomous oscillator sampled over time, t is redundant with x,v (on the trajectory).
    # Given the ranges, x and v are small, t is large.
    # Often in these tasks, if t is included, it's a drive.
    # Let's try to fit the drive with a fixed frequency? No, frequency is unknown.
    # Let's assume the nonlinearity is the main challenge and the drive is simple (e.g., sinusoidal).
    # If we must include t, we should reduce kernels.
    
    # Let's try 2 kernels + time terms.
    # p[0:4] -> 2 centers (2 each)
    # p[4] -> width
    # p[5:7] -> 2 weights
    # p[7] -> sin(t) coeff
    # p[8] -> cos(t) coeff
    # p[9] -> constant offset
    
    # This structure is more robust for driven systems.
    
    # Recalculate with 2 kernels
    cx1 = 1.5 * np.tanh(p[0])
    cx2 = 1.5 * np.tanh(p[1])
    cv1 = 1.5 * np.tanh(p[2])
    cv2 = 1.5 * np.tanh(p[3])
    
    sigma = 0.2 + 0.8 * (1.0 / (1.0 + np.exp(-p[4])))
    
    w1 = p[5]
    w2 = p[6]
    
    d1_x = x_norm - cx1
    d1_v = v_norm - cv1
    rbf1 = np.exp(-(d1_x**2 + d1_v**2) / (2 * sigma**2))
    
    d2_x = x_norm - cx2
    d2_v = v_norm - cv2
    rbf2 = np.exp(-(d2_x**2 + d2_v**2) / (2 * sigma**2))
    
    y_rbf = w1 * rbf1 + w2 * rbf2
    
    # Time terms
    # t is in [30, 50]. Frequency unknown. 
    # Using raw t in sin/cos with unknown frequency is hard.
    # If we assume the drive frequency is 1 rad/s (common), sin(t) and cos(t) work.
    # If the frequency is different, this will fail.
    # However, the previous program used sin(t)/cos(t) and got a score.
    # Let's use sin(t) and cos(t) with fitted coefficients.
    
    y_drive = p[7] * np.sin(t) + p[8] * np.cos(t)
    
    y_offset = p[9]
    
    y = y_rbf + y_drive + y_offset
    
    return y

# EVOLVE-BLOCK-END
