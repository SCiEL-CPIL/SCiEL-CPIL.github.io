# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Local-Phase Model: RBFs in (x,v) modulated by Fourier terms in t.
    
    Structure:
    - Two Gaussian RBFs centered in the phase plane to capture local amplitude modulation.
    - Each RBF is multiplied by cos(t) and sin(t) to capture the periodic driving.
    - Additional global polynomial terms for the conservative/dissipative forces.
    
    Parameters (10 total):
    - params[0]: Weight for G1 * cos(t)
    - params[1]: Weight for G1 * sin(t)
    - params[2]: Weight for G2 * cos(t)
    - params[3]: Weight for G2 * sin(t)
    - params[4]: Global bias
    - params[5]: Linear x coeff
    - params[6]: Linear v coeff
    - params[7]: x^3 coeff (Duffing)
    - params[8]: x^2 * v coeff (Van der Pol damping)
    - params[9]: x * v coeff (Coupling)
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Fixed Gaussian centers
    cx1, cv1 = 0.0, 0.0
    cx2, cv2 = 0.1, 0.1
    sigma = 0.2
    sigma2 = 2.0 * sigma * sigma
    
    # Compute Gaussians
    g1 = np.exp(-((x - cx1)**2 + (v - cv1)**2) / sigma2)
    g2 = np.exp(-((x - cx2)**2 + (v - cv2)**2) / sigma2)
    
    # Fourier terms
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    
    # Local drive components
    drive1_c = params[0] * g1 * cos_t
    drive1_s = params[1] * g1 * sin_t
    drive2_c = params[2] * g2 * cos_t
    drive2_s = params[3] * g2 * sin_t
    
    # Global physical terms
    bias = params[4]
    lin_x = params[5] * x
    lin_v = params[6] * v
    duffing = params[7] * (x ** 3)
    vdp_damping = params[8] * (x ** 2) * v
    coupling = params[9] * x * v
    
    pred = drive1_c + drive1_s + drive2_c + drive2_s + bias + lin_x + lin_v + duffing + vdp_damping + coupling
    
    return pred
# EVOLVE-BLOCK-END
