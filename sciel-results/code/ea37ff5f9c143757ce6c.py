import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_T0 = None

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate: RBF basis (8 Gaussians) + explicit b*s coupling + intercept.
    
    params layout (10 total):
      params[0:8]  -> weights for 8 RBF basis functions
      params[8]    -> weight for explicit b*s substrate-biomass coupling term
      params[9]    -> intercept / bias
    
    The explicit b*s term captures density-dependent substrate limitation,
    which is a key biological mechanism in E. coli growth kinetics.
    """
    # Normalize inputs to [0, 1] based on training ranges
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    T_n = (temp - 1.0) / 39.0
    p_n = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] for stability
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    T_n = np.clip(T_n, 0.0, 1.0)
    p_n = np.clip(p_n, 0.0, 1.0)
    
    # Define 8 RBF centers in 4D normalized space
    # Centers: [b, s, T, p]
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],  # corner 1
        [1.0, 1.0, 1.0, 1.0],  # corner 2
        [0.0, 1.0, 0.0, 1.0],  # corner 3
        [1.0, 0.0, 1.0, 0.0],  # corner 4
        [0.5, 0.0, 0.5, 0.0],  # mid-edge 1
        [0.0, 0.5, 0.0, 0.5],  # mid-edge 2
        [1.0, 0.5, 1.0, 0.5],  # mid-edge 3
        [0.5, 1.0, 0.5, 1.0],  # mid-edge 4
    ])
    
    # Gaussian width (fixed)
    sigma = 0.35
    
    # Stack inputs into 4D
    x = np.stack([b_n, s_n, T_n, p_n], axis=-1)  # shape: (n, 4)
    
    # Compute distances to each center
    diffs = x[:, np.newaxis, :] - centers[np.newaxis, :, :]  # shape: (n, 8, 4)
    dist2 = np.sum(diffs ** 2, axis=-1)  # shape: (n, 8)
    
    # Compute RBF basis values
    basis = np.exp(-dist2 / (2.0 * sigma ** 2))  # shape: (n, 8)
    
    # RBF weighted sum
    rbf_out = basis @ params[0:8]  # shape: (n,)
    
    # Explicit substrate-biomass coupling term: b_n * s_n
    # This captures Monod-like saturation / density-dependent limitation
    coupling = b_n * s_n  # shape: (n,)
    
    # Combine: RBF + coupling + intercept
    y = rbf_out + params[8] * coupling + params[9]
    
    return y
