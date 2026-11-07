import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural Change: Sparse Sum of Gaussian Basis Functions
    # Model: y = offset + sum_{i=1}^{3} weight_i * exp(-||x - center_i||^2 / (2 * width^2))
    # where x = [t_norm, ph_norm, s_norm] (3D feature space)
    # 
    # Parameters (10 total):
    # p[0]: offset (baseline growth rate)
    # p[1], p[2], p[3]: center of Gaussian 1 (t, ph, s)
    # p[4]: weight of Gaussian 1
    # p[5], p[6], p[7]: center of Gaussian 2 (t, ph, s)
    # p[8]: weight of Gaussian 2
    # p[9]: shared isotropic width (squared for positivity)
    
    # Shared width parameter (squared to ensure positivity)
    width_sq = p[9]**2 + 1e-6
    
    # Compute 3D distances for each Gaussian
    # Gaussian 1
    dist1_sq = (t_norm - p[1])**2 + (ph_norm - p[2])**2 + (s_norm - p[3])**2
    gauss1 = np.exp(-dist1_sq / (2.0 * width_sq))
    
    # Gaussian 2
    dist2_sq = (t_norm - p[5])**2 + (ph_norm - p[6])**2 + (s_norm - p[7])**2
    gauss2 = np.exp(-dist2_sq / (2.0 * width_sq))
    
    # Combine: offset + weighted sum of Gaussians
    y = p[0] + p[4] * gauss1 + p[8] * gauss2
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
