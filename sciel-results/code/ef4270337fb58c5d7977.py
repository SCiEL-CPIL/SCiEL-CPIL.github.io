import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fully parametric composite basis of 3 Gaussian bumps.
    
    Strategy:
    - Instead of a fixed basis with linear weights, the 10 parameters define the
      functional form itself: 3 centers, 3 widths, and 3 amplitudes.
    - This allows BFGS to move the bumps to where the function has features,
      change their sharpness, and scale their height, which is a structural
      advantage over fixed-width RBFs that can only scale existing fixed shapes.
    - Parameters are transformed to ensure positivity for widths (via exp) and
      reasonable ranges for centers (via tanh or similar scaling if needed, 
      though direct optimization in [-2, 2] is often stable enough with BFGS 
      if initialized well, but we start from ones so we need to map carefully).
      
    Parameter mapping:
    p[0:3] -> Centers c1, c2, c3. We map from R to [-2, 2] using 2*tanh(p) to keep them bounded.
    p[3:6] -> Widths sigma1, sigma2, sigma3. We map from R to (0, inf) using exp(p). 
             To prevent numerical issues with extremely small/large sigmas, we might 
             clamp or use a softplus-like behavior, but exp is standard. 
             Starting from 1, exp(1) ~ 2.7, which is a bit wide. 
             Maybe scale: sigma = 0.5 * exp(p). Start 1 -> 0.5*2.7 ~ 1.35. 
             Let's try sigma = 0.2 * exp(p). Start 1 -> 0.2*2.7 ~ 0.54. This is a good initial width.
    p[6:9] -> Amplitudes A1, A2, A3. Directly used.
    p[9]   -> Global offset/mean, or unused? 
              The task says params is length 10. 
              Let's use p[9] as a global constant offset (bias) to help fit the mean level, 
              which can speed up convergence of the amplitudes.
    
    Wait, the prompt says "fits params, an array of length 10". 
    If I use 3 centers + 3 widths + 3 amplitudes + 1 offset = 10 params. Perfect.
    
    Formula:
    y(x) = A1 * exp(-(x-c1)^2 / (2*sigma1^2)) + 
           A2 * exp(-(x-c2)^2 / (2*sigma2^2)) + 
           A3 * exp(-(x-c3)^2 / (2*sigma3^2)) + 
           A0
    
    Where:
    c1 = 2.0 * tanh(p[0])
    c2 = 2.0 * tanh(p[1])
    c3 = 2.0 * tanh(p[2])
    sigma1 = 0.2 * exp(p[3])
    sigma2 = 0.2 * exp(p[4])
    sigma3 = 0.2 * exp(p[5])
    A1 = p[6]
    A2 = p[7]
    A3 = p[8]
    A0 = p[9]
    """
    # Deadline guard: though this is a closed-form eval, it's fast.
    # No explicit loop, so no deadline check needed inside, but good practice.
    # The evaluator calls this many times during BFGS.
    
    x_flat = x.reshape(-1)
    
    # Extract and transform parameters
    # Centers
    c1 = 2.0 * np.tanh(params[0])
    c2 = 2.0 * np.tanh(params[1])
    c3 = 2.0 * np.tanh(params[2])
    
    # Widths (ensure positive and reasonable scale)
    # Using 0.2 * exp(p) means p=0 -> 0.2, p=1 -> 0.54, p=-1 -> 0.07.
    # This covers a good range of local to global features.
    s1 = 0.2 * np.exp(params[3])
    s2 = 0.2 * np.exp(params[4])
    s3 = 0.2 * np.exp(params[5])
    
    # Amplitudes
    a1 = params[6]
    a2 = params[7]
    a3 = params[8]
    
    # Offset
    a0 = params[9]
    
    # Compute Gaussians
    # Vectorized computation for speed
    g1 = np.exp(-((x_flat - c1) ** 2) / (2.0 * s1 ** 2))
    g2 = np.exp(-((x_flat - c2) ** 2) / (2.0 * s2 ** 2))
    g3 = np.exp(-((x_flat - c3) ** 2) / (2.0 * s3 ** 2))
    
    # Combine
    predictions = a1 * g1 + a2 * g2 + a3 * g3 + a0
    
    return predictions
