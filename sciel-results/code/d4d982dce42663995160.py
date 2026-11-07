"""
Strategy: Sparse Mixture of Gaussians (SMoG) with Optimized Centers.

Rationale:
The previous RBF model used fixed centers, which limited its ability to capture the true
optimal growth conditions if they differed from the hardcoded biological priors. 
This new approach treats the centers as learnable parameters. 
We model the growth rate as a weighted sum of 4 Gaussian kernels in 3D space 
(Temperature, pH, Substrate).

The parameters are:
- 3 shared widths (sigma_t, sigma_p, sigma_s) to ensure numerical stability and reduce 
  the dimensionality of the non-convex part of the optimization.
- 12 center coordinates (3 per Gaussian) to allow the model to find the peaks.
- 4 amplitudes (w_1..w_4) for the linear combination.

Total parameters: 3 + 12 + 4 = 19? 
Wait, the limit is 10 parameters.
Let's re-evaluate.
We have 10 parameters.
Option A: 2 Gaussians.
- 2 widths (shared t, p, s? No, maybe 1 shared width? Or 3 shared widths?)
- If 1 shared width: 1 param.
- Centers: 2 * 3 = 6 params.
- Amplitudes: 2 params.
- Density correction: 1 param (linear in b).
Total: 1 + 6 + 2 + 1 = 10.

This fits the budget. A 2-component mixture is often sufficient for bimodal or 
asymmetric responses. If the response is more complex, the widths can be large 
to effectively act as broader basis functions, or the centers can move to capture 
different regimes.

To improve robustness, we will use:
- 1 shared isotropic width (sigma) in normalized space.
- 2 Gaussian centers: (t1, p1, s1) and (t2, p2, s2).
- 2 Amplitudes: w1, w2.
- 1 Density coefficient: c_b (linear term: 1 + c_b * b_norm).

Total: 1 + 6 + 2 + 1 = 10 parameters.

This is structurally different from the previous 4-center fixed-RBF model because:
1. Centers are optimized (not fixed).
2. Fewer, more flexible components.
3. Shared width enforces a specific scale of interaction, reducing local minima traps 
   associated with independent widths per center.
"""

import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a 2-component Sparse Mixture of Gaussians.
    
    Inputs:
    b: Population density
    s: Substrate
    temp: Temperature
    pH: pH level
    params: np.ndarray of length 10
        p[0]: Shared width (sigma) in normalized space
        p[1:4]: Center 1 (t1, ph1, s1) in normalized space
        p[4:7]: Center 2 (t2, ph2, s2) in normalized space
        p[7]: Amplitude 1 (w1)
        p[8]: Amplitude 2 (w2)
        p[9]: Density linear coefficient (c_b)
    """
    # Normalize inputs to [0, 1] range to make width parameters scale-invariant
    # Ranges estimated from task description:
    # b: [1, 5] -> [0, 1]
    # s: [1, 5] -> [0, 1]
    # temp: [1, 40] -> [0, 1]
    # pH: [2, 12] -> [0, 1]
    
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0
    
    p = params
    
    # Extract parameters
    sigma = np.abs(p[0]) + 1e-4  # Ensure positive width, avoid division by zero
    c1 = p[1:4]  # Center 1: [t, ph, s]
    c2 = p[4:7]  # Center 2: [t, ph, s]
    w1 = p[7]
    w2 = p[8]
    c_b = p[9]
    
    # Calculate squared distances for each point to each center
    # Input shapes: (N,)
    # Center shapes: (3,)
    
    # Distances for Center 1
    dt1 = t_n - c1[0]
    dph1 = ph_n - c1[1]
    ds1 = s_n - c1[2]
    dist2_1 = (dt1**2 + dph1**2 + ds1**2) / (2.0 * sigma**2)
    
    # Distances for Center 2
    dt2 = t_n - c2[0]
    dph2 = ph_n - c2[1]
    ds2 = s_n - c2[2]
    dist2_2 = (dt2**2 + dph2**2 + ds2**2) / (2.0 * sigma**2)
    
    # Gaussian values
    # Clip the exponent to prevent underflow/overflow issues in exp, 
    # though with normalized inputs and reasonable sigma, it should be stable.
    g1 = np.exp(-dist2_1)
    g2 = np.exp(-dist2_2)
    
    # Weighted sum
    growth_term = w1 * g1 + w2 * g2
    
    # Density correction
    # Linear in b_n to capture inhibition or stimulation
    density_term = 1.0 + c_b * b_n
    
    result = growth_term * density_term
    
    return result
