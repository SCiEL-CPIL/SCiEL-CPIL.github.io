import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Adaptive Gaussian Radial Basis Functions (RBFs)
    
    Structural Difference from Current Program:
    The current program uses a Chebyshev polynomial basis, which is linear in the 
    parameters. This means the 'shape' of the basis functions is fixed, and only 
    their amplitudes (coefficients) are learned. While stable, this limits the 
    model's ability to adapt to local features (like sharp peaks or kinks) unless 
    the degree is very high (which causes numerical instability).
    
    This new strategy uses a non-linear basis: 5 Gaussian RBFs.
    The 10 parameters are split into:
    - 5 centers (mu_i)
    - 5 widths (sigma_i)
    - 5 amplitudes (A_i)
    Wait, that's 15 params. We only have 10.
    
    Revised Strategy: 5 RBFs with 2 params each? No, that's too rigid.
    
    Let's use 5 RBFs. 
    Params: 
    - p[0:5] are centers (mu).
    - p[5:10] are widths (sigma).
    - Amplitudes? We don't have enough params for independent amplitudes if we 
      use 5 RBFs with centers and widths.
      
    Alternative: 4 RBFs.
    Params:
    - p[0:4] centers
    - p[4:8] widths
    - p[8:10] global amplitude and offset? 
    This is still linear in the last two but non-linear in the first 8.
    
    Better Alternative: 5 RBFs, but fix the relative widths or use a single width?
    If we use a single width for all, we lose local adaptability.
    
    Let's try 5 RBFs with centers and amplitudes, but fixed widths?
    No, fixed widths is what the previous RBF inspiration (score 639k) did, and 
    it was worse than the Chebyshev (1.27B). Wait, 639k is better than 1.27B?
    Yes, lower is better. So 639k is a much better score than 1.27B.
    
    The current program (Chebyshev) scored 1.27B.
    The RBF inspiration scored 639k.
    The Sinusoid inspiration scored 832k.
    
    So the current program is actually performing *worse* than the simpler RBF 
    inspiration. The Chebyshev basis might be struggling because BFGS from 
    all-ones start with a linear model might converge to a poor local minimum 
    or the scaling is off, or the function is highly non-polynomial.
    
    Let's go back to RBFs but make them *adaptive* to beat the fixed-width RBFs.
    
    Model: Sum of 5 Gaussians.
    y = sum_{i=1 to 5} A_i * exp(-((x - mu_i)^2) / (2 * sigma_i^2))
    
    We have 10 params.
    Let's use 5 centers (mu_1..mu_5) and 5 widths (sigma_1..sigma_5).
    What about amplitudes?
    If we don't have amplitude params, we can't fit the magnitude well.
    
    Option: 4 RBFs.
    4 centers, 4 widths, 2 amplitudes? No.
    
    Option: 5 RBFs, but the amplitudes are learned by the BFGS as well?
    No, the `params` array is fixed length 10. The evaluator fits `params`.
    So `equation` must define the mapping from `params` to `y`.
    If the model is non-linear in params, BFGS will try to optimize all 10.
    
    Let's define:
    p[0:5] -> centers (mu)
    p[5:10] -> amplitudes (A)
    Widths: Fixed? If widths are fixed, it's the same as the 639k score.
    To improve, we need adaptive widths.
    
    Let's use 4 RBFs.
    p[0:4] -> centers
    p[4:8] -> widths
    p[8:10] -> amplitudes? Only 2 amplitudes for 4 RBFs? No.
    
    How about:
    5 RBFs.
    p[0:5] -> centers
    p[5:10] -> widths
    Amplitudes: Can we derive them? No.
    
    Maybe the function is simpler?
    If the function is a sum of peaks, we need amplitudes.
    
    Let's try a different non-linear basis: **Sigmoidal Basis** (like a neural net layer).
    y = sum_{i=1 to 5} A_i * sigmoid(w_i * (x - b_i))
    Params: 5 w_i, 5 b_i. No amplitudes.
    
    Let's look at the "Learnable-Width GP" inspiration (31k). That was the best.
    It likely had more parameters or a different setup.
    
    With only 10 params, we are constrained.
    Let's try **5 RBFs** with **shared width**.
    p[0:5] centers
    p[5:10] amplitudes
    Width is fixed.
    This is the 639k model.
    
    To beat 639k, we need better.
    What if we use **4 RBFs** with **individual widths** and **individual amplitudes**?
    4*3 = 12 params. Too many.
    
    What if we use **3 RBFs** with **individual widths** and **individual amplitudes**?
    3*3 = 9 params.
    p[0:3] centers
    p[3:6] widths
    p[6:9] amplitudes
    p[9] offset (bias)
    
    This allows each peak to have its own height and width. This is more flexible 
    than shared width (639k) and more stable than 5 RBFs with no amplitudes.
    
    Let's implement 3 Adaptive RBFs + 1 Bias.
    
    Numerical Stability:
    - Widths must be positive. We can use exp(p) or abs(p) + eps.
    - Centers can be anywhere.
    
    BFGS starting from all ones:
    mu_i = 1.0
    sigma_i = 1.0 (if using exp, then width=1)
    A_i = 1.0
    bias = 1.0
    
    This is a non-linear optimization problem. BFGS should handle it if the 
    landscape is not too rugged. 3 RBFs is a relatively low-dimensional space (10D).
    """
    
    start_time = time.time()
    TIME_LIMIT = 145.0

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    if time.time() - start_time > TIME_LIMIT:
        return np.zeros_like(x)

    # Extract parameters
    # 3 RBFs
    # p[0:3] centers
    # p[3:6] raw widths -> transform to positive
    # p[6:9] amplitudes
    # p[9] bias
    
    centers = p[0:3]
    raw_widths = p[3:6]
    amplitudes = p[6:9]
    bias = p[9]
    
    # Ensure widths are positive and not too small to avoid division by zero or exp overflow
    # Use softplus-like or exp
    # exp(x) can overflow if x is large. 
    # Let's clip raw_widths to avoid extreme values.
    # Or use 1 + exp(raw_widths) to ensure > 1? No, we want small widths too.
    # Let's use abs(raw_widths) + 1e-3.
    # If BFGS pushes width to 0, the RBF becomes a delta. This is good for spikes.
    # But numerically, exp(-large) -> 0. That's fine.
    # The risk is if width is very small, the gradient might be unstable.
    
    widths = np.abs(raw_widths) + 1e-6
    
    # Compute RBFs
    # x is (N,), centers is (3,)
    # We need (N, 3)
    
    x_col = x[:, np.newaxis]
    c_row = centers[np.newaxis, :]
    w_row = widths[np.newaxis, :]
    a_row = amplitudes[np.newaxis, :]
    
    # diff = x - mu
    diff = x_col - c_row
    
    # exp(-diff^2 / (2 * width^2))
    # To avoid overflow in division if width is tiny:
    # inv_w2 = 1.0 / (2.0 * w_row**2)
    # This can be huge.
    # Better: exp(-0.5 * (diff / w_row)**2)
    
    scaled_diff = diff / w_row
    exponents = -0.5 * scaled_diff * scaled_diff
    
    # Clip exponents to avoid underflow issues (though underflow to 0 is fine)
    # Overflow is not possible since exponent is negative.
    # But if w is tiny, scaled_diff is huge, square is huge, exp is 0.
    
    rbf_values = np.exp(exponents)
    
    # Weighted sum
    y = np.sum(rbf_values * a_row, axis=1) + bias
    
    if time.time() - start_time > TIME_LIMIT:
        return np.zeros_like(x)

    if not np.all(np.isfinite(y)):
        # Fallback to constant mean if something went wrong
        return np.full_like(x, bias)
        
    return y
