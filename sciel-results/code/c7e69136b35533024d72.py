"""
Strategy: Local Kernel Regression (RBF) with Adaptive Centers.

The previous strategies attempted to fit a global functional form (polynomial, rational, 
physical model) using all 10 parameters as global coefficients. This approach often 
suffers from bias if the true dynamics are complex or local, and variance if the global 
form is too rigid.

This new strategy reframes the problem as a local approximation:
1. We use a Radial Basis Function (RBF) network structure: 
   prediction = sum_{i=1}^{10} p_i * exp(-|| (x, v) - c_i ||^2 / (2 * sigma^2))
2. The 10 parameters `p` are the weights of the kernels.
3. The centers `c_i` are NOT free parameters (we only have 10 slots). Instead, we 
   "freeze" the centers at 10 specific data points. 
4. To choose good centers, we perform a lightweight internal search:
   a. Fit a simple linear model (OLS) to get initial residuals.
   b. Select the 10 data points with the largest absolute residuals (or spread out points) 
      as the RBF centers.
   c. Set sigma based on the typical distance between points.
5. The BFGS optimizer then only needs to fit the 10 weights `p`, which is a convex 
   problem (linear in weights) if the kernel matrix is well-conditioned. However, 
   since the centers are fixed based on the data distribution, this allows the model 
   to capture local variations that a global polynomial might miss, while avoiding 
   the ill-conditioning of high-degree polynomials.
   
   Wait, the task says "equation(<inputs>, params)". The `params` are fitted by the 
   evaluator. The evaluator does NOT know my internal center selection logic unless 
   I hardcode it or compute it inside `equation`. 
   
   CRITICAL CONSTRAINT: The evaluator calls `equation(X, params)` where `params` is 
   the result of BFGS. The BFGS minimizes MSE. 
   If I compute centers inside `equation`, they will depend on the input `X` passed 
   to `equation`. During fitting, `equation` is called with the training data `X_train`. 
   So I can compute centers based on `X_train` inside `equation`. 
   
   BUT, `equation` is also called with `X_test`. If I compute centers based on 
   `X_test`, the centers will be different, and the weights `p` (fitted on train) 
   will be applied to different kernels. This will break the model.
   
   Therefore, the centers MUST be fixed constants or derived from the data in a way 
   that is consistent between train and test? No, the test data is different. 
   The standard approach for RBF in this context is that the "basis functions" are 
   fixed. If I use data-dependent centers, they must be the SAME for train and test. 
   Since I don't have access to the test data during fitting, and I don't have access 
   to the train data when evaluating test (other than the input array passed to equation), 
   I cannot use data-dependent centers that vary between calls unless I assume the 
   input distribution is similar and use "universal" centers or centers derived from 
   the *range* of the data.
   
   However, the prompt says: "The inputs in training range...". 
   If I define centers based on the *known training range* (which is given in the 
   contract: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]), I can place 10 
   fixed centers in this space. This makes the model a fixed-basis RBF network. 
   The parameters `p` are the weights. This is a linear model in `p`. 
   
   Why is this different/better? 
   - Polynomials have global bias. 
   - RBFs with fixed centers in the data range can represent local bumps. 
   - With 10 centers, we have 10 degrees of freedom, same as before, but the 
     basis functions are localized. This is often more efficient for capturing 
     nonlinearities in a bounded domain than high-degree polynomials which 
     oscillate (Runge's phenomenon) or require many terms for local features.
   
   Let's place 10 centers in a grid over the training range.
   x_range: [-0.5728, 0.5275]
   v_range: [-0.1754, 0.2821]
   
   We can use a 3x3 grid plus 1 center? Or just 10 specific points. 
   Let's use a 3x3 grid of points and one center at the origin, or just 10 points 
   sampled from the range. 
   
   Actually, a better strategy for "10 parameters" is to use a **Neural Network-like 
   structure with fixed random weights**? No, that's too complex.
   
   Let's stick to **Fixed RBF Centers**.
   Centers: 10 points evenly distributed in the 2D input space.
   Sigma: Chosen to be about 1/3 of the domain width to ensure overlap.
   
   This is a linear model: y = Phi(x) @ p.
   BFGS will solve for p. Since it's linear, BFGS should converge very fast and 
   accurately to the global minimum (which is the OLS solution). 
   
   Why would this beat the polynomial? 
   - Condition number: RBF kernel matrices are often better conditioned than 
     Vandermonde matrices for high-degree polynomials, especially if centers are 
     well-spaced.
   - Flexibility: Localized bases can fit sharp changes better.
   
   Let's implement this.
"""

import numpy as np

# Precompute fixed RBF centers and sigma
# Training ranges:
# x: [-0.5728, 0.5275]
# v: [-0.1754, 0.2821]

# Create 10 centers. Let's use a 3x3 grid + 1 center, or just 10 points.
# A 3x3 grid has 9 points. Let's add one more at the center or random.
# Actually, let's just use 10 points from a 4x3 grid or similar.
# Or simply:
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

# 3x3 grid
x_grid = np.linspace(x_min, x_max, 3)
v_grid = np.linspace(v_min, v_max, 3)
centers = []
for xv in x_grid:
    for vv in v_grid:
        centers.append([xv, vv])
# We have 9 centers. Add one more at the mean or random.
centers.append([(x_min + x_max) / 2, (v_min + v_max) / 2])
centers = np.array(centers) # Shape (10, 2)

# Sigma: width of the domain is ~1.1 for x, ~0.45 for v.
# Let's scale inputs to make the kernel isotropic.
# Or just pick a sigma that works.
# If we don't scale, the kernel is anisotropic. 
# Better to scale inputs inside equation.
x_scale = (x_max - x_min) / 2.0
v_scale = (v_max - v_min) / 2.0
x_center = (x_max + x_min) / 2.0
v_center = (v_max + v_min) / 2.0

# Normalize centers to [-1, 1] roughly
centers_norm = np.column_stack([
    (centers[:, 0] - x_center) / x_scale,
    (centers[:, 1] - v_center) / v_scale
])

# Sigma in normalized space. Let's use sigma = 0.8.
# exp(-d^2 / (2 * sigma^2))
SIGMA = 0.8

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with fixed centers.
    x, v: input arrays
    params: 10 weights
    """
    # Normalize inputs
    x_norm = (x - x_center) / x_scale
    v_norm = (v - v_center) / v_scale
    
    # Compute distances to centers
    # x_norm shape (N,), centers_norm shape (10, 2)
    # dist_sq shape (N, 10)
    dx = x_norm[:, np.newaxis] - centers_norm[:, 0][np.newaxis, :]
    dv = v_norm[:, np.newaxis] - centers_norm[:, 1][np.newaxis, :]
    
    dist_sq = dx**2 + dv**2
    
    # Kernel matrix K shape (N, 10)
    K = np.exp(-dist_sq / (2 * SIGMA**2))
    
    # Prediction = K @ params
    result = K @ params
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
