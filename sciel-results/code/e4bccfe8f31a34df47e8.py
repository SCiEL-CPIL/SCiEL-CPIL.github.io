"""
Strategy: Neural Network Universal Approximator.
Instead of a fixed linear basis, we use a deep feedforward network with fixed random weights.
The parameters are the weights of the final layer and biases of all layers.
This allows the function to approximate any continuous function on [0.1, 6] with high precision.
The evaluator's BFGS will fit the final linear combination of these rich features.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Deadline guard
START_TIME = time.time()
DEADLINE = START_TIME + 140.0

# Fixed random weights for the hidden layers to ensure determinism and stability
# We use a fixed seed to generate the "architecture" once.
np.random.seed(42)

# Architecture:
# Input: 1
# Hidden 1: 20 neurons
# Hidden 2: 20 neurons
# Hidden 3: 20 neurons
# Output: 1
#
# Parameters layout (length 10):
# This is tricky because a standard NN has many more than 10 params.
# However, the task says "params, an array of length 10".
# This is a severe constraint. A 10-parameter NN is too small to be a universal approximator
# unless we use a very specific structure or if we interpret "params" differently.
#
# Wait, the task says: "The evaluator fits params, an array of length 10".
# This means I can ONLY use 10 free parameters.
#
# Re-evaluating the "Neural Network" strategy with only 10 params:
# A 10-parameter NN is essentially a very small model.
# However, we can use the 10 parameters to define the *structure* of the function
# or use them as coefficients in a more complex, non-linear basis that is *not* linear in the parameters.
#
# Better Strategy: **Non-Linear Basis with 10 Coefficients**
# The current program uses a linear combination of 10 basis functions.
# This is equivalent to linear regression. BFGS is overkill but works.
# The issue is that if the target function requires more than 10 basis functions, we fail.
#
# Alternative: **Rational Functions / Pade Approximants**
# We can define a function that is a ratio of two polynomials of degree 4.
# f(x) = (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) / (1 + b1*x + b2*x^2 + b3*x^3 + b4*x^4)
# This uses 9 parameters (a0..a4, b1..b4). The 10th can be a scale or offset.
# Rational functions can approximate functions with poles or complex behavior much better than polynomials
# within a limited parameter budget.
#
# Let's try a Pade approximant approach.
# f(x) = (P(x)) / (Q(x))
# P(x) = sum_{i=0}^{4} p_i x^i
# Q(x) = 1 + sum_{i=1}^{4} q_i x^i  (Fixed constant term 1 for identifiability)
# Total params: 5 + 4 = 9.
# We have 10. Let's add a global scale s.
# f(x) = s * (P(x)) / (Q(x))
#
# Or even better: **Sum of Gaussians/Radial Basis Functions**
# With 10 params, we can define, say, 2 Gaussians with 4 params each (center, width, amplitude, offset)
# and 2 global params.
# But Gaussian centers are hard to fit with BFGS if initialized poorly.
#
# Let's stick to a **High-Order Rational Function** or a **Mixed Basis** that is non-linear in params.
# Actually, the most powerful use of 10 params is likely a **Polynomial of Degree 9**?
# No, degree 9 polynomial has 10 coefficients.
# f(x) = c0 + c1*x + ... + c9*x^9
# This is linear in params. BFGS will solve it exactly (if it converges).
# Why did the current program score 23138?
# The current program uses: 1, x, x^2, x^3, x^4, 1/x, log(x), sqrt(x), exp(-x), sin(x).
# This is a mixed basis.
# If the target is a polynomial, the mixed basis might be ill-conditioned or not span the space well enough
# if the target has high-frequency components that x^9 would catch but sin(x) won't.
#
# Let's try a **Pure Polynomial of Degree 9**.
# It spans a larger space of smooth functions than the mixed basis if the target is smooth.
# Also, we can try **Chebyshev Polynomials** scaled to [0.1, 6].
# Chebyshev polynomials are better conditioned for approximation.
#
# Let's implement a Chebyshev basis of degree 9.
# T_0(x), T_1(x), ..., T_9(x)
# Mapped to [0.1, 6].
#
# Strategy:
# 1. Map x from [0.1, 6] to t in [-1, 1].
# 2. Use Chebyshev polynomials T_k(t) for k=0..9.
# 3. f(x) = sum_{k=0}^9 p_k T_k(t)
#
# This is still linear in params.
#
# What if the target is NOT smooth?
# The prompt says "noiseless".
#
# Let's look at the score: 23138. This is extremely high.
# Normalized MSE = 23138.
# This implies the model is completely wrong.
# Why?
# 1. The basis functions might be causing numerical instability (e.g., x^4 at x=6 is 1296, x^9 is huge).
# 2. BFGS might be failing to converge due to ill-conditioning.
# 3. The target function might be something like sin(10x) which requires high frequency.
#
# If the target is high frequency, a degree 9 polynomial is not enough.
#
# Let's try a **Fourier Series** approach with 10 parameters.
# f(x) = a0 + a1*sin(x) + b1*cos(x) + a2*sin(2x) + b2*cos(2x) + ...
# With 10 params, we can do up to 4 harmonics (a0, a1, b1, a2, b2, a3, b3, a4, b4) -> 9 params.
# Or 5 harmonics if we assume even/odd symmetry? No.
#
# Let's try a **Hybrid**:
# 4 params for a low-order polynomial (trend)
# 6 params for a sinusoidal component (3 harmonics: a1, b1, a2, b2, a3, b3)
#
# f(x) = (p0 + p1*x + p2*x^2) + (p3*sin(x) + p4*cos(x) + p5*sin(2x) + p6*cos(2x) + p7*sin(3x) + p8*cos(3x)) + p9
#
# This covers both smooth trends and periodic behavior.
#
# To make it more robust, let's use a **Scaled Domain** for the sinusoids to allow for higher frequencies?
# No, frequencies are fixed in the basis.
#
# Let's try **Gaussian Radial Basis Functions** with fixed centers.
# Centers: 0.1, 1.6, 3.1, 4.6, 6.0
# Widths: Fixed to 0.5
# f(x) = sum_{i=1}^5 p_i * exp(-(x-c_i)^2 / (2*0.5^2))
# This uses 5 params. We have 10.
# Let's use 10 centers?
# Centers: linspace(0.1, 6.0, 10)
# Widths: Fixed to 0.4
# f(x) = sum_{i=1}^10 p_i * exp(-(x-c_i)^2 / (2*0.4^2))
#
# RBFs are excellent universal approximators for smooth functions and can handle local variations.
# BFGS should be able to fit the amplitudes p_i easily because the basis functions are well-behaved (bounded, smooth).
#
# Let's implement this RBF strategy.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a sum of Gaussian Radial Basis Functions.
    params: [p_1, ..., p_10]
    x: Input array
    """
    # Ensure x is a 1D array for consistent processing
    x = np.asarray(x).flatten()
    
    # Fixed centers distributed across the input domain [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed width (sigma) for the Gaussians
    # A width of ~0.3-0.5 is good for resolving features in [0.1, 6]
    sigma = 0.4
    
    # Calculate distances
    # x shape: (N,), centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    # K shape: (N, 10)
    K = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination
    # params shape: (10,)
    pred = K @ params
    
    return pred

# EVOLVE-BLOCK-END
