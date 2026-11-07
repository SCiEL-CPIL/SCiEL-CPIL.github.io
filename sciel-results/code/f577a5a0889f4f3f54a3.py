"""
Symbolic regression for a nonlinear oscillator combining Chebyshev polynomial basis
with physics-informed nonlinear terms.

Strategy: Use a hybrid basis that combines:
1. Complete Chebyshev polynomial basis (total degree 2) for numerical stability
2. Physics-motivated higher-order terms (cubic x, cubic v, interaction terms)
3. This gives the optimizer both well-conditioned low-order terms and 
   flexible higher-order nonlinearities specific to oscillator dynamics
"""
import numpy as np

# Normalization constants derived from training ranges
X_MIN = -0.5728
X_MAX = 0.5275
V_MIN = -0.1754
V_MAX = 0.2821

# Map inputs to [-1, 1] for Chebyshev polynomials
X_SCALE = 2.0 / (X_MAX - X_MIN)
X_SHIFT = -(X_MAX + X_MIN) / (X_MAX - X_MIN)

V_SCALE = 2.0 / (V_MAX - V_MIN)
V_SHIFT = -(V_MAX + V_MIN) / (V_MAX - V_MIN)

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression using a hybrid Chebyshev + physics-informed basis.
    
    Basis functions (where X, V are normalized to [-1, 1]):
    0: T0(X)T0(V) = 1 (constant)
    1: T1(X)T0(V) = X (linear position - restoring force)
    2: T0(X)T1(V) = V (linear velocity - damping)
    3: T2(X)T0(V) = 2X^2 - 1 (quadratic position)
    4: T1(X)T1(V) = X*V (interaction term)
    5: T0(X)T2(V) = 2V^2 - 1 (quadratic velocity)
    6: X^3 (cubic position - Duffing nonlinearity, in original scale)
    7: V^3 (cubic velocity - nonlinear damping, in original scale)
    8: X^2*V (parametric excitation, in original scale)
    9: X*V^2 (velocity-dependent force, in original scale)
    
    The first 6 terms form a complete Chebyshev basis of total degree 2,
    providing good numerical conditioning. The last 4 terms add physics-motivated
    higher-order nonlinearities in the original input scale, allowing the model
    to capture Duffing-type behavior and nonlinear damping effects.
    """
    # Normalize inputs to [-1, 1] for Chebyshev polynomials
    X = x * X_SCALE + X_SHIFT
    V = v * V_SCALE + V_SHIFT
    
    # Compute Chebyshev polynomials for X (up to degree 2)
    T0_X = np.ones_like(X)
    T1_X = X
    T2_X = 2.0 * X**2 - 1.0
    
    # Compute Chebyshev polynomials for V (up to degree 2)
    T0_V = np.ones_like(V)
    T1_V = V
    T2_V = 2.0 * V**2 - 1.0
    
    # First 6 basis functions: Complete Chebyshev basis of total degree 2
    b0 = T0_X * T0_V          # 1
    b1 = T1_X * T0_V          # X
    b2 = T0_X * T1_V          # V
    b3 = T2_X * T0_V          # 2X^2 - 1
    b4 = T1_X * T1_V          # X*V
    b5 = T0_X * T2_V          # 2V^2 - 1
    
    # Last 4 basis functions: Physics-motivated higher-order terms in original scale
    # These capture nonlinear oscillator dynamics
    b6 = x**3                 # Cubic restoring force (Duffing)
    b7 = v**3                 # Cubic nonlinear damping
    b8 = x**2 * v             # Parametric excitation
    b9 = x * v**2             # Velocity-dependent force
    
    # Linear combination
    prediction = (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
    
    return prediction
