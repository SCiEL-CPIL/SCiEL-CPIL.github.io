import numpy as np
import time

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Decoupled Elastic-Plastic Model with Internal Pre-optimization.
    
    The stress-strain curve of aluminium typically has:
    1. An initial linear elastic region (Hooke's law).
    2. A yield point where plastic deformation begins.
    3. A plastic hardening region.
    
    The current program uses a single global polynomial. This strategy uses a 
    physically motivated piecewise-smooth model:
    Stress = E(t) * strain + H(t) * (strain - yield_strain(t))_+^n
    
    However, the `(x)_+` term is non-smooth. To allow BFGS to work efficiently 
    on the *entire* parameter vector, we use a smooth approximation:
    Stress = E(t) * strain + H(t) * softplus(k * (strain - yield_strain(t)))^n
    
    But wait, the task says the evaluator fits `params` using BFGS starting from 
    all ones. We cannot change the fitting procedure. We must provide a function 
    that is smooth and differentiable.
    
    Let's rethink. The "different strategy" should be about the *function form* 
    that makes the optimization landscape easier for BFGS, or a different 
    mathematical structure.
    
    A common issue with polynomials is that they don't naturally respect the 
    "flatness" of the yield plateau or the "linearity" of the elastic region 
    without high-order terms fighting each other.
    
    Let's try a **Smooth Min / Softplus based model** that explicitly separates 
    elastic and plastic components using a smooth switch, but parameterized 
    such that BFGS can navigate it.
    
    Actually, a very powerful strategy for symbolic regression with BFGS is to 
    use **exponential or sigmoid-based saturation** functions which are naturally 
    smooth and bounded, avoiding the Runge phenomenon of polynomials.
    
    Model:
    Stress = (E * strain) * (1 - sigmoid(k * (strain - s_y))) + (E * s_y + H * (strain - s_y)^p) * sigmoid(k * (strain - s_y))
    
    This is getting complex and might have many local minima.
    
    Let's look at Inspiration 1: Rational Function (Padé-like). It scored 4.2171, 
    which is better than the current 2.5573.
    
    Let's try a different family: **Exponential Saturation**.
    
    Stress = A(t) * (1 - exp(-B(t) * strain)) + C(t) * strain
    
    This model captures:
    - Linear growth at small strain (approx C*t + A*B*t)
    - Saturation or slower growth at large strain if A is negative? No, 
      1-exp(-B*s) goes to 1. So it adds a constant.
      If we use exp(B*s), it grows exponentially.
      
    Let's try: Stress = C(t) * strain + A(t) * (exp(B(t) * strain) - 1)
    
    This can model hardening.
    
    However, the most robust "different" strategy that often outperforms 
    polynomials in physical curves is the **Power Law** or **Neural Network 
    style** but we are limited to 10 params.
    
    Let's go with a **Rational Function** as suggested by the inspiration, but 
    specifically designed to have a numerator that allows a linear term and a 
    denominator that controls the curvature/yield.
    
    Model:
    Stress = (p0 + p1*strain + p2*strain^2 + p3*strain^3) / (1 + p4*strain + p5*strain^2)
    
    This is a [3/2] Padé approximant. It has 6 parameters for the strain part.
    We have 4 more parameters for temperature dependence.
    
    Let's define:
    Numerator Coeffs:
    N0 = p0 + p1*temp
    N1 = p2 + p3*temp
    N2 = p4 + p5*temp
    N3 = p6 + p7*temp
    
    Denominator Coeffs:
    D1 = p8 + p9*temp
    D2 = 1.0 (fixed to 1 for scale, or maybe another param? No, we only have 10 params total)
    
    Wait, we have 10 params total.
    If we use temp-dependent coefficients for both N and D, we run out of params.
    
    Let's simplify.
    Stress = (p0 + p1*strain + p2*strain^2) / (1 + p3*strain + p4*strain^2)
    
    This has 5 params. We have 5 left for temp dependence.
    
    Let's make the coefficients linear in temp:
    c0 = p0 + p1*temp
    c1 = p2 + p3*temp
    c2 = p4 + p5*temp
    c3 = p6 + p7*temp
    c4 = p8 + p9*temp
    
    Stress = (c0 + c1*strain + c2*strain^2) / (1 + c3*strain + c4*strain^2)
    
    This is a [2/2] Padé approximant with temperature-dependent coefficients.
    It is smooth, differentiable, and rational functions are often better at 
    capturing asymptotic behavior and plateaus than polynomials.
    
    Let's implement this.
    """
    s = strain
    t = temp
    p = params
    
    # Unpack parameters
    # Numerator coefficients
    n0 = p[0] + p[1] * t
    n1 = p[2] + p[3] * t
    n2 = p[4] + p[5] * t
    
    # Denominator coefficients (constant term is 1)
    d1 = p[6] + p[7] * t
    d2 = p[8] + p[9] * t
    
    # Calculate numerator and denominator
    # Using Horner's method for stability
    num = n2
    num = num * s + n1
    num = num * s + n0
    
    den = d2
    den = den * s + d1
    den = den * s + 1.0
    
    # Avoid division by zero
    # The denominator should be non-zero in the physical range.
    # If it gets close to zero, we might have issues.
    # For safety, we can add a small epsilon, but that breaks differentiability.
    # Given the inputs are in [-0.001, 0.96] and temp in [0.06, 1], 
    # and params start at 1, den = 1 + 1*s + 1*s^2 is always positive.
    # BFGS will try to find good params.
    
    stress = num / den
    
    return stress
