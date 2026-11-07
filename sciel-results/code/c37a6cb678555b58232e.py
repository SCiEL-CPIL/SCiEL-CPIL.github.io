import numpy as np

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Smooth Rational Function Model (Padé Approximation Style).
    
    Instead of a high-order polynomial which can oscillate and struggle to 
    capture the flat yield plateau without overshooting, we use a rational 
    function structure that naturally bounds the plastic response or 
    transitions smoothly between elastic and plastic regimes.
    
    The model is structured as:
    Stress = E(t) * strain / (1 + k(t) * strain^2) + H(t) * strain^2 / (1 + k(t) * strain^2)
    
    Wait, a simpler and more robust structure for stress-strain with a 
    yield point is often modeled by a hyperbolic or rational term that 
    saturates or changes slope.
    
    Let's use a model that is linear in the plastic hardening parameters 
    to ensure BFGS convergence, but uses a smooth non-linear basis for 
    the elastic-to-plastic transition.
    
    Proposed Model:
    Stress = E(t) * strain * (1 - sigmoid(k*(strain - eps_y))) 
           + (E(t)*eps_y + H(t)*(strain - eps_y)) * sigmoid(k*(strain - eps_y))
    
    This is non-linear in k and eps_y. BFGS might struggle if k is large.
    
    Alternative: A rational function that mimics the J2 plasticity flow 
    without explicit piecewise logic.
    
    Let's try a specific rational form that is known to fit metal stress-strain 
    well:
    Stress = (A(t) * strain + B(t) * strain^2) / (1 + C(t) * strain + D(t) * strain^2)
    
    This is linear in A and B if C and D are fixed, but here all are params.
    However, the current program is linear in params. The inspiration suggests 
    that linear-in-params is better for BFGS.
    
    Let's stick to a basis function approach but choose better basis functions 
    than pure polynomials. We can use "bump" functions or localized polynomials 
    that are still linear in the coefficients.
    
    New Strategy: Localized Polynomial Basis (B-Spline like)
    Use a set of basis functions that are smooth and localized around the 
    yield point, allowing the model to fit the elastic part and plastic part 
    independently without the global oscillations of a high-order polynomial.
    
    Basis Functions:
    1. Elastic part: s (linear)
    2. Plastic part: (s - s0)_+ approximated by softplus or smooth max.
    
    Since we want to avoid non-linearity in params for BFGS stability, 
    we can use a fixed smooth max operator with a fixed width, and let 
    the params scale the basis functions.
    
    Model:
    Stress = c0(t) + c1(t)*s + c2(t)*softplus(s - s_yield) + c3(t)*softplus(s - s_yield)^2
    
    But s_yield is a param? If s_yield is a param, it's non-linear.
    
    Let's look at the inspiration: "Linear-in-parameters model eliminates local minima".
    The current program is linear in params. The score dropped from 2.2 to 2.5? 
    Wait, the current program score is 2.55. The parent was 1.56. 
    The diff shows the parent used a Piecewise Linear (PWL) model with smooth max.
    The current program uses a High-Order Polynomial.
    The parent (1.56) was better than the current (2.55).
    
    So the strategy should be to return to a structure similar to the parent 
    (Piecewise/Smooth Max) but improved.
    
    The parent used:
    Stress = E(t) * strain + Hardening(t) * <strain - YieldStrain(t)>
    
    This is linear in E and Hardening. YieldStrain is a param, making it non-linear 
    in YieldStrain. However, if the smooth max is wide enough, BFGS can handle it.
    
    Let's try a different structural change: 
    Use a **Neural Network-like Activation** (e.g., Swish or GELU) applied to a 
    linear projection, but since we only have 10 params, we can't do a full net.
    
    Instead, let's use a **Sum of Smoothed Step Functions** (Smoothed Heaviside) 
    with fixed breakpoints, and let the params be the slopes.
    
    Fixed breakpoints: e.g., 0.002, 0.01, 0.05, 0.2, 0.5.
    Basis functions:
    b0 = 1
    b1 = s
    b2 = H(s - 0.002)
    b3 = H(s - 0.01)
    b4 = H(s - 0.05)
    b5 = H(s - 0.2)
    
    And temperature dependence:
    Stress = sum_{i=0}^5 (a_i + b_i * temp) * b_i(s)
    
    Total params: 6 * 2 = 12. Too many. We have 10.
    
    Let's reduce:
    b0 = 1
    b1 = s
    b2 = H(s - s_y1)
    b3 = H(s - s_y2)
    
    Params:
    a0, b0 (for b0) -> 2
    a1, b1 (for b1) -> 2
    a2, b2 (for b2) -> 2
    a3, b3 (for b3) -> 2
    s_y1, s_y2 -> 2
    
    Total 10.
    
    This is non-linear in s_y1, s_y2. But only 2 non-linear params. 
    The rest are linear. This should be much better than a 4th order polynomial 
    which has global effects.
    
    We will use a smooth Heaviside: H(x) = 0.5 * (1 + tanh(x / sigma)).
    To keep it stable, we can fix sigma to a small value relative to the scale, 
    or use a fixed sigma like 0.001.
    
    However, if s_y1 and s_y2 are free, BFGS might struggle if they are far apart 
    or if the gradient is flat.
    
    Alternative: Fix the breakpoints and use 10 linear params.
    Breakpoints: 0.001, 0.005, 0.02, 0.1, 0.5.
    Basis:
    b0 = 1
    b1 = s
    b2 = H(s - 0.001)
    b3 = H(s - 0.005)
    b4 = H(s - 0.02)
    
    We need 10 params.
    Let's have temperature dependence for the main elastic and plastic parts.
    
    Stress = (a0 + b0*t) * 1 
           + (a1 + b1*t) * s 
           + (a2 + b2*t) * H(s - 0.002)
           + (a3 + b3*t) * H(s - 0.01)
           + a4 * H(s - 0.1)
           + a5 * H(s - 0.5)
    
    Params: a0,b0,a1,b1,a2,b2,a3,b3,a4,a5 -> 10 params.
    All linear in params!
    This avoids the non-linearity of yield strain.
    The fixed breakpoints allow the model to "kink" at specific strains.
    
    This is a structural change from the global polynomial to a localized, 
    linear-in-params basis with fixed smooth steps.
    """
    s = strain
    t = temp
    p = params
    
    # Fixed smooth step functions (Heaviside approximations)
    # Using tanh for smoothness.
    # Scale factor for tanh to control sharpness.
    # Strain range is approx 0 to 1.
    # Yield is typically small (0.001 - 0.01).
    
    # We define 4 smooth steps at fixed locations.
    # Locations chosen to cover elastic, yield, and hardening regions.
    
    # b0: Constant
    b0 = np.ones_like(s)
    
    # b1: Linear (Elastic)
    b1 = s
    
    # b2: Step at 0.002 (Early yield)
    # Using a fixed width for the transition to keep it smooth for BFGS
    # tanh(x/eps)
    eps = 0.0005
    b2 = 0.5 * (1 + np.tanh((s - 0.002) / eps))
    
    # b3: Step at 0.01 (Full yield/Plastic flow)
    b3 = 0.5 * (1 + np.tanh((s - 0.01) / eps))
    
    # b4: Step at 0.1 (Hardening region)
    b4 = 0.5 * (1 + np.tanh((s - 0.1) / eps))
    
    # b5: Step at 0.5 (Late hardening)
    b5 = 0.5 * (1 + np.tanh((s - 0.5) / eps))
    
    # Assign params to coefficients with temperature dependence where appropriate.
    # p[0], p[1]: Constant term (a0 + b0*t)
    # p[2], p[3]: Linear term (a1 + b1*t)
    # p[4], p[5]: Early yield term (a2 + b2*t)
    # p[6], p[7]: Full yield term (a3 + b3*t)
    # p[8]: Hardening term 1 (a4) - assume less temp dependence or fixed
    # p[9]: Hardening term 2 (a5) - assume less temp dependence or fixed
    
    # To maximize flexibility with 10 params, we allow temp dependence for 
    # the first 4 terms (8 params) and use the last 2 for the higher strain terms.
    
    c0 = p[0] + p[1] * t
    c1 = p[2] + p[3] * t
    c2 = p[4] + p[5] * t
    c3 = p[6] + p[7] * t
    c4 = p[8]
    c5 = p[9]
    
    # Linear combination
    stress = c0 * b0 + c1 * b1 + c2 * b2 + c3 * b3 + c4 * b4 + c5 * b5
    
    return stress
