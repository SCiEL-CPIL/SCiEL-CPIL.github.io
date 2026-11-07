"""
Strategy: Piecewise-Linear Physical Model with Smooth Hinges.
Instead of a generic function approximator (like RBF), we use a model that
explicitly encodes the physics of aluminium: an elastic region followed by
plastic deformation (yielding and hardening).

The model is:
sigma(eps, T) = E_eff(T) * eps + sigma_y(T) * h(eps - eps_y, k) + hardening(eps - eps_y, T)

Where:
- E_eff(T) is the effective elastic modulus, which decreases with temperature.
- sigma_y(T) is the yield stress, which also decreases with temperature.
- h(x, k) is a smooth hinge function that is 0 for x <= 0 and approximates x for x > 0.
  Specifically, h(x, k) = x * (1 - exp(-k * x)) for x > 0, and 0 otherwise.
  This ensures C1 continuity at the yield point.
- The hardening term captures the post-yield behavior, often modeled as a
  saturating function or a lower slope.

Parameters (10 total):
p0: Elastic Modulus E0 (at T=0)
p1: Temperature coefficient for E (softening)
p2: Yield Stress Y0 (at T=0)
p3: Temperature coefficient for Y (softening)
p4: Yield Strain eps_y (threshold)
p5: Hinge sharpness k (controls how quickly it transitions from elastic to plastic)
p6: Plastic Hardening Slope H0
p7: Temperature coefficient for H
p8: Saturation Stress for hardening (if used) or offset
p9: Global bias/offset

We structure the function to be linear in some parameters and non-linear in others,
but the hinge structure helps guide the optimizer. To ensure stability with BFGS
starting from all ones, we use soft constraints (tanh/sigmoid) to keep parameters
in reasonable physical ranges if they drift, or simply rely on the smoothness
of the hinge to provide good gradients.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a piecewise-linear physical model with smooth hinges.
    Inputs:
        strain: array of strain values
        temp: array of temperature values
        params: array of 10 parameters
    Output:
        array of predicted stress values
    """
    # Deadline guard
    start_time = time.time()
    timeout = 110.0 # 10s margin
    
    eps = np.asarray(strain, dtype=np.float64).flatten()
    T = np.asarray(temp, dtype=np.float64).flatten()
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Sanitize parameters to prevent numerical issues
    # We use tanh to bound some parameters if they grow too large, 
    # but for BFGS, it's often better to let them float and rely on the loss landscape.
    # However, we must ensure the hinge sharpness k (p5) is positive.
    k = np.abs(p5) + 1e-6
    
    # 1. Elastic Modulus E(T)
    # E(T) = E0 * (1 - c_T * T)
    # If p1 is negative, it softens. We assume p1 is the coefficient.
    # To allow for both softening and hardening (though Al softens), we just use linear.
    E_T = p0 * (1.0 - p1 * T)
    
    # 2. Yield Stress Y(T)
    Y_T = p2 * (1.0 - p3 * T)
    
    # 3. Plastic Strain Calculation
    # Plastic strain is the part of total strain beyond yield.
    # We use a smooth hinge to avoid discontinuities in the derivative.
    # h(x) = x * (1 - exp(-k*x)) for x > 0
    # This is 0 at x=0, and approaches x as x->inf.
    # It is C1 continuous.
    x = eps - p4
    # np.where is not differentiable at the boundary, but the function values match.
    # For BFGS, we need gradients. A better smooth approximation for x>0 is:
    # h(x) = 0.5 * (x + sqrt(x^2 + 1/k^2)) - 1/(2k)  ... this is for |x|
    # Let's stick to the exponential one but ensure it's smooth everywhere.
    # Actually, x * (1 - exp(-k*x)) is smooth everywhere if we define it for x<0 too?
    # If x < 0, exp(-k*x) > 1, so (1-exp(-k*x)) < 0. x is negative. Product is positive?
    # No, physically plastic strain should be 0 for x<0.
    # Let's use a smooth ReLU-like function:
    # h(x) = x / (1 + exp(-k*x)) ? No, that's sigmoid.
    # Let's use: h(x) = 0.5 * (x + sqrt(x^2 + 1/k^2)) - 1/(2k) is for absolute value.
    # For one-sided:
    # We can use: h(x) = x * (1 - exp(-k * x)) for x > 0, and 0 for x <= 0.
    # To make it differentiable, we can use a smooth approximation:
    # h(x) = (x * (1 - exp(-k*x))) * sigmoid(k*x * 10) ? 
    # Simpler: Just use the exponential form and clip the negative part to 0 using a soft max?
    # Actually, the standard smooth hinge is:
    # h(x) = 0.5 * (x + sqrt(x^2 + 1/k^2)) - 1/(2k) is NOT one-sided.
    # Let's use: h(x) = x * (1 - exp(-k*x)) for all x, but note that for x<0, 
    # this function becomes positive and convex, which is unphysical.
    # To force it to 0 for x<0, we can multiply by a smooth step: S(x) = 1/(1+exp(-k*x)).
    # h_smooth(x) = x * (1 - exp(-k*x)) * S(x) ?
    # Let's try a different smooth hinge:
    # h(x) = 1/k * (exp(k*x) - 1) for x > 0? No.
    
    # Let's use the "softplus" based hinge:
    # h(x) = softplus(k*x) - 1/k
    # softplus(u) = log(1 + exp(u))
    # h(0) = log(2) - 1/k. Not zero.
    # Adjusted: h(x) = softplus(k*x) - softplus(0) = log(1+exp(k*x)) - log(2)
    # h(0) = 0.
    # h'(x) = exp(k*x) / (1 + exp(k*x)) = sigmoid(k*x).
    # h'(0) = 0.5.
    # This is a very smooth, C-infinity hinge.
    # As x -> inf, h(x) ~ k*x - log(2). Slope is k.
    # As x -> -inf, h(x) ~ 0.
    # This is perfect.
    
    # Calculate h(eps - p4)
    # To avoid overflow in exp, we can use np.logaddexp(0, k*x) - np.log(2)
    # log(1 + exp(u)) = logaddexp(0, u)
    u = k * (eps - p4)
    # Clip u to prevent overflow in exp if k is large and eps is large
    # But logaddexp is stable.
    h_val = np.logaddexp(0.0, u) - np.log(2.0)
    
    # 4. Plastic Hardening Term
    # The term Y_T * h_val represents the yield stress activating after yield.
    # But we also want a hardening slope.
    # The derivative of h_val is sigmoid(k*x).
    # So the contribution to stress is Y_T * h_val.
    # The slope in the plastic region is Y_T * k * sigmoid(k*x).
    # This is not a constant slope. It starts at 0 and goes to Y_T * k.
    # This might be too flexible or not flexible enough.
    
    # Let's add a separate linear hardening term that activates after yield.
    # We can use the same hinge function for the hardening slope.
    # Stress = E_T * eps + H_T * h_val
    # Where H_T is the hardening coefficient.
    # But wait, if we use E_T * eps + H_T * h_val, the elastic slope is E_T.
    # The plastic slope is E_T + H_T * k * sigmoid(k*x).
    # This allows the slope to increase (hardening) or decrease (if H is negative).
    
    # Let's refine the parameters:
    # p0: E0
    # p1: E_temp_coeff
    # p2: Y0 (This will be scaled into the hardening term)
    # p3: Y_temp_coeff
    # p4: eps_y
    # p5: k (sharpness)
    # p6: H0 (Hardening modulus)
    # p7: H_temp_coeff
    # p8: Offset
    # p9: Extra term? Maybe a quadratic elastic term? Or just keep it simple.
    
    # Let's use:
    # sigma = E_T * eps + H_T * h_val + p8
    # This uses 8 parameters. We have 10.
    # We can add a temperature-dependent offset or a quadratic strain term.
    # Let's add a small quadratic elastic term to allow for slight curvature in elastic region.
    # sigma = E_T * eps + p9 * eps^2 + H_T * h_val + p8
    
    E_T = p0 * (1.0 - p1 * T)
    H_T = p6 * (1.0 - p7 * T)
    
    # Elastic part
    sigma_elastic = E_T * eps + p9 * eps**2
    
    # Plastic part
    sigma_plastic = H_T * h_val
    
    # Total stress
    sigma = sigma_elastic + sigma_plastic + p8
    
    # Safety check
    if not np.all(np.isfinite(sigma)):
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
        
    return sigma
# EVOLVE-BLOCK-END
