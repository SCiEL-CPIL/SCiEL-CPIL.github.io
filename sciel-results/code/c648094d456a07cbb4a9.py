"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Mixture of Experts (MoE) model.
    
    Strategy:
    We model the growth rate as a weighted sum of three linear experts.
    The weights are determined by a softmax gating network based on temperature and pH,
    which allows the model to switch between different physiological regimes:
    1. Expert 0: Low temperature / Acidic regime (limited growth)
    2. Expert 1: Optimal temperature / Neutral pH regime (high growth potential)
    3. Expert 2: High temperature / Alkaline regime (inhibition/death)
    
    The experts themselves are linear functions of b and s (and constants),
    capturing the dependence on population density and substrate.
    The non-linearity comes from the interaction between the regime selection
    (gates) and the linear responses within each regime.
    
    Parameters (10 total):
    - 3 params for gating biases (one per expert)
    - 3 params for gating scales (inverse temperature of softmax)
    - 4 params for the linear coefficients of the experts (shared across experts for simplicity? 
      No, let's give each expert a bias and a slope for s and b? 
      That would be 3 experts * 3 coeffs = 9 params. 
      Plus 3 gating biases = 12 params. Too many.
      
      Let's simplify:
      Experts are linear in s and b: E_k = w_k0 + w_k1 * s + w_k2 * b
      That's 3 coeffs per expert * 3 experts = 9 coeffs.
      Gating: g_k = exp(scale_k * (bias_k + h_k(temp, pH)))
      To keep it within 10 params:
      We can fix the scales or biases? 
      
      Alternative structure:
      Gate depends on temp and pH.
      Let's use a simpler gate:
      g_temp = exp(-((temp - center_temp)/width_temp)^2)
      g_ph = exp(-((pH - center_ph)/width_ph)^2)
      
      But we need 10 params.
      
      Let's try this structure:
      Expert 0 (Cold/Acid):  E0 = p0 + p1 * s
      Expert 1 (Optimal):    E1 = p2 + p3 * s + p4 * b
      Expert 2 (Hot/Alkaline): E2 = p5 + p6 * s
      
      Gating:
      We want weights w0, w1, w2 that sum to 1.
      Let's use a logistic switch based on temp and pH.
      w1 = sigmoid(p7 * (temp - p8)) * sigmoid(p9 * (pH - p10)) ... wait, 10 params max.
      
      Let's allocate:
      p0, p1: Expert 0 intercept, slope_s
      p2, p3, p4: Expert 1 intercept, slope_s, slope_b
      p5, p6: Expert 2 intercept, slope_s
      p7: Gate center temp
      p8: Gate width temp
      p9: Gate center pH
      p10: Gate width pH
      
      Wait, that's 11 params.
      
      Let's drop p4 (slope_b for Expert 1) or assume b effect is global?
      Or combine pH and Temp gates?
      
      Let's try:
      Gate is based on a "Stress Score" = (temp - p7)^2 / p8^2 + (pH - p9)^2 / p10^2
      Weight for Optimal (Expert 1) = exp(-Stress Score)
      Weight for others = (1 - Weight_Optimal) / 2
      
      This uses 4 params for the gate (p7, p8, p9, p10).
      Remaining 6 params for experts.
      Expert 0 (Stress): p0 + p1 * s
      Expert 1 (Optimal): p2 + p3 * s + p4 * b
      Expert 2 (Stress2 - maybe redundant? Or just one stress expert?)
      
      If we use 1 Optimal Expert and 1 General Stress Expert:
      Prediction = w_opt * E_opt + (1 - w_opt) * E_stress
      E_opt = p2 + p3 * s + p4 * b
      E_stress = p5 + p6 * s
      
      Total params:
      Gate: p7 (temp center), p8 (temp width), p9 (pH center), p10 (pH width) -> 4 params
      E_opt: p2, p3, p4 -> 3 params
      E_stress: p5, p6 -> 2 params
      Total: 4 + 3 + 2 = 9 params.
      We have 10. Add a global bias or slope for b in stress?
      E_stress = p5 + p6 * s + p7 * b ? No, p7 is used.
      
      Let's use:
      p0: E_stress intercept
      p1: E_stress slope_s
      p2: E_stress slope_b
      p3: E_opt intercept
      p4: E_opt slope_s
      p5: E_opt slope_b
      p6: Gate temp center
      p7: Gate temp width
      p8: Gate pH center
      p9: Gate pH width
      
      Total 10 params.
      
      Gate Logic:
      z_temp = (temp - p6) / p7
      z_ph = (pH - p8) / p9
      stress = z_temp**2 + z_ph**2
      w_opt = np.exp(-stress)
      w_stress = 1.0 - w_opt
      
      Prediction = w_opt * (p3 + p4*s + p5*b) + w_stress * (p0 + p1*s + p2*b)
      
      This is differentiable everywhere.
      BFGS should handle it well if initialized reasonably.
      Initial params all ones:
      p6=1, p7=1, p8=1, p9=1.
      Centers at 1, width 1.
      This might be far from optimal (Temp range 1-12, pH 1-12).
      But BFGS will move them.
      
      To help BFGS, we can scale inputs inside the function before passing to gate logic?
      No, the task says "keep their names and order". We can normalize inside.
      Let's normalize temp and pH to [0, 1] range for the gate calculation to make param scales more uniform.
      temp_norm = (temp - 1) / 11
      ph_norm = (pH - 1) / 11
      
      Then p6, p8 are centers in [0,1]. p7, p9 are widths in [0,1].
      This helps BFGS converge faster as gradients are better scaled.
      
      Similarly, s and b can be normalized for the linear parts to avoid large gradients?
      b in [1,5], s in [1,40].
      Let's normalize s and b too for the linear experts to keep param magnitudes similar.
      b_norm = (b - 1) / 4
      s_norm = (s - 1) / 39
      
      Then:
      E_opt = p3 + p4 * s_norm + p5 * b_norm
      E_stress = p0 + p1 * s_norm + p2 * b_norm
      
      This keeps all parameters in a similar range (roughly -2 to 2 for growth rates).
      
      Let's implement this.
    """
    
    # Normalize inputs to [0, 1] for better conditioning of BFGS
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    t_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 11.0
    
    # Extract parameters
    p0, p1, p2 = params[0], params[1], params[2]  # Stress expert: intercept, slope_s, slope_b
    p3, p4, p5 = params[3], params[4], params[5]  # Optimal expert: intercept, slope_s, slope_b
    p6, p7, p8, p9 = params[6], params[7], params[8], params[9] # Gate: t_center, t_width, ph_center, ph_width
    
    # Ensure widths are positive to avoid division by zero or negative variance in gaussian-like decay
    # We use exp(-stress), so width acts as denominator.
    # If width is negative, stress becomes negative, exp grows. 
    # BFGS might explore negative widths. 
    # To be safe, use abs(width) + small epsilon.
    t_width = np.abs(p7) + 1e-6
    ph_width = np.abs(p9) + 1e-6
    
    # Calculate gate weights
    z_temp = (t_norm - p6) / t_width
    z_ph = (ph_norm - p8) / ph_width
    
    stress = z_temp**2 + z_ph**2
    
    # w_opt is high when stress is low (near center)
    w_opt = np.exp(-stress)
    w_stress = 1.0 - w_opt
    
    # Calculate expert outputs
    e_opt = p3 + p4 * s_norm + p5 * b_norm
    e_stress = p0 + p1 * s_norm + p2 * b_norm
    
    # Combine
    prediction = w_opt * e_opt + w_stress * e_stress
    
    return prediction
# EVOLVE-BLOCK-END
