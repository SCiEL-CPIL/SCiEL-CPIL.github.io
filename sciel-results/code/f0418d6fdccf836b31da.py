"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a flexible additive basis expansion.
    
    Combines the mechanistic insights from inspiration 1 (Monod saturation,
    Gaussian temperature/pH optima, density inhibition) with the explicit
    interaction structure that allows BFGS to navigate the loss landscape.
    
    The model is purely linear in params, which makes the optimization problem
    convex (least squares), allowing BFGS to reliably find the global optimum.
    
    Basis functions:
    0: bias (p0)
    1: Monod saturation s/(1+s) (p1)
    2: Gaussian temperature peak exp(-(temp-37)^2 / 100) (p2)
    3: Gaussian pH peak exp(-(pH-7)^2 / 1) (p4)
    4: Density inhibition exp(-p6*b) -> wait, this makes it non-linear in p6.
    
    Correction: To keep it linear in params for convex optimization, we should
    avoid exp(-p6*b) where p6 is a parameter. Instead, we can use a fixed
    inhibition scale or a polynomial term. However, inspiration 1 used exp(-p6*b)
    and scored 1.3545. The non-linearity in p6 is the only source of non-convexity.
    Since BFGS can handle mild non-convexity, and the other terms are linear,
    this might be okay. But to be safer and potentially faster/better, let's look
    at inspiration 2 which was fully linear and scored 1.1517. Inspiration 1 was better.
    
    Let's stick close to Inspiration 1's structure but refine the widths to be
    fixed or squared to ensure stability, and ensure the linear terms are
    well-scaled.
    
    Actually, let's look at the inputs:
    b: [1, 4.998]
    s: [1, 4.999]
    temp: [1.002, 39.98]
    pH: [2, 12]
    
    Inspiration 1:
    f_s = s / (1.0 + s)
    f_temp = p2 * np.exp(-((temp - 37.0)**2) / (p3**2 + 1e-6))
    f_ph = p4 * np.exp(-((pH - 7.0)**2) / (p5**2 + 1e-6))
    f_b = np.exp(-p6 * b)
    f_st = p7 * f_s * (temp - 37.0)
    f_sp = p8 * f_s * (pH - 7.0)
    f_tp = p9 * (temp - 37.0) * (pH - 7.0)
    y = p0 + p1 * f_s + f_temp + f_ph + f_b + f_st + f_sp + f_tp
    
    Note: f_b = exp(-p6*b) is non-linear in p6.
    f_temp is non-linear in p3 (width).
    f_ph is non-linear in p5 (width).
    
    This mixed structure worked well. I will use this structure but ensure
    numerical stability and perhaps add a linear term for b to allow for
    a more flexible density effect if the exponential inhibition isn't the
    only factor. However, we only have 10 params.
    
    Let's refine the basis to be more robust.
    
    p0: bias
    p1: coeff for Monod(s)
    p2: amplitude for Temp Gaussian
    p3: width for Temp Gaussian (squared in denom)
    p4: amplitude for pH Gaussian
    p5: width for pH Gaussian (squared in denom)
    p6: rate for Density Inhibition exp(-p6*b)
    p7: coeff for interaction s * (temp - 37)
    p8: coeff for interaction s * (pH - 7)
    p9: coeff for interaction (temp - 37) * (pH - 7)
    
    This is exactly Inspiration 1. Why did it score 1.3545?
    The current program scores 1.0.
    Inspiration 2 (linear) scored 1.1517.
    
    I will use the structure of Inspiration 1 but with slight modifications:
    1. Use fixed widths for the Gaussians? No, fitting widths allows the model
       to adapt to the specific spread of the data.
    2. Ensure the density term is robust. If p6 is negative, exp(-p6*b) grows.
       We can constrain p6 to be positive by using p6**2 + epsilon?
       If we use p6**2, then exp(-(p6**2)*b). This forces inhibition.
       If the data shows stimulation at low density, this might fail.
       But E. coli usually shows inhibition at high density (crowding).
       
    Let's try to combine the linear flexibility of Insp 2 with the mechanistic
    core of Insp 1.
    
    Revised Plan:
    Keep the mechanistic core of Insp 1.
    Replace the linear interaction terms with slightly more flexible ones if possible?
    No, stick to what worked.
    
    One potential issue: The starting point for BFGS is all ones.
    p3=1 -> width=1. Temp range is ~1 to 40. (37-1)^2/1 = 1296. exp(-1296) ~ 0.
    So the temp term is effectively 0 at start for most points, except near 37.
    This is fine.
    
    p5=1 -> width=1. pH range 2 to 12. (7-2)^2/1 = 25. exp(-25) ~ 0.
    
    p6=1 -> exp(-b). b is 1 to 5. exp(-1) to exp(-5).
    
    This seems reasonable.
    
    I will output the code based on Inspiration 1, as it is the highest scoring
    inspiration and fits the biological intuition well. I will add a small
    linear term for b if I can fit it, but I'm out of params.
    
    Wait, I can swap one interaction for a linear b term?
    The interactions f_st, f_sp, f_tp capture the coupling.
    If I remove f_tp (temp-pH interaction) and add a linear b term, would it be better?
    Temp and pH are likely independent in their effects on growth, or their interaction
    is weaker than the main effects. The substrate interactions are more likely
    to be significant (growth depends on both).
    
    Let's try:
    0: bias
    1: Monod(s)
    2: Amp Temp
    3: Width Temp
    4: Amp pH
    5: Width pH
    6: Rate Density Inhibition
    7: Linear b (to capture any non-exponential density effect or baseline)
    8: s * (temp - 37)
    9: s * (pH - 7)
    
    This drops the temp-pH interaction.
    
    Alternatively, keep Insp 1 exactly. It scored 1.3545.
    I will provide a slightly cleaned-up version of Insp 1 to ensure it's
    robust and fast.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure widths are positive and stable
    w_t = p3**2 + 1e-6
    w_ph = p5**2 + 1e-6
    
    # 1. Substrate: Monod-like saturation
    # Using s/(1+s) is robust and bounded [0,1] for s>0
    f_s = s / (1.0 + s)
    
    # 2. Temperature: Gaussian centered at 37 (optimal for E. coli)
    # p2 is amplitude, p3 controls width
    f_temp = p2 * np.exp(-((temp - 37.0)**2) / w_t)
    
    # 3. pH: Gaussian centered at 7.0
    # p4 is amplitude, p5 controls width
    f_ph = p4 * np.exp(-((pH - 7.0)**2) / w_ph)
    
    # 4. Density: Exponential inhibition
    # p6 is the rate constant. If p6 > 0, high density reduces growth.
    # If p6 < 0, it would increase, which is unlikely for density.
    # To be safe against BFGS wandering to negative p6 causing overflow,
    # we could use p6**2, but that prevents the optimizer from finding
    # a "no inhibition" state (p6=0) easily if the landscape is flat?
    # Actually, if p6=0, exp(0)=1, which is a constant.
    # Let's stick to p6. The range of b is small (1-5), so exp(-p6*b)
    # is unlikely to overflow unless p6 is huge negative.
    f_b = np.exp(-p6 * b)
    
    # 5. Interactions
    # Substrate modulates temperature sensitivity
    f_st = p7 * f_s * (temp - 37.0)
    
    # Substrate modulates pH sensitivity
    f_sp = p8 * f_s * (pH - 7.0)
    
    # Temp-pH interaction
    f_tp = p9 * (temp - 37.0) * (pH - 7.0)
    
    # Combine additively
    y = p0 + p1 * f_s + f_temp + f_ph + f_b + f_st + f_sp + f_tp
    
    return y
# EVOLVE-BLOCK-END
