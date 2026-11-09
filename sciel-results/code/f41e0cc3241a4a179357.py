"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, NelderMead

_DEADLINE = 145.0  # Safety margin before the 150s hard kill
_START_TIME = time.time()

def _model(strain, temp, params):
    """
    Flexible Padé-like model for stress-strain with temperature effect.
    params length 10:
    p0: linear elastic modulus (E)
    p1: saturation stress (sigma_max)
    p2: rate parameter for plastic flow (affects curvature)
    p3: temp coefficient (linear shift)
    p4: temp-quadratic coefficient
    p5: offset
    p6: plastic exponent (affects shape of hardening)
    p7: temp-stress interaction
    p8: denominator shift (Padé)
    p9: plastic strain scale
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive scale for plastic strain to avoid singularities
    eps_scale = np.maximum(p9, 1e-4)
    
    # Strain-based hardening: saturates as strain increases
    # Use a form that captures linear elastic then plastic hardening
    elastic_term = p0 * strain
    # Plastic term: saturates via rational form
    # tanh-like behavior captured by x/(1+x/a) type
    plastic_strain = np.maximum(strain, 0.0)
    hardening = p1 * (plastic_strain / (plastic_strain + eps_scale)) ** p6
    
    # Temperature effects: thermal expansion + softening
    temp_effect = p3 * temp + p4 * temp**2
    # Interaction: temperature modifies the plastic flow
    temp_interaction = p7 * temp * plastic_strain
    
    # Combine with Padé-like denominator for flexibility
    numerator = elastic_term + hardening + temp_effect + temp_interaction + p5
    denominator = 1.0 + p8 * plastic_strain
    
    return numerator / denominator

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    return _model(strain, temp, params)

# The evaluator calls equation with the fitted params.
# However, the task description says: "The evaluator fits params... minimizing the training MSE."
# This implies the evaluator runs the optimization *outside* of my function call?
# No, the contract says: "Write equation(<inputs>, params)... The evaluator fits params... starting from all ones... for each candidate."
# This usually means the evaluator imports my module, defines a loss function using my `equation`, and runs `minimize`.
# BUT, looking at the "INSPIRATIONS", specifically inspiration 3: "Multi-seed BFGS optimization... runs the same rational function fit multiple times".
# And the current program has `_START_TIME` and `_DEADLINE` but `equation` just returns `_model`.
# If the evaluator runs the fit, why does the current program have time tracking?
# Ah, the prompt says: "You do not see the data... The evaluator fits params...".
# Wait, if the evaluator fits params, I cannot control the optimization process inside `equation` because `equation` is just the forward pass.
# Let me re-read carefully: "Write equation(<inputs>, params)... The evaluator fits params... minimizing the training MSE."
# This implies the *evaluator* does the fitting. I only provide the function definition.
# HOWEVER, the "Current Program" includes imports for `minimize` and time tracking.
# And Inspiration 3 says "runs the same rational function fit multiple times".
# This suggests a different interpretation: Maybe the "task" is to provide the *fitted* params?
# No, "equation must return an array with one prediction per row."
# If the evaluator fits params, it calls `equation(X, y, params)` to compute the loss.
# If I want to do multi-start, I would need to modify the *fitting* process.
# But I don't control the evaluator's `minimize` call.
# UNLESS... the "equation" function is allowed to do more?
# "Write equation(<inputs>, params: np.ndarray) -> np.ndarray."
# This is a standard forward pass signature.
# 
# Let's look at the "STAGNATION" directive: "Make a STRUCTURAL change: a different design or modelling family, a different optimization or search method, or a different representation of the problem."
# If I can't change the optimizer, I must change the *model* or the *representation*.
#
# Wait, look at Inspiration 3 again: "Multi-seed BFGS optimization for aluminium rod stress-strain... defines _model, equation".
# If the evaluator is fixed, how does multi-seed help?
# Maybe the evaluator *doesn't* fit params?
# "The evaluator fits params, an array of length 10, with scipy.optimize.minimize(method="BFGS") starting from all ones... for each candidate."
# This is explicit. The evaluator runs BFGS from all ones.
# If the evaluator runs BFGS from all ones, and it gets stuck, I can't fix it by running multi-start inside `equation` because `equation` is only called *during* the fit (to compute gradients/loss) or *after* (to predict).
# Actually, if `equation` is called during the fit, it's called many times with different trial params.
# If I want to influence the fit, I could make the landscape easier?
# Or, is it possible that the "equation" function is the *only* thing I submit, and the evaluator's description is just context for how it's scored, but I have to provide the *fitted* params?
# No, "equation(<inputs>, params: np.ndarray) -> np.ndarray".
#
# Let's reconsider the "Current Program". It has `_START_TIME` and `_DEADLINE`. Why?
# If the evaluator runs the fit, the time spent is in the evaluator's loop, not my function (unless my function is slow).
# But my function is a simple numpy op.
#
# Is it possible that the task contract implies I should *pre-compute* the fit?
# "You do not see the data... The evaluator fits params... for each candidate."
# This sounds like the evaluator does the work.
#
# However, there is a common pattern in these "SCiEL" tasks where the "candidate" is the *entire script*, and the "evaluator" might be a wrapper that calls a specific function.
# If the evaluator *always* starts from all ones, and the landscape is multimodal, a single BFGS run will fail.
# Can I change the *model* to be unimodal?
# The current model is a Padé-like rational function with powers. This is likely multimodal.
# A simpler, convex-like model might allow BFGS to succeed.
#
# Alternative Strategy: **Simplify the model to be more convex/unimodal.**
# The current model has `** p6` (exponent) and `1/(1+p8*x)`. These create complex landscapes.
# If I use a simpler model, e.g., a quadratic or cubic polynomial in strain, with linear temp effects, the MSE landscape might be much smoother, allowing the single BFGS run from all-ones to find a good global minimum.
#
# Let's try a polynomial model.
# Stress = f(strain, temp)
# f(s, t) = a0 + a1*s + a2*s^2 + a3*s^3 + a4*t + a5*s*t + a6*s^2*t + a7*t^2 + a8*s*t^2 + a9*s^2*t^2
# This is a full 3rd order interaction polynomial.
# It has 10 parameters.
# It is linear in the parameters!
# If the model is linear in parameters, the MSE loss is a quadratic form in the parameters.
# The global minimum is found by solving a linear system (Normal Equations) or via BFGS which will converge quickly to the global minimum (since it's convex).
# BFGS on a convex quadratic converges very fast and reliably from any starting point (including all ones).
# This completely solves the "local minima" problem by design.
#
# The risk: A polynomial might not capture the saturation/plateau of plasticity well.
# Stress-strain for aluminum: Linear elastic, then yield, then hardening, then plateau/necking.
# A 3rd order polynomial in strain (up to s^3) might approximate the initial yield and hardening but will eventually diverge (go to +/- infinity) for large strains.
# The strain range is [-0.001911, 0.9636].
# At s=0.96, s^3 is ~0.88.
# If the true curve saturates, a polynomial will overshoot.
# However, if the "training range" is the only thing that matters for the fit, and the "test" is in-distribution, a polynomial might fit the *training* data very well if the data doesn't extend to extreme saturation, or if the polynomial coefficients are constrained by the fit to mimic saturation in the range.
# But wait, the score is on a "hidden in-distribution test split".
# If the test split has similar strain ranges, a polynomial fit might be sufficient.
# The current model (Padé) is bounded, which is physically better.
# But if the BFGS from all-ones fails to find the good Padé parameters, the score is bad.
# A linear-in-parameters model guarantees the global optimum is found.
#
# Let's check the number of parameters.
# I need exactly 10 parameters.
# Model:
# y = p0 + p1*s + p2*s^2 + p3*s^3 + p4*t + p5*s*t + p6*s^2*t + p7*t^2 + p8*s*t^2 + p9*s^2*t^2
# This is a general 2nd order in temp and 3rd order in strain interaction?
# Let's count:
# 1. p0 (const)
# 2. p1 (s)
# 3. p2 (s^2)
# 4. p3 (s^3)
# 5. p4 (t)
# 6. p5 (s*t)
# 7. p6 (s^2*t)
# 8. p7 (t^2)
# 9. p8 (s*t^2)
# 10. p9 (s^2*t^2)
#
# This covers a lot of interaction.
# Is it flexible enough?
# The elastic region is linear in s. p1 captures E.
# The plastic region is non-linear. s^2 and s^3 capture curvature.
# Temp effects are captured by t, t^2, and interactions.
#
# This model is linear in parameters. The optimizer (BFGS) will essentially perform a linear regression (or close to it, as BFGS on a quadratic converges in N steps).
# This should yield a much better fit than a stuck non-linear optimizer.
#
# Let's implement this.

def _model_poly(strain, temp, params):
    s = strain
    t = temp
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    s2 = s * s
    s3 = s2 * s
    t2 = t * t
    
    # Terms
    term0 = p0
    term1 = p1 * s
    term2 = p2 * s2
    term3 = p3 * s3
    term4 = p4 * t
    term5 = p5 * s * t
    term6 = p6 * s2 * t
    term7 = p7 * t2
    term8 = p8 * s * t2
    term9 = p9 * s2 * t2
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

# Replace the model used in equation
def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    return _model_poly(strain, temp, params)

# EVOLVE-BLOCK-END
