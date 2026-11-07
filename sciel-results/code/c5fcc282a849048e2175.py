import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(b)
    
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_norm = np.clip((b - 1.0) / (4.998 - 1.0), 0, 1)
    s_norm = np.clip((s - 1.0) / (4.999 - 1.0), 0, 1)
    temp_norm = np.clip((temp - 1.002) / (39.98 - 1.002), 0, 1)
    ph_norm = np.clip((pH - 2.0) / (12.0 - 2.0), 0, 1)
    
    # Structural change: Combine additive polynomial flexibility with multiplicative gating.
    # The current program (score 1.18) uses a pure additive polynomial.
    # Inspiration 1 (score 41.9) uses a pure multiplicative mechanism (Monod * Gaussian * Gaussian).
    # Inspiration 2 (score 0.88) uses a hybrid but with raw inputs.
    
    # Hypothesis: The true relationship likely has a multiplicative core (growth requires
    # suitable T and pH, and substrate saturation) but also additive nuances or offsets
    # that a pure multiplicative model struggles to fit if the baseline isn't zero.
    # Or, the additive polynomial is too flexible in the wrong places (overfitting noise
    # in low-signal regions) while missing the structural constraints of the biology.
    
    # New Structure:
    # y = (Additive Polynomial Base) * (Multiplicative Gating Factors)
    # OR
    # y = p[0] + p[1] * (Monod(s) * Gaussian(T) * Gaussian(PH)) * b_factor
    
    # Let's try a hybrid:
    # 1. Substrate: Monod-like saturation (smooth, positive). 
    #    p[0] controls half-saturation.
    # 2. Temperature: Gaussian bell curve.
    #    p[1] = center (in normalized space 0-1), p[2] = width.
    # 3. pH: Gaussian bell curve.
    #    p[3] = center, p[4] = width.
    # 4. Biomass: Linear scaling factor.
    #    p[5] = coefficient for b_norm.
    # 5. Offset: Additive constant.
    #    p[6] = offset.
    # 6. Residual Additive Terms: To capture non-multiplicative effects.
    #    p[7] * temp_norm
    #    p[8] * ph_norm
    #    p[9] * b_norm
    
    # However, pure multiplicative models (Inspiration 1) scored very poorly (41.9).
    # The current additive model (1.18) is decent.
    # Inspiration 2 (0.88) was worse than 1.18? No, 0.8755 is lower than 1.18, so it's BETTER?
    # Wait. "Scoring (lower is better): normalized MSE".
    # Current score: 1.1804.
    # Inspiration 1 score: 41.9375 (Very Bad).
    # Inspiration 2 score: 0.8755 (Better than 1.18).
    # Inspiration 3 score: 0.0000 (Invalid/Timeout?).
    
    # So Inspiration 2 (Hybrid: Base multiplicative + additive interactions) is the best performer so far.
    # It used:
    # base = p[0] * monod * temp_factor * ph_factor
    # interaction_b = p[6] * b * monod
    # interaction_tp = p[7] * temp_factor * ph_factor
    # offset = p[8]
    # result = base + interaction_b + interaction_tp + offset
    # It had 9 parameters effectively used? p[0] to p[9] is 10 params.
    # p[0] base scale
    # p[1] ks
    # p[2] t_opt
    # p[3] t_width
    # p[4] ph_opt
    # p[5] ph_width
    # p[6] interaction b
    # p[7] interaction tp
    # p[8] offset
    # p[9] hill coefficient for monod
    
    # The current program is a pure additive polynomial.
    # The directive says: "Make a STRUCTURAL change... do not micro-tune constants."
    # The current program is stagnating? "population best has not improved".
    # My parent is 1.18. Inspiration 2 is 0.88.
    # I should try to improve upon the structural idea of Inspiration 2 or combine it with the robustness of the current normalization.
    
    # Let's refine Inspiration 2's structure.
    # Inspiration 2 used raw inputs for b, s, temp, pH in some terms and normalized in others?
    # No, it used:
    # ks = exp(p[1])
    # monod = (s ** hill) / (ks ** hill + s ** hill)  # Uses raw s
    # temp_factor = exp(-0.5 * ((temp - t_opt) / t_width) ** 2) # Uses raw temp
    # ph_factor = exp(-0.5 * ((pH - ph_opt) / ph_width) ** 2) # Uses raw pH
    # interaction_b = p[6] * b * monod # Uses raw b
    
    # The current program uses normalized inputs [0,1]. This is generally better for optimization stability.
    # Let's take the *structure* of Inspiration 2 (Multiplicative Core + Additive Interactions) but apply it to *normalized* inputs.
    
    # Structure:
    # 1. Substrate Saturation (Monod): s_norm / (k_s + s_norm)
    #    k_s = exp(p[0])
    # 2. Temperature Gate: Gaussian on temp_norm
    #    t_center = p[1]
    #    t_width = exp(p[2])
    #    t_gate = exp(-0.5 * ((temp_norm - t_center) / t_width)**2)
    # 3. pH Gate: Gaussian on ph_norm
    #    ph_center = p[3]
    #    ph_width = exp(p[4])
    #    ph_gate = exp(-0.5 * ((ph_norm - ph_center) / ph_width)**2)
    # 4. Biomass Effect: Linear on b_norm
    #    b_linear = p[5] * b_norm
    # 5. Combine:
    #    core = p[6] * monod * t_gate * ph_gate
    #    # Additive interactions to capture deviations from pure multiplicative
    #    add_b = p[7] * b_norm * monod
    #    add_t = p[8] * t_gate
    #    add_ph = p[9] * ph_gate
    #    result = core + add_b + add_t + add_ph
    
    # This uses all 10 params.
    # p[0]: k_s
    # p[1]: t_center
    # p[2]: t_width (log scale)
    # p[3]: ph_center
    # p[4]: ph_width (log scale)
    # p[5]: b_linear_coeff
    # p[6]: core_scale
    # p[7]: b_interact_coeff
    # p[8]: t_add_coeff
    # p[9]: ph_add_coeff
    
    # Wait, I didn't use p[5] in the draft above. Let's integrate p[5] into the core or as a separate term.
    # Maybe p[5] is a linear term for b_norm added to the result?
    
    # Revised Structure:
    # monod = s_norm / (exp(p[0]) + s_norm)
    # t_gate = exp(-0.5 * ((temp_norm - p[1]) / exp(p[2]))**2)
    # ph_gate = exp(-0.5 * ((ph_norm - p[3]) / exp(p[4]))**2)
    
    # core = p[5] * monod * t_gate * ph_gate
    
    # Additive terms to allow the model to fit non-zero baselines or linear trends that multiplicative models miss:
    # add_b = p[6] * b_norm
    # add_t = p[7] * temp_norm
    # add_ph = p[8] * ph_norm
    # add_const = p[9]
    
    # result = core + add_b + add_t + add_ph + add_const
    
    # This is a very flexible model. It can approximate a pure additive polynomial if the gates are wide (t_width, ph_width large) and monod is linear (k_s large).
    # It can approximate a pure multiplicative model if the additive terms are small.
    # This "Structural Change" moves from a pure additive polynomial (current) to a "Gated Additive" model, which is a different family (similar to Inspiration 2 but with normalized inputs and a slightly different interaction set).
    
    # Let's implement this.
    
    k_s = np.exp(p[0])
    monod = s_norm / (k_s + s_norm)
    
    t_center = p[1]
    t_width = np.exp(p[2])
    t_gate = np.exp(-0.5 * ((temp_norm - t_center) / t_width)**2)
    
    ph_center = p[3]
    ph_width = np.exp(p[4])
    ph_gate = np.exp(-0.5 * ((ph_norm - ph_center) / ph_width)**2)
    
    core = p[5] * monod * t_gate * ph_gate
    
    add_b = p[6] * b_norm
    add_t = p[7] * temp_norm
    add_ph = p[8] * ph_norm
    add_const = p[9]
    
    result = core + add_b + add_t + add_ph + add_const
    
    return result
