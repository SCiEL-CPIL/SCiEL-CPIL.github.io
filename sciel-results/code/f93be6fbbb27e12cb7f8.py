# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Mixture of Experts (MoE) architecture.
    
    This model replaces the global additive structure with a soft-gated convex
    combination of three distinct expert functions. The gating mechanism allows
    the model to capture complex, non-uniform response surfaces and sharp
    transitions (e.g., thermal death or pH extremes) by dynamically weighting
    experts based on input conditions.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate concentration (range ~[1, 40])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[1, 14])
    
    Parameters (10):
    p0: Softness parameter for gating (inverse temperature, used as exp(p0))
    p1: Expert 1 (Baseline) - Weight for Monod term
    p2: Expert 1 (Baseline) - Bias
    p3: Expert 2 (Thermal) - Weight for Gaussian temp term
    p4: Expert 2 (Thermal) - Gaussian center (temp)
    p5: Expert 3 (Acidic) - Weight for Gaussian pH term
    p6: Expert 3 (Acidic) - Gaussian center (pH)
    p7: Gate 1 - Logit weight for temp
    p8: Gate 2 - Logit weight for pH
    p9: Gate 3 - Logit weight for substrate (b/s interaction)
    
    Structure:
    1. Normalize inputs.
    2. Compute gate logits from normalized temp, pH, and substrate.
    3. Apply softmax to get expert weights.
    4. Compute expert outputs:
       - Expert 1: Monod(s) * p1 + p2
       - Expert 2: Gaussian(temp, p4) * p3
       - Expert 3: Gaussian(pH, p6) * p5
    5. Output: weighted sum of experts.
    """
    p = params

    # Normalize inputs to [0, 1] range for stability and gating
    # b: [1, 5]
    b_norm = (b - 1.0) / 4.0
    # s: [1, 40]
    s_norm = (s - 1.0) / 39.0
    # temp: [1, 40]
    temp_norm = (temp - 1.0) / 39.0
    # pH: [1, 14]
    ph_norm = (pH - 1.0) / 13.0

    # 1. Gating Mechanism
    # Softness parameter controls how sharp the gating is
    # Higher exp(p0) -> sharper gating (more like hard assignment)
    # Lower exp(p0) -> softer gating (more uniform)
    softness = np.exp(p[0])
    
    # Logits for gating
    # We use normalized inputs directly in the linear combination
    # This allows the model to learn which input drives which expert
    logit_1 = p[7] * temp_norm + p[9] * s_norm
    logit_2 = p[8] * ph_norm
    logit_3 = 0.0  # Baseline expert has a constant logit (or could be linked to b)
    # Let's include b in the baseline gate to allow density-dependent regime switching
    logit_3 = p[9] * b_norm - 1.0 # Fixed offset to balance
    
    # Shift logits to ensure numerical stability in softmax
    max_logit = np.maximum(np.maximum(logit_1, logit_2), logit_3)
    exp_l1 = np.exp(softness * (logit_1 - max_logit))
    exp_l2 = np.exp(softness * (logit_2 - max_logit))
    exp_l3 = np.exp(softness * (logit_3 - max_logit))
    
    sum_exp = exp_l1 + exp_l2 + exp_l3
    # Avoid division by zero
    sum_exp = np.where(sum_exp == 0, 1e-6, sum_exp)
    
    w1 = exp_l1 / sum_exp
    w2 = exp_l2 / sum_exp
    w3 = exp_l3 / sum_exp

    # 2. Expert Functions
    
    # Expert 1: Substrate-limited growth (Monod)
    # Use a fixed Ks or derive from params? We have limited params.
    # Let's use a fixed Ks=1.0 for the base expert, scaled by p1.
    # Monod term: s / (Ks + s). With s_norm, Ks_norm ~ 1/39?
    # Better to use raw s for Monod to keep biological meaning, but s varies [1, 40].
    # Ks = 1.0 is reasonable for E. coli.
    monod = s / (1.0 + s)
    expert_1 = p[1] * monod + p[2]

    # Expert 2: Thermal Optimum (Gaussian)
    # Centered at p4 (in normalized space [0,1])
    # Width fixed to capture the bell shape within the domain
    width_temp = 0.15
    temp_centered = temp_norm - p[4]
    gaussian_temp = np.exp(-(temp_centered ** 2) / (2.0 * width_temp ** 2))
    expert_2 = p[3] * gaussian_temp

    # Expert 3: pH Optimum (Gaussian)
    # Centered at p6 (in normalized space [0,1])
    width_ph = 0.15
    ph_centered = ph_norm - p[6]
    gaussian_ph = np.exp(-(ph_centered ** 2) / (2.0 * width_ph ** 2))
    expert_3 = p[5] * gaussian_ph

    # 3. Combine Experts
    result = w1 * expert_1 + w2 * expert_2 + w3 * expert_3

    # Ensure the output is finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
