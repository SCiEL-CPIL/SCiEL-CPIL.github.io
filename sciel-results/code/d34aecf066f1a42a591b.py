"""E. coli growth rate model: Mixture of Experts with soft gating."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixture of Experts model for E. coli growth rate.
    
    Three linear experts with a soft gate based on temp, pH, and s.
    The gate uses softmax over three scores, allowing the model to
    smoothly transition between different regimes (e.g., low temp,
    optimal temp, high temp/death).
    
    mu = sum_k gate_k * (w0_k + w1_k*b + w2_k*s + w3_k*temp + w4_k*pH)
       + p9
    
    Parameters (10):
    p0: Gate weight for temp (expert 0)
    p1: Gate weight for pH (expert 0)
    p2: Gate bias (expert 0)
    p3: Gate weight for temp (expert 1)
    p4: Gate weight for pH (expert 1)
    p5: Gate bias (expert 1)
    p6: Gate weight for temp (expert 2)
    p7: Gate weight for pH (expert 2)
    p8: Gate bias (expert 2)
    p9: Global bias
    
    Wait, that's only 10 params for gates but no expert weights.
    I need 10 params total. Let me restructure.
    
    Option: 3 experts, each with 2 weights (for temp and pH), plus 1 global bias.
    That's 3*2 + 1 = 7 params. I have 3 more.
    
    Better: 3 experts, each with a linear function of (b, s, temp, pH).
    But 3 experts * 4 inputs = 12 weights + 1 bias = 13. Too many.
    
    Alternative: Use a simpler MoE where experts are fixed basis functions
    and only the gate weights are learned. But that limits expressiveness.
    
    Let me try: 2 experts instead of 3.
    Expert 1: linear in (b, s, temp, pH) -> 4 weights
    Expert 2: linear in (b, s, temp, pH) -> 4 weights
    Gate: sigmoid of (p8*temp + p9*pH) -> 2 params
    Total: 4 + 4 + 2 = 10 params.
    
    mu = gate * (w10 + w11*b + w12*s + w13*temp + w14*pH)
       + (1-gate) * (w20 + w21*b + w22*s + w23*temp + w24*pH)
    
    But that's 10 weights for experts + 2 for gate = 12. Still too many.
    
    Let me be more creative. Use shared structure.
    
    mu = base + gate * delta
    
    base = p0 + p1*b + p2*s + p3*temp + p4*pH
    delta = p5 + p6*b + p7*s + p8*temp + p9*pH
    gate = sigmoid(p10*temp + p11*pH)  # but p10 and p11 would be params 11 and 12
    
    That's 12 params. I only have 10.
    
    Let me use 3 experts with a shared linear structure but different biases.
    
    Actually, let me just use a different approach: a polynomial model with
    carefully chosen terms to capture the key features.
    
    mu = p0 + p1*b + p2*s + p3*temp + p4*pH 
        + p5*temp^2 + p6*pH^2 
        + p7*temp*pH + p8*b*s + p9*temp*s
    
    That's 10 params. This is a quadratic model with interactions.
    It can capture curvature in temp and pH, and interactions.
    
    Let me try this.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Quadratic model with key interactions
    pred = (p0
            + p1 * b
            + p2 * s
            + p3 * temp
            + p4 * pH
            + p5 * temp ** 2
            + p6 * pH ** 2
            + p7 * temp * pH
            + p8 * b * s
            + p9 * temp * s)
    
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
