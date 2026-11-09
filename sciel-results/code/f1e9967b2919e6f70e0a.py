# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a Mixture of Experts model where temperature and pH determine
    a soft-gated convex combination of three linear expert functions.
    Each expert has different sensitivities to the inputs, allowing
    the model to capture complex, non-uniform response surfaces.
    
    params layout (10 parameters):
    - params[0]: global offset/bias
    - params[1:4]: expert 0 coefficients (b, s, temp, pH) - actually 4 coeffs per expert
    Wait, 3 experts * 4 inputs = 12, too many. Let's use 3 experts with 2 params each?
    
    Let me rethink. We have 10 params.
    Expert 0: base linear model
    Expert 1: high temp specialist
    Expert 2: low pH specialist
    
    Actually, let's use a simpler MoE:
    - 3 experts, each is a linear function of (b, s, temp, pH)
    - But that's 3*4 = 12 params just for experts, plus gates
    
    Let's use:
    - params[0]: shared bias
    - params[1:5]: shared linear coefficients for b, s, temp, pH
    - params[5:8]: gating parameters (temperature center, temperature width, pH sensitivity)
    - params[8:10]: expert-specific adjustments
    
    Actually, let me design this more carefully with exactly 10 params:
    
    Expert 0 (baseline): w0_b * b + w0_s * s + w0_t * temp + w0_ph * pH + c0
    Expert 1 (heat-tolerant): different weights
    Expert 2 (acid-tolerant): different weights
    
    Gates based on temp and pH.
    
    Let's use:
    params[0]: base coefficient for b
    params[1]: base coefficient for s  
    params[2]: base coefficient for temp
    params[3]: base coefficient for pH
    params[4]: bias term
    params[5]: temp gate center
    params[6]: temp gate width
    params[7]: pH gate center
    params[8]: expert 1 extra temp coefficient
    params[9]: expert 2 extra pH coefficient
    
    Hmm, this is getting complicated. Let me try a cleaner approach.
    
    MoE with 3 experts:
    - Expert weights are determined by softmax of linear functions of temp and pH
    - Each expert is a linear model
    
    params[0:4]: Expert 0: [b_coef, s_coef, temp_coef, ph_coef]
    params[4:8]: Expert 1: [b_coef, s_coef, temp_coef, ph_coef] 
    params[8]: bias for all experts
    params[9]: temperature gate scale
    
    But we need pH in the gate too. Let's use:
    params[0:4]: Expert 0 coefficients for [b, s, temp, pH]
    params[4:8]: Expert 1 coefficients for [b, s, temp, pH]
    params[8]: gate temperature center
    params[9]: gate temperature width
    
    And use a fixed pH effect in the gate, or blend.
    
    Actually, let's simplify: use 2 experts with 4 params each = 8, plus 2 gate params.
    """
    # Unpack parameters
    p = params
    
    # Expert 0: baseline linear model
    w0_b, w0_s, w0_t, w0_ph = p[0], p[1], p[2], p[3]
    # Expert 1: alternative linear model
    w1_b, w1_s, w1_t, w1_ph = p[4], p[5], p[6], p[7]
    
    # Gate parameters
    gate_temp_center = p[8]
    gate_temp_width = np.maximum(np.abs(p[9]), 1e-6)  # Avoid division by zero
    
    # Compute expert outputs
    expert0 = w0_b * b + w0_s * s + w0_t * temp + w0_ph * pH
    expert1 = w1_b * b + w1_s * s + w1_t * temp + w1_ph * pH
    
    # Compute gate weights based on temperature
    # Use a smooth sigmoid-like gate
    temp_diff = (temp - gate_temp_center) / gate_temp_width
    gate_weight = 1.0 / (1.0 + np.exp(-temp_diff))
    
    # Blend the two experts
    result = (1.0 - gate_weight) * expert0 + gate_weight * expert1
    
    # Ensure non-negative growth rate (biological constraint)
    # Actually, let's not enforce this hard since the target can be negative
    # based on the range [-0.00277, 1.553]
    
    return result
# EVOLVE-BLOCK-END
