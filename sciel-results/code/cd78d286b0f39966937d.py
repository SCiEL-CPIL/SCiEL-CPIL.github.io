import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Mixture of Experts (MoE) structure.
    
    The model uses a soft-gated convex combination of three linear expert functions
    to capture complex, non-uniform response surfaces. The gates are driven by
    temperature and pH using sigmoid activations to allow sharp transitions.
    Substrate follows Monod kinetics, and density provides inhibition.
    
    Parameters (10):
    p0: Monod half-saturation constant K_s
    p1: Density inhibition constant
    p2: Gate 1 weight (temperature coefficient)
    p3: Gate 1 bias
    p4: Gate 2 weight (pH coefficient)
    p5: Gate 2 bias
    p6: Expert 1 scale
    p7: Expert 2 scale
    p8: Expert 3 scale
    p9: Overall amplitude / bias
    """
    # Deadline guard
    t0 = time.time()
    
    # Extract parameters
    K_s = np.abs(params[0]) + 1e-6
    b_half = np.abs(params[1]) + 1e-6
    g1_w = params[2]
    g1_b = params[3]
    g2_w = params[4]
    g2_b = params[5]
    e1_scale = params[6]
    e2_scale = params[7]
    e3_scale = params[8]
    amplitude = params[9]
    
    # Monod kinetics for substrate
    monod = s / (K_s + s + 1e-12)
    
    # Density inhibition
    dens_term = 1.0 / (1.0 + (b / b_half) ** 2)
    
    # Compute gates using sigmoid functions
    # Gate 1: based on temperature
    gate1_arg = g1_w * temp + g1_b
    # Clip to prevent overflow in sigmoid
    gate1_arg = np.clip(gate1_arg, -50, 50)
    gate1 = 1.0 / (1.0 + np.exp(-gate1_arg))
    
    # Gate 2: based on pH
    gate2_arg = g2_w * pH + g2_b
    gate2_arg = np.clip(gate2_arg, -50, 50)
    gate2 = 1.0 / (1.0 + np.exp(-gate2_arg))
    
    # Gate 3: complement of gate1 (ensures convex combination)
    gate3 = 1.0 - gate1
    
    # Ensure gates are non-negative and sum to 1
    # This is already satisfied by construction, but let's be safe
    gates = np.column_stack([gate1, gate2, gate3])
    gate_sums = gates.sum(axis=1, keepdims=True)
    gates = gates / (gate_sums + 1e-12)
    
    # Expert functions: linear in inputs
    # Expert 1: favors low temperature
    expert1 = e1_scale * (temp / 40.0 + pH / 12.0)
    
    # Expert 2: favors mid-range pH
    expert2 = e2_scale * (temp / 40.0 - pH / 12.0 + 0.5)
    
    # Expert 3: favors high temperature
    expert3 = e3_scale * (1.0 - temp / 40.0 + pH / 12.0)
    
    # Combine experts with gates
    experts = np.column_stack([expert1, expert2, expert3])
    moe_output = (gates * experts).sum(axis=1)
    
    # Final result
    result = amplitude * monod * dens_term * moe_output + amplitude * 0.1
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard: if we're running long, just return
    if time.time() - t0 > 140:
        return result
    
    return result
