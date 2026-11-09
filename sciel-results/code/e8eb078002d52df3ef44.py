"""
Symbolic regression for stress-strain of an aluminium rod.
Uses a multiplicative temperature-modulated elastic-plastic model with
smooth hyperbolic saturation for hardening.
The 10 parameters are fitted by the evaluator using BFGS.
"""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model structure:
    - Temperature-dependent elastic modulus E(T) and yield stress sigma_y(T)
      via low-order polynomials in centered temperature
    - Smooth hyperbolic sine saturation models plastic hardening after yield
    - Multiplicative coupling captures physical interplay between T and plasticity
    
    Parameters:
    p0: base stress offset
    p1: elastic modulus at reference temperature
    p2: temperature coefficient for elastic modulus (linear)
    p3: temperature coefficient for elastic modulus (quadratic)
    p4: yield strain location (kink)
    p5: yield stress magnitude at reference temperature
    p6: temperature coefficient for yield stress (linear)
    p7: temperature coefficient for yield stress (quadratic)
    p8: hardening saturation amplitude
    p9: hardening rate (controls how quickly saturation is reached)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Center temperature for better conditioning
    tc = temp - 0.5
    
    # Temperature-dependent elastic modulus: E(T) = p1 + p2*tc + p3*tc^2
    E_T = p1 + p2 * tc + p3 * tc * tc
    
    # Temperature-dependent yield stress: sigma_y(T) = p5 + p6*tc + p7*tc^2
    sigma_y_T = p5 + p6 * tc + p7 * tc * tc
    
    # Elastic stress contribution (linear in strain)
    elastic_stress = E_T * strain
    
    # Plastic strain: amount of strain beyond yield point
    # Use a smooth approximation to max(strain - p4, 0) via softplus-like function
    # plastic_strain = (1/p9) * log(1 + exp(p9 * (strain - p4)))
    # But for numerical stability, use np.logaddexp
    x = p9 * (strain - p4)
    # Softplus: log(1+exp(x)) = logaddexp(0, x)
    softplus = np.logaddexp(0.0, x)
    plastic_strain = softplus / p9 if abs(p9) > 1e-12 else np.maximum(strain - p4, 0.0)
    
    # Hardening: hyperbolic sine saturation
    # The hardening stress grows with plastic strain but saturates
    # Use: sigma_h = p8 * tanh(k * plastic_strain) for bounded hardening
    # Or: sigma_h = p8 * (1 - exp(-k * plastic_strain)) for exponential saturation
    # Let's use a combination: sinh-based for initial growth, then saturation
    # sigma_h = p8 * sinh(a * plastic_strain) / sinh(a) for small plastic_strain
    # Actually, let's use: sigma_h = p8 * (1 - exp(-p9 * plastic_strain)) * (p9 * plastic_strain + 1) / (p9 * plastic_strain + 1)
    # Simpler: use tanh for smooth saturation
    # sigma_h = p8 * tanh(p9 * plastic_strain)
    # But we already used p9 for the softplus rate. Let's restructure.
    
    # Actually, let me reconsider the parameter allocation.
    # Let's use:
    # p0: offset
    # p1: E at T=0.5
    # p2: dE/dT
    # p3: d2E/dT2
    # p4: yield strain
    # p5: sigma_y at T=0.5
    # p6: dsigma_y/dT
    # p7: d2sigma_y/dT2
    # p8: hardening amplitude
    # p9: hardening rate
    
    # Recompute with this understanding
    # The softplus rate and hardening rate are both p9, which may be fine
    # or we can use a fixed rate for softplus and p9 for hardening
    
    # Let's use a fixed smoothing parameter for the softplus to free up p9
    # Actually, let's keep p9 for hardening rate and use a moderate fixed value for softplus
    
    # Redo: use fixed smoothing for yield transition
    # plastic_strain = max(strain - p4, 0) smoothed with fixed rate
    # This frees p9 for hardening saturation rate
    
    # Fixed smoothing rate for yield transition
    k_smooth = 50.0
    x_smooth = k_smooth * (strain - p4)
    softplus_smooth = np.logaddexp(0.0, x_smooth)
    plastic_strain = softplus_smooth / k_smooth
    
    # Hardening stress with tanh saturation
    # sigma_h = p8 * tanh(p9 * plastic_strain)
    # This gives:
    # - Linear growth for small plastic_strain: p8 * p9 * plastic_strain
    # - Saturates to p8 for large plastic_strain
    
    hardening_stress = p8 * np.tanh(p9 * plastic_strain)
    
    # Total stress: elastic + yield offset + hardening
    # The yield stress acts as a threshold, so:
    # stress = p0 + E_T * strain + sigma_y_T * (plastic_strain > 0 indicator) + hardening
    # But we want smooth transition. Let's use:
    # stress = p0 + E_T * strain + sigma_y_T * tanh(p9 * plastic_strain) / tanh(p9 * plastic_strain) * ...
    
    # Simpler physical model:
    # stress = E_T * min(strain, p4) + sigma_y_T * (1 - exp(-p9 * max(strain - p4, 0)))
    # But this is not smooth at p4.
    
    # Let's use a clean formulation:
    # stress = p0 + E_T * strain + (sigma_y_T + hardening_stress) * (plastic_strain / max(plastic_strain, epsilon))
    # This is getting complicated. Let me simplify.
    
    # Clean physical model:
    # 1. Elastic response: E_T * strain
    # 2. Once strain exceeds yield, add yield stress + hardening
    #    Use smooth transition via the softplus-derived plastic_strain
    
    # The yield stress should be added gradually as plastic strain develops
    # Use: yield_contribution = sigma_y_T * (1 - exp(-p9 * plastic_strain))
    # This smoothly turns on the yield stress
    
    yield_contribution = sigma_y_T * (1.0 - np.exp(-p9 * plastic_strain))
    
    # Total stress
    stress = p0 + E_T * strain + yield_contribution + hardening_stress
    
    # Ensure finite values
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
