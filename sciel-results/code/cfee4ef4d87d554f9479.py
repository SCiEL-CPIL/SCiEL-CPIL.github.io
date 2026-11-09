import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
            s = strain
            t = temp
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
            
            # Elastic modulus decreases with temp
            E = p0 + p1 * t
            # Yield stress decreases with temp
            Sy = p2 + p3 * t
            # Hardening rate
            H = p4 + p5 * t
            # Saturation strain scale
            K = p6 + p7 * t
            
            # Ensure K is not too small to avoid division by zero or extreme gradients
            # np.maximum(K, 1e-6) is non-differentiable. 
            # Instead, use a smooth saturation: 1 / (1 + s/K) is rational.
            # Let's use the rational form for the hardening part only, which is safer.
            # Hardening = Sy * (s / (s + K))
            # This is 0 at s=0, approaches Sy as s->inf.
            # It has an inflection point? No, it's concave.
            # Stress = E*s + Sy * s/(s+K)
            # This is a common model.
            
            # Let's use this structure with more params.
            # Term 1: Linear elastic
            term1 = E * s
            
            # Term 2: Saturation hardening
            # s / (s + K) can be written as 1 - K/(s+K)
            # To avoid issues if K is negative or s+K is zero, we can assume K > 0?
            # BFGS doesn't know that.
            # Let's use np.tanh which is safe.
            
            # Let's try a mix of rational and tanh?
            # No, keep it simple and robust.
            
            # Model:
            # sigma = (p0 + p1*t) * s + (p2 + p3*t) * tanh((p4 + p5*t) * s) + (p6 + p7*t) * s**2 + p8*t + p9*s*t
            
            # This uses all 10 params.
            # The s**2 term adds curvature.
            
            return term1 + (p2 + p3 * t) * np.tanh((p4 + p5 * t) * s) + (p6 + p7 * t) * s**2 + p8 * t + p9 * s * t
