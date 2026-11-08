import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
       """
       Acceleration of a nonlinear oscillator.
       Structure:
       - Rational Stiffness: (p0*x + p1*x^3) / (1 + p2*x^2)
       - Rational Damping: (p3*v + p4*v^3) / (1 + p5*v^2)
       - Van der Pol Term: p6 * x^2 * v
       - Cross Term: p7 * x * v
       - Velocity-Stiffness Coupling: p8 * x * v^2
       - Offset: p9
       """
       p = params
       
       # Precompute powers
       x2 = x * x
       x3 = x2 * x
       v2 = v * v
       v3 = v2 * v
       
       # Rational Stiffness
       stiff_num = p[0] * x + p[1] * x3
       stiff_den = 1.0 + p[2] * x2
       # Ensure denominator is not zero (though with x in [-0.6, 0.6] and p starting at 1, it's safe)
       # We can add a tiny epsilon for safety if p[2] becomes large negative, but BFGS usually stays in reasonable bounds.
       term_stiff = stiff_num / stiff_den
       
       # Rational Damping
       damp_num = p[3] * v + p[4] * v3
       damp_den = 1.0 + p[5] * v2
       term_damp = damp_num / damp_den
       
       # Nonlinear Damping / Coupling
       term_vdp = p[6] * x2 * v
       term_cross = p[7] * x * v
       term_vel_stiff = p[8] * x * v2
       
       # Offset
       offset = p[9]
       
       return term_stiff + term_damp + term_vdp + term_cross + term_vel_stiff + offset
