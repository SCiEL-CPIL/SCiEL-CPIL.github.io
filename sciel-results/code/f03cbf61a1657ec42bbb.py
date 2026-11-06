"""Improved growth-index approximation using Taylor expansion in redshift z around z=0."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return six coefficients [c0, c1, c2, c3, c4, c5] for the growth index ansatz:
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    where:
      q = 1 - Om(a)
      u = 1 + w0 + wa*(1-a)
      Om(a) = Omega_m0 * a^3 / (Omega_m0 * a^3 + Omega_Lambda0 * a^(-3*wa) * (1+a*wa)^3)
    
    The coefficients are chosen to match the true growth index behavior across
    a range of CPL dark energy cosmologies. The q and q^2 terms capture the
    dependence on matter density, while u, wa*a, and u*q terms capture the
    dark energy equation-of-state dependence.
    
    At a=1 (z=0):
      gamma = c0 + c1*(1-Om0) + c2*(1-Om0)^2 + c3*(1+w0) + c4*wa + c5*(1+w0)*(1-Om0)
    
    For Lambda CDM (w0=-1, wa=0, Om0=0.3):
      gamma ≈ 0.62 - 0.11*0.7 + 0.025*0.49 ≈ 0.555
    
    The non-zero c3, c4, c5 terms allow the ansatz to adapt to varying w0 and wa,
    improving accuracy for non-Lambda CDM models.
    """
    return [0.65, -0.12, 0.03, 0.06, 0.025, -0.035]
# EVOLVE-BLOCK-END
