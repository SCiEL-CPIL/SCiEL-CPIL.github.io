"""Improved growth index approximation using a compact ansatz."""
# EVOLVE-BLOCK-START
def construct():
    # Coefficients for gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # c0: Base growth index (approx 0.55 for Lambda CDM)
    # c1: Correction for matter density evolution (q)
    # c3: Correction for dark energy equation of state (u)
    return [0.54, -0.12, 0.05, 0.08, 0.0, 0.0]
# EVOLVE-BLOCK-END
