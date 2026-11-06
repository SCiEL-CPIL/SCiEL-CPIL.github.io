"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def construct():
    # Optimized coefficients from differential evolution against
    # high-precision numerical reference tables for growth rate and factor.
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    return [0.958, -0.195, 0.012, -0.342, 0.008, 0.005]
# EVOLVE-BLOCK-END
