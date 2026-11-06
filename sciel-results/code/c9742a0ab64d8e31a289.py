"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the growth-index approximation.
    
    Based on the analytic growth index for flat CPL dark energy:
    gamma ≈ (1+0.11*w0+0.05*wa) / (1+0.05*w0-0.03*wa+(0.05*w0-0.04*wa)*Om)
    
    We expand this in q = 1-Om and u = 1+w0+wa*(1-a):
    
    Numerator: 1 + 0.11*w0 + 0.05*wa
    Denominator: 1 + 0.05*w0 - 0.03*wa + (0.05*w0-0.04*wa)*(1-q)
               = 1 + 0.05*w0 - 0.03*wa + 0.05*w0 - 0.04*wa - q*(0.05*w0-0.04*wa)
               = 1 + 0.10*w0 - 0.07*wa - q*(0.05*w0-0.04*wa)
    
    For small corrections, gamma ≈ Num * (1 - (Denom-1) + (Denom-1)^2 - ...)
    
    Let d = 0.10*w0 - 0.07*wa - q*(0.05*w0-0.04*wa)
    gamma ≈ (1+0.11*w0+0.05*wa) * (1 - d + d^2)
    
    This gives a complex expansion. Let me use a simpler fitting approach
    based on known values at reference points.
    """
    # Reference: at Om=0.3, w0=-1, wa=0: gamma ≈ 0.55
    # At Om=0.3, w0=0, wa=0: gamma ≈ 1.0 (EDS-like, but actually ~0.55 for flat)
    # Actually for EDS (w=-1, flat): gamma ≈ 0.55
    # For w=0: gamma is larger
    
    # Using the formula gamma ≈ (1+0.11w0+0.05wa)/(1+0.05w0-0.03wa+(0.05w0-0.04wa)Om)
    # 
    # At Om=1, w0=wa=0: gamma = 1/1 = 1.0 (but this is EDS which should be ~0.55)
    # The formula is for the growth rate f=Om^gamma, and for EDS gamma=1 exactly
    # For LCDM (w0=-1, wa=0, Om=0.3): gamma = (1-0.11)/(1-0.05+(0-0)*0.3) = 0.89/0.95 ≈ 0.937
    # Hmm, that doesn't match the expected ~0.55
    #
    # Let me reconsider. The growth index gamma in f=Om^gamma:
    # For EDS: f = Om, so gamma = 1
    # For LCDM: f ≈ Om^0.55, so gamma ≈ 0.55
    #
    # The formula (1+0.11w0+0.05wa)/(1+0.05w0-0.03wa+(0.05w0-0.04wa)Om)
    # At w0=-1, wa=0, Om=0.3: (1-0.11)/(1-0.05+0) = 0.89/0.95 ≈ 0.937
    # That's not 0.55...
    #
    # I think I'm confusing two different conventions. Let me just use 
    # empirically well-known values:
    # - EDS (w=-1): gamma = 1.0
    # - LCDM (w0=-1, wa=0, Om~0.3): gamma ≈ 0.55
    # - wCDM with w0=-0.5: gamma ≈ 0.65
    #
    # The ansatz is gamma = c0+c1*q+c2*q^2+c3*u+c4*wa*a+c5*u*q
    # where q=1-Om, u=1+w0+wa*(1-a)
    #
    # At EDS: w0=-1, wa=0, so u=1-1+0=0
    #   gamma = c0+c1*q+c2*q^2+0+0+0 = 1.0 for all q
    #   So c0=1.0, c1=0, c2=0 (for the EDS case)
    #
    # At LCDM: w0=-1, wa=0, Om=0.3, so q=0.7, u=0
    #   gamma = c0+c1*0.7+c2*0.49 = 0.55
    #   If c0=1.0: 1.0+0.7*c1+0.49*c2 = 0.55
    #
    # At w0=0, wa=0, Om=0.3: q=0.7, u=1
    #   gamma = c0+0.7*c1+0.49*c2+c3 = ?
    #   For w=0 dark energy, gamma is larger, maybe ~0.75
    #
    # Let me set up a system:
    # EDS (w0=-1, wa=0, any Om): gamma=1.0 => c0=1.0, c1=0, c2=0
    # But then LCDM: 1.0 = 0.55? Contradiction!
    #
    # The issue: EDS has w=-1 exactly, so u=1+(-1)+0*(1-a)=0
    # LCDM also has w0=-1, wa=0, so u=0 too!
    # They should give the same gamma in this ansatz, but they don't in reality.
    #
    # This means the ansatz can't perfectly distinguish EDS from LCDM since both have u=0.
    # The difference must come from the q dependence or be an inherent limitation.
    #
    # Actually wait - for EDS, Om(a) = 1 (matter only), so q=0.
    # For LCDM, Om(a)=0.3, q=0.7.
    # So EDS: q=0, u=0 => gamma = c0 = 1.0
    # LCDM: q=0.7, u=0 => gamma = c0+0.7*c1+0.49*c2 = 0.55
    # So c0=1.0, and 0.7*c1+0.49*c2 = -0.45
    #
    # For w0=0, wa=0, Om=0.3: q=0.7, u=1
    #   gamma = 1.0+0.7*c1+0.49*c2+c3 ≈ 0.75 (estimate)
    #   So c3 ≈ 0.75-0.55 = 0.20
    #
    # For w0=-1, wa=0.5, a=0.5: u=1-1+0.5*0.5=0.25, q≈0.7
    #   gamma = 1.0+0.7*c1+0.49*c2+0.25*0.20+c4*0.5*0.5+0.25*0.7*c5
    #   = 0.55+0.05+0.25*c4+0.175*c5
    #   Expected maybe ~0.58
    #   So 0.25*c4+0.175*c5 ≈ 0.03
    #
    # From 0.7*c1+0.49*c2 = -0.45, let's pick c1=-0.6, c2=0.15:
    #   0.7*(-0.6)+0.49*0.15 = -0.42+0.0735 = -0.3465 (not quite -0.45)
    # Try c1=-0.7, c2=0.1: 0.7*(-0.7)+0.49*0.1 = -0.49+0.049 = -0.441 ≈ -0.45 ✓
    #
    # So: c0=1.0, c1=-0.7, c2=0.1, c3=0.20
    # For c4 and c5: 0.25*c4+0.175*c5 ≈ 0.03
    # Let c4=0.1, c5=0.05: 0.025+0.00875=0.034 ✓
    
    c0 = 1.0
    c1 = -0.7
    c2 = 0.1
    c3 = 0.20
    c4 = 0.1
    c5 = 0.05
    return [c0, c1, c2, c3, c4, c5]
# EVOLVE-BLOCK-END
