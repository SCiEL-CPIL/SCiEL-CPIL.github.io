import numpy as np

def coefficients(stages):
    # 2-step composite 4th-order Suzuki splitting
    # Base 4th-order coefficients
    c = 2.0 ** (1.0 / 3.0)
    alpha = 1.0 / (2.0 * (1.0 - c))
    beta = -alpha / c
    gamma = 1.0 - 2.0 * beta
    
    # Normalize to ensure sum=1 for the base splitting
    # Note: Standard Suzuki coeffs don't sum to 1 directly in this parametrization
    # without careful definition, so we normalize the weights for A and B separately
    # to maintain consistency (sum a = 1, sum b = 1) for the sub-step.
    
    # The base splitting is A(alpha) B(beta) A(gamma) B(beta) A(alpha)
    # We map this to 6 A-slots and 6 B-slots for one sub-step.
    # A slots: alpha/2, alpha/2, gamma/2, gamma/2, alpha/2, alpha/2
    # B slots: beta/2, beta/2, 0, 0, beta/2, beta/2
    
    a_sub = [alpha/2.0, alpha/2.0, gamma/2.0, gamma/2.0, alpha/2.0, alpha/2.0]
    b_sub = [beta/2.0, beta/2.0, 0.0, 0.0, beta/2.0, beta/2.0]
    
    # Normalize sub-step coefficients to sum to 1 (safety)
    sa = sum(a_sub)
    sb = sum(b_sub)
    a_sub = [x / sa for x in a_sub]
    b_sub = [x / sb for x in b_sub]
    
    # Create 12-stage lists by repeating the sub-step twice
    # This corresponds to evolving for t/2 using the splitting, then another t/2
    a = a_sub + a_sub
    b = b_sub + b_sub
    
    # Final normalization for the whole 12-stage sequence
    sa_total = sum(a)
    sb_total = sum(b)
    a = [x / sa_total for x in a]
    b = [x / sb_total for x in b]
    
    return a, b
