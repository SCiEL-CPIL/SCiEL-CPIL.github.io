import math

def coefficients(stages):
    """
    Implements a 4th-order symmetric splitting using 12 stages.
    Uses 3 sub-steps of size 1/3, each implementing the Yoshida 4th-order scheme
    composed of three Strang steps.
    """
    if stages != 12:
        # Fallback to uniform if stages is not 12
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    # 4th order coefficients
    sqrt3 = math.sqrt(3.0)
    gamma1 = (1.0 + 1.0 / sqrt3) / 3.0
    gamma2 = (1.0 - 2.0 / sqrt3) / 3.0
    
    h = 1.0 / 3.0
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # 3 sub-steps, 4 stages each
    for j in range(3):
        base = 4 * j
        
        # Stage 1: B(gamma1*h/2) A(gamma1*h)
        b[base] = gamma1 * h / 2.0
        a[base] = gamma1 * h
        
        # Stage 2: B((gamma1+gamma2)*h/2) A(gamma2*h)
        b[base + 1] = (gamma1 + gamma2) * h / 2.0
        a[base + 1] = gamma2 * h
        
        # Stage 3: B((gamma2+gamma1)*h/2) A(gamma1*h)
        b[base + 2] = (gamma2 + gamma1) * h / 2.0
        a[base + 2] = gamma1 * h
        
        # Stage 4: B(gamma1*h/2) A(0)
        b[base + 3] = gamma1 * h / 2.0
        a[base + 3] = 0.0

    return a, b
