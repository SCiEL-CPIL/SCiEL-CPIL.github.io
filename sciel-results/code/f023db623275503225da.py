import time

# EVOLVE-BLOCK-START
def coefficients(stages):
    """4th-order symmetric Suzuki-Trotter product formula for 12 stages."""
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # 3 substeps of 4th order Suzuki
    # Each substep uses t/3
    # B coeffs: c1/6, (c1+c2)/6, (c1+c2)/6, c1/3, (c1+c2)/6, (c1+c2)/6, c1/3, (c1+c2)/6, (c1+c2)/6, c1/6
    # A coeffs: c1/3, c2/3, c1/3, c1/3, c2/3, c1/3, c1/3, c2/3, c1/3
    
    b_vals = [
        c1/6.0,
        (c1+c2)/6.0,
        (c1+c2)/6.0,
        c1/3.0,
        (c1+c2)/6.0,
        (c1+c2)/6.0,
        c1/3.0,
        (c1+c2)/6.0,
        (c1+c2)/6.0,
        c1/6.0
    ]
    
    a_vals = [
        c1/3.0,
        c2/3.0,
        c1/3.0,
        c1/3.0,
        c2/3.0,
        c1/3.0,
        c1/3.0,
        c2/3.0,
        c1/3.0
    ]
    
    for i in range(min(10, stages)):
        b[i] = b_vals[i]
    for i in range(min(9, stages)):
        a[i] = a_vals[i]
        
    return a, b
# EVOLVE-BLOCK-END
