import numpy as np

def coefficients(stages):
    # 4th-order symmetric product formula using Suzuki coefficients
    # The scheme is: exp(-i c1 t B) exp(-i d1 t A) exp(-i c2 t B) exp(-i d2 t A) exp(-i c3 t B) exp(-i d3 t A) exp(-i c4 t B)
    # We map this to the 12-stage format B1 A1 ... B12 A12
    # by placing the non-zero coefficients in the first slots and zeroing the rest.
    # This preserves the operator sequence order.
    
    alpha = 2.0**(1.0/3.0)
    
    # Suzuki 4th-order coefficients
    c1 = 1.0 / (2.0 * (2.0 - alpha))
    c2 = 0.5 - c1
    c3 = c2
    c4 = c1
    
    d1 = alpha / (2.0 - alpha)
    d2 = 1.0 - 2.0 * d1
    d3 = d1
    
    # Initialize lists with zeros
    a = [0.0] * 12
    b = [0.0] * 12
    
    # Assign coefficients
    # Sequence: B(c1) A(d1) B(c2) A(d2) B(c3) A(d3) B(c4)
    b[0] = c1
    a[0] = d1
    b[1] = c2
    a[1] = d2
    b[2] = c3
    a[2] = d3
    b[3] = c4
    # a[3] remains 0, which is correct as the Suzuki scheme ends with B
    
    # Verify sums (should be 1)
    # sum(b) = c1+c2+c3+c4 = 2*c1 + 2*c2 = 2*c1 + 2*(0.5-c1) = 1
    # sum(a) = d1+d2+d3 = 2*d1 + (1-2*d1) = 1
    
    return a, b
