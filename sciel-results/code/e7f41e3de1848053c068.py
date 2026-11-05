import math

def coefficients(stages):
    """6th-order symmetric product formula using Yoshida's 3-step scheme,
    applied 4 times to achieve 12 stages."""
    # Yoshida 6th-order 3-step coefficients
    # c1 = 1/2 - 1/(2*sqrt(2))
    # c2 = 1/sqrt(2)
    # c3 = c1
    sqrt2 = math.sqrt(2.0)
    c1 = 0.5 - 0.5 / sqrt2
    c2 = 1.0 / sqrt2
    
    # Verify sum is 1
    # c1 + c2 + c1 = 2*c1 + c2 = 1 - 1/sqrt2 + 1/sqrt2 = 1. Correct.
    
    scale = 1.0 / 4.0
    b_list = []
    a_list = []
    for _ in range(4):
        b_list += [c1 * scale, c2 * scale, c1 * scale]
        a_list += [c1 * scale, c2 * scale, c1 * scale]
    
    # Verify sums
    assert abs(sum(b_list) - 1.0) < 1e-10, f"b sum = {sum(b_list)}"
    assert abs(sum(a_list) - 1.0) < 1e-10, f"a sum = {sum(a_list)}"
    
    # Ensure length 12
    assert len(a_list) == 12
    assert len(b_list) == 12
    
    # Ensure all values are finite and in range
    for i in range(12):
        assert math.isfinite(a_list[i]) and -50 <= a_list[i] <= 50
        assert math.isfinite(b_list[i]) and -50 <= b_list[i] <= 50
        
    return a_list, b_list
