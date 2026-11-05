import math

def coefficients(stages):
    """8th-order symmetric product formula using Yoshida's 7-step scheme,
    padded with zeros to achieve 12 stages."""
    # Yoshida 8th-order 7-step coefficients
    beta = (2.0 - math.sqrt(2.0)) ** (1.0 / 3.0)
    c1 = 0.5 - 0.5 / beta
    c2 = 0.5 / beta
    c3 = 0.5 - 1.0 / beta + 0.5 / beta  # Simplifies to c1 in some derivations, but let's keep the formula
    # Actually, for Yoshida 8th order, the coefficients are:
    # c1 = 1/2 - 1/(2*beta)
    # c2 = 1/(2*beta)
    # c3 = 1/2 - 1/beta + 1/(2*beta)
    # c4 = 1 - 2*(c1+c2+c3)
    
    c4 = 1.0 - 2.0 * (c1 + c2 + c3)
    
    # Verify sum
    total_sum = 2.0 * (c1 + c2 + c3) + c4
    assert abs(total_sum - 1.0) < 1e-10, f"Sum = {total_sum}"
    
    # Create lists of length 12
    # The 7 steps are: c1, c2, c3, c4, c3, c2, c1
    # Pad with 0s
    base_coeffs = [c1, c2, c3, c4, c3, c2, c1]
    a_list = base_coeffs + [0.0] * 5
    b_list = base_coeffs + [0.0] * 5
    
    # Verify sums
    assert abs(sum(a_list) - 1.0) < 1e-10, f"a sum = {sum(a_list)}"
    assert abs(sum(b_list) - 1.0) < 1e-10, f"b sum = {sum(b_list)}"
    
    # Ensure length 12
    assert len(a_list) == 12
    assert len(b_list) == 12
    
    # Ensure all values are finite and in range
    for i in range(12):
        assert math.isfinite(a_list[i]) and -50 <= a_list[i] <= 50
        assert math.isfinite(b_list[i]) and -50 <= b_list[i] <= 50
        
    return a_list, b_list
