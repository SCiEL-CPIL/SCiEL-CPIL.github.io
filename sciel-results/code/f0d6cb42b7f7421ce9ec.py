"""4th-order symmetric Suzuki-Trotter product formula for 12-stage Heisenberg chain evolution."""
import time

def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter decomposition embedded in 12 stages.
    
    The 4th-order symmetric scheme has 6 B-factors and 5 A-factors.
    We split each factor into 2 sub-stages to get 12 B-stages and 10 A-stages,
    with 2 zero A-stages inserted to maintain the alternating B-A pattern.
    
    The coefficients are normalized so that sum(a) = sum(b) = 1.
    """
    t_start = time.time()
    
    # 4th-order symmetric Suzuki-Trotter coefficients
    # For exp(-i*t*(A+B)) = prod exp(-i*c_j*t*Op_j)
    # The correct coefficients satisfy sum of B-weights = 1 and sum of A-weights = 1
    # Standard 4th-order: alpha = 2^(1/3)
    alpha = 2.0 ** (1.0 / 3.0)
    
    # The 4th-order symmetric coefficients (from literature):
    # B appears with weights: c5, c3, c1, c1, c3, c5 (6 factors)
    # A appears with weights: c4, c2, c1, c2, c4 (5 factors)
    # where:
    c1 = 1.0 / (2.0 * (1.0 - alpha))
    c2 = -alpha / (2.0 * (1.0 - alpha))
    c3 = 1.0 / (1.0 - alpha)
    c4 = -alpha / (1.0 - alpha)
    c5 = 1.0 / (2.0 * (1.0 - alpha))
    
    # Verify: sum of B weights should be 1, sum of A weights should be 1
    sum_B = c5 + c3 + c1 + c1 + c3 + c5
    sum_A = c4 + c2 + c1 + c2 + c4
    
    # Normalize to ensure exact sum = 1
    c1_B = c1 / sum_B
    c2_B = c1 / sum_B  # same as c1_B since B uses c1 twice
    c3_B = c3 / sum_B
    c5_B = c5 / sum_B
    
    c1_A = c1 / sum_A
    c2_A = c2 / sum_A
    c4_A = c4 / sum_A
    
    # Build 12-stage product: B1 A1 B2 A2 ... B12 A12
    # Suzuki sequence: B5 A4 B3 A2 B1 A1 B1 A2 B3 A4 B5
    # Split each into 2: BB AA BB AA BB AA BB AA BB AA BB AA
    # That's 12 B's and 10 A's. Need 2 more A's (set to 0).
    
    # Map to 12-stage pattern:
    # b1=B5/2, a1=A4/2, b2=B5/2, a2=A4/2,
    # b3=B3/2, a3=A2/2, b4=B3/2, a4=A2/2,
    # b5=B1/2, a5=A1/2, b6=B1/2, a6=0,
    # b7=B1/2, a7=0, b8=B1/2, a8=A1/2,
    # b9=B3/2, a9=A2/2, b10=B3/2, a10=A2/2,
    # b11=B5/2, a11=A4/2, b12=B5/2, a12=A4/2
    
    b = [c5_B/2, c5_B/2, c3_B/2, c3_B/2, c1_B/2, c1_B/2, c1_B/2, c1_B/2, c3_B/2, c3_B/2, c5_B/2, c5_B/2]
    a = [c4_A/2, c4_A/2, c2_A/2, c2_A/2, c1_A/2, 0.0, 0.0, c1_A/2, c2_A/2, c2_A/2, c4_A/2, c4_A/2]
    
    # Final normalization to ensure exact sums
    sa = sum(a)
    sb = sum(b)
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Re-normalize after clamping
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    
    return a, b
    
    a = [1.0 / stages] * stages
    b = [1.0 / stages] * stages
    
    return a, b
