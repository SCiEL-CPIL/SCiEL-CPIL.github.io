import math

def coefficients(stages):
    """
    4th-order Suzuki-Trotter splitting mapped to 12 stages.
    Uses the structure S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    with S_2(t) = exp(-i t/2 B) exp(-i t A) exp(-i t/2 B).
    This results in a sequence of 4 B-exponents and 3 A-exponents,
    which is padded with zeros to fit the 12-stage (B, A) format.
    """
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    # Non-zero coefficients for B
    b1 = c1 / 2.0
    b2 = (c1 + c2) / 2.0
    b3 = c1 / 2.0
    
    # Non-zero coefficients for A
    a1 = c1
    a2 = c2
    a3 = c1
    
    # Map to 12 stages
    # Sequence: B_b1 A_0 B_0 A_a1 B_b2 A_0 B_0 A_a2 B_b3 A_0 B_0 A_a3 ... wait, the order is (B, A) pairs.
    # My derived sequence was:
    # B_{b1} A_{a1} B_{b2} A_{a2} B_{b3} A_{a3}
    # Wait, the combination was:
    # B_{c1/2} A_{c1} B_{(c1+c2)/2} A_{c2} B_{(c1+c2)/2} A_{c1} B_{c1/2}
    # This is 4 B's and 3 A's.
    # Let's place them in the 12 slots.
    
    b = [0.0] * 12
    a = [0.0] * 12
    
    # Place B's at indices 0, 2, 4, 6? No, the pairs are (B_k, A_k).
    # So B_0, A_0, B_1, A_1, ...
    # We want to place the non-zero B's and A's.
    # B's are at positions 0, 1, 2, 3 in the combined sequence?
    # The sequence is B A B A B A B.
    # So B_0, A_0, B_1, A_1, B_2, A_2, B_3.
    # But the format is (B_0, A_0), (B_1, A_1), ...
    # So we can map:
    # (B_0, A_0) = (b1, 0)
    # (B_1, A_1) = (0, a1)  -> No, this would be B_0 A_0 B_1 A_1.
    # The product is exp(-i b0 t B) exp(-i a0 t A) exp(-i b1 t B) ...
    # So if we set b0=b1, a0=0, b1=0, a1=a1, then we have B_{b1} A_0 B_0 A_{a1}.
    # This is not the same as B_{b1} A_{a1}.
    # We need to ensure the order is B then A.
    # So we can put B's in some slots and A's in others.
    # To get B_{b1} A_{a1} B_{b2} A_{a2} ..., we can set:
    # (b0, a0) = (b1, 0)
    # (b1, a1) = (0, a1)  -> This gives B_{b1} A_0 B_0 A_{a1}. This is NOT B_{b1} A_{a1}.
    # The product is sequential.
    # So we must have non-zero b_k and a_k in the same pair? No.
    # If b_k=0 and a_k!=0, it's just A.
    # If b_k!=0 and a_k=0, it's just B.
    # So we can represent B A B A as:
    # (b0, a0) = (b1, 0)
    # (b1, a1) = (0, a1)
    # (b2, a2) = (b2, 0)
    # (b3, a3) = (0, a2)
    # This works.
    
    b[0] = b1
    a[0] = 0.0
    
    b[1] = 0.0
    a[1] = a1
    
    b[2] = b2
    a[2] = 0.0
    
    b[3] = 0.0
    a[3] = a2
    
    b[4] = b2
    a[4] = 0.0
    
    b[5] = 0.0
    a[5] = a3
    
    b[6] = b3
    a[6] = 0.0
    
    # Remaining are 0
    
    return a, b
