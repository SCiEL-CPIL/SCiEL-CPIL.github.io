"""Product-formula design for Heisenberg-chain time evolution using 4th-order Suzuki-Trotter."""
import time

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter product formula for 12 stages.
    
    The product formula is U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A).
    
    We use 3 substeps of a 4th-order Yoshida decomposition.
    The 4th-order formula is:
        S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    where S_2(t) = exp(-i t A/2) * exp(-i t B) * exp(-i t A/2)
    
    c1 = 1/(2 - 2^(1/3))
    c2 = -2^(1/3)/(2 - 2^(1/3))
    
    Each S_2 step contributes one B factor and two A factors.
    For 3 S_4 substeps, each with 3 S_2 sub-steps, we have 9 S_2 steps total.
    But we need exactly 12 (a, b) pairs.
    
    Let's reconsider: each S_2(t) = exp(-i t A/2) * exp(-i t B) * exp(-i t A/2)
    This has the form: A-factor, B-factor, A-factor.
    
    But our formula is prod exp(-i b_k t B) exp(-i a_k t A), i.e., B then A for each k.
    
    So we need to restructure. Let's think of it as:
    S_2(t) = exp(-i t A/2) * exp(-i t B) * exp(-i t A/2)
    
    In our (B, A) pair form, we can write this as:
    - First: exp(-i 0 * B) * exp(-i t/2 * A)  -> b=0, a=t/2
    - Second: exp(-i t * B) * exp(-i 0 * A)   -> b=t, a=0
    - Third: exp(-i 0 * B) * exp(-i t/2 * A)  -> b=0, a=t/2
    
    So each S_2 step uses 3 (b, a) pairs.
    3 S_4 substeps * 3 S_2 steps each = 9 S_2 steps * 3 pairs = 27 pairs. Too many.
    
    For 12 stages, let's use a different approach.
    
    Actually, let's use the fact that we can combine consecutive A factors.
    
    Alternative: Use 4th-order with a different decomposition.
    
    Let me use the standard 4th-order coefficients and map them properly.
    
    The 4th-order formula with 6 exponentials (alternating A, B, A, B, A, B) is:
    exp(-i a1 t A) exp(-i b1 t B) exp(-i a2 t A) exp(-i b2 t B) exp(-i a3 t A) exp(-i b3 t B)
    
    But our form is prod exp(-i b_k t B) exp(-i a_k t A), which starts with B.
    
    Let's just use the 4th-order Yoshida coefficients directly.
    
    For a 4th-order 6-stage formula (which we can pad to 12 stages):
    The coefficients for the 4th-order formula S_4(t) are:
    c1 = 1/(2 - 2^(1/3)) ≈ 1.3512071919596575
    c2 = -2^(1/3)/(2 - 2^(1/3)) ≈ -0.7024143839193150
    
    S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    
    Each S_2(u) = exp(-i u A/2) * exp(-i u B) * exp(-i u A/2)
    
    So S_4(t) = exp(-i c1*t*A/2) * exp(-i c1*t*B) * exp(-i c1*t*A/2)
                * exp(-i c2*t*A/2) * exp(-i c2*t*B) * exp(-i c2*t*A/2)
                * exp(-i c1*t*A/2) * exp(-i c1*t*B) * exp(-i c1*t*A/2)
    
    Combining consecutive A terms:
    = exp(-i c1*t*A/2) * exp(-i c1*t*B) * exp(-i (c1/2 + c2/2)*t*A) * exp(-i c2*t*B)
      * exp(-i (c2/2 + c1/2)*t*A) * exp(-i c1*t*B) * exp(-i c1*t*A/2)
    
    = exp(-i c1*t*A/2) * exp(-i c1*t*B) * exp(-i (c1+c2)/2*t*A) * exp(-i c2*t*B)
      * exp(-i (c1+c2)/2*t*A) * exp(-i c1*t*B) * exp(-i c1*t*A/2)
    
    Now c1 + c2 = 1/(2-2^(1/3)) - 2^(1/3)/(2-2^(1/3)) = (1-2^(1/3))/(2-2^(1/3))
    
    Let's compute numerically:
    c1 ≈ 1.3512071919596575
    c2 ≈ -0.7024143839193150
    c1 + c2 ≈ 0.6487928080403425
    (c1+c2)/2 ≈ 0.32439640402017126
    
    So we have 7 factors: A, B, A, B, A, B, A
    
    In our (b, a) pair form where each pair is exp(-i b t B) exp(-i a t A):
    We need to convert this sequence.
    
    The sequence is: A(c1/2), B(c1), A((c1+c2)/2), B(c2), A((c1+c2)/2), B(c1), A(c1/2)
    
    To express as prod exp(-i b_k t B) exp(-i a_k t A):
    We can write:
    k=1: b=0, a=c1/2        (just the first A)
    k=2: b=c1, a=(c1+c2)/2  (B then A)
    k=3: b=c2, a=(c1+c2)/2  (B then A)
    k=4: b=c1, a=c1/2       (B then A)
    
    That's only 4 pairs. We need 12.
    
    Let's use 3 substeps of this 4-stage 4th-order formula, giving 12 pairs total.
    Each substep has 4 (b, a) pairs with the above coefficients scaled by 1/3.
    
    Wait, that gives 3*4 = 12. 
    
    But actually, I should just use the 4th-order formula directly and pad.
    
    Let me use a simpler approach: use the 4th-order coefficients and repeat.
    """
    # 4th-order Yoshida coefficients
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    # For 12 stages, use 3 substeps of a 4-stage 4th-order formula
    # Each 4-stage formula in (b, a) form:
    # Stage 1: b=0, a=c1/2
    # Stage 2: b=c1, a=(c1+c2)/2
    # Stage 3: b=c2, a=(c1+c2)/2
    # Stage 4: b=c1, a=c1/2
    
    n_substeps = stages // 4
    if n_substeps < 1:
        n_substeps = 1
    
    a = []
    b = []
    
    scale = 1.0 / n_substeps
    a1 = c1 / 2.0 * scale
    a2 = (c1 + c2) / 2.0 * scale
    b1 = c1 * scale
    b2 = c2 * scale
    
    for _ in range(n_substeps):
        a.append(a1)
        b.append(0.0)
        a.append(a2)
        b.append(b1)
        a.append(a2)
        b.append(b2)
        a.append(a1)
        b.append(b1)
    
    # Pad to exactly 'stages' length
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    
    # Ensure exactly 'stages' elements
    a = a[:stages]
    b = b[:stages]
    
    # Normalize to ensure sum(a) = sum(b) = 1
    sum_a = sum(a)
    sum_b = sum(b)
    
    if abs(sum_a) > 1e-15:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-15:
        b = [x / sum_b for x in b]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
