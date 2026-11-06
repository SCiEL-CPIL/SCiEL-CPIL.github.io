import numpy as np

def coefficients(stages):
    """
    Returns coefficients for a product formula U(t) = prod_{k=1}^{stages} exp(-i b_k t B) exp(-i a_k t A).
    Uses a 4th-order symmetric Suzuki-Trotter scheme for stages=12, falling back to Strang splitting otherwise.
    """
    if stages == 12:
        # 4th order Suzuki-Trotter coefficients
        # S_4(t) = S_2(c1*t) S_2(c2*t) S_2(c1*t)
        # c1 = 1 / (2 - 2^(1/3)), c2 = -2^(1/3) / (2 - 2^(1/3))
        x = 2.0 ** (1.0 / 3.0)
        c1 = 1.0 / (2.0 - x)
        c2 = -x / (2.0 - x)
        
        # Expansion:
        # B: c1/2
        # A: c1
        # B: (c1+c2)/2
        # A: c2
        # B: (c1+c2)/2
        # A: c1
        # B: c1/2
        
        # Map to 12 stages: b_k, a_k
        # We have 7 B terms and 3 A terms.
        # Structure: B1 A1 B2 A2 B3 A3 B4 A4 B5 A5 B6 A6 B7 A7 ...
        # The expansion ends with B. So the last A is 0.
        
        a = [0.0] * stages
        b = [0.0] * stages
        
        a[0] = c1
        b[0] = c1 / 2.0
        
        a[1] = c2
        b[1] = (c1 + c2) / 2.0
        
        a[2] = c1
        b[2] = (c1 + c2) / 2.0
        
        a[3] = 0.0
        b[3] = c1 / 2.0
        
        # Remaining stages are 0
        # Sum a = c1 + c2 + c1 = 2c1 + c2 = 1
        # Sum b = c1/2 + (c1+c2)/2 + (c1+c2)/2 + c1/2 = c1 + c1 + c2 = 1
        
        return a, b
    else:
        # Fallback: Strang splitting (2nd order)
        substeps = stages // 2
        if substeps == 0:
            return [1.0] + [0.0]*(stages-1), [1.0] + [0.0]*(stages-1)
            
        a = [0.0] * stages
        b = [0.0] * stages
        
        # Distribute Strang steps
        # Each Strang step: 0.5/substeps for A, 1/substeps for B (split into 0.5/substeps before and after A)
        # But the form is B A B A ...
        # Let's just use the simple Strang logic from the seed, adapted to the B-A order.
        # Seed: a=[0.5/sub]*2*sub, b=[1/sub, 0]*sub
        # This corresponds to:
        # Step 1: B(1/sub) A(0.5/sub) ... wait, seed b has 1/sub then 0.
        # Let's stick to the seed's logic for fallback as it is known to be valid.
        
        # Re-implementing seed logic for fallback
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        
        # Pad to length stages
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        a = a[:stages]
        b = b[:stages]
        
        # Normalize to ensure sum is 1 if truncation happened (unlikely if stages is multiple of 2)
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a != 0:
            a = [x / sum_a for x in a]
        if sum_b != 0:
            b = [x / sum_b for x in b]
            
        return a, b
