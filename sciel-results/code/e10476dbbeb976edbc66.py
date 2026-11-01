def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 10, 01, 11 (Note: C index order in W is k*n + i, so k is row, i is col)
    # W index: k*n + i.
    # Local C indices:
    # 0: k=0, i=0 -> 0
    # 1: k=0, i=1 -> 1? No, k*n+i. n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # Strassen M1..M7
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A11) * (B00 + B11)  <- Wait, standard Strassen:
    
    # Standard Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},       # M1: B00 + B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01 - B11
        {2: 1, 0: -1},      # M4: B10 - B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00 + B01
        {2: 1, 3: 1}        # M7: B10 + B11
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00 + C11
        {1: 1, 3: -1},      # M2: C10 - C11
        {2: 1, 3: 1},       # M3: C01 + C11
        {0: 1, 1: 1},       # M4: C00 + C10
        {0: -1, 2: 1},      # M5: -C00 + C01
        {3: 1},             # M6: C11
        {0: 1}              # M7: C00
    ]
    
    # Note: W index mapping in code:
    # w_idx = k * n + i
    # Local C index 0 (k=0, i=0) -> w_idx 0
    # Local C index 1 (k=0, i=1) -> w_idx 1
    # Local C index 2 (k=1, i=0) -> w_idx 2
    # Local C index 3 (k=1, i=1) -> w_idx 3
    
    # My W_coeffs above used:
    # M1: C00, C11 -> indices 0, 3. Correct.
    # M2: C10, C11 -> indices 2, 3. 
    # Wait, C10 is k=1, i=0 -> index 2.
    # C11 is k=1, i=1 -> index 3.
    # So M2 should be {2: 1, 3: -1}.
    # My previous W_coeffs[1] was {1: 1, 3: -1}. This corresponds to C01 and C11?
    # C01 is k=0, i=1 -> index 1.
    # So {1: 1, 3: -1} is C01 - C11.
    # But Strassen M2 contributes to C10 and C11.
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    # So M2 should have +1 at C10 (index 2) and -1 at C11 (index 3).
    # So W_coeffs[1] should be {2: 1, 3: -1}.
    
    # Let's re-verify W_coeffs.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W[0] (C00): M1(+), M4(+), M5(-), M7(+)
    # W[1] (C01): M3(+), M5(+)
    # W[2] (C10): M2(+), M4(+)
    # W[3] (C11): M1(+), M2(-), M3(+), M6(+)
    
    # So for each r, W_coeffs[s] should have entries for the C indices involved in M_s.
    # M1: C00(+), C11(+) -> {0: 1, 3: 1}
    # M2: C10(+), C11(-) -> {2: 1, 3: -1}
    # M3: C01(+), C11(+) -> {1: 1, 3: 1}
    # M4: C00(+), C10(+) -> {0: 1, 2: 1}
    # M5: C00(-), C01(+) -> {0: -1, 1: 1}
    # M6: C11(+) -> {3: 1}
    # M7: C00(+) -> {0: 1}
    
    # The current program has:
    # W_coeffs[1] = {1: 1, 3: -1} -> C01 - C11. Incorrect. Should be C10 - C11.
    # W_coeffs[2] = {2: 1, 3: 1} -> C10 + C11. Incorrect. Should be C01 + C11.
    # W_coeffs[3] = {0: 1, 1: 1} -> C00 + C01. Incorrect. Should be C00 + C10.
    # W_coeffs[4] = {0: -1, 2: 1} -> -C00 + C10. Incorrect. Should be -C00 + C01.
    
    # It seems the current program has bugs in W_coeffs!
    # Let's check the current program's W_coeffs again.
    # W_coeffs = [
    #     {0: 1, 3: 1},       # M1: C00, C11. Correct.
    #     {1: 1, 3: -1},      # M2: C01, -C11. Incorrect.
    #     {2: 1, 3: 1},       # M3: C10, C11. Incorrect.
    #     {0: 1, 1: 1},       # M4: C00, C01. Incorrect.
    #     {0: -1, 2: 1},      # M5: -C00, C10. Incorrect.
    #     {3: 1},             # M6: C11. Correct.
    #     {0: 1}              # M7: C00. Correct.
    # ]
    
    # If the current program is buggy, it might be producing an invalid decomposition?
    # But the score is 1.1429, which is > 1.0.
    # If it were invalid, score would be 0.
    # So it must be valid.
    # Did I misinterpret the index mapping?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # k is row, i is col.
    # C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Wait, the problem says "multiplying a 2 by 2 matrix by a 2 by 8 matrix".
    # Usually $C = A B$. $A$ is $2 \times 2$, $B$ is $2 \times 8$. $C$ is $2 \times 8$.
    # Indices: $C_{i,k}$ where $i \in \{0,1\}, k \in \{0..7\}$.
    # $W$ has shape $(p \times n, \text{rank}) = (8 \times 2, \text{rank}) = (16, \text{rank})$.
    # Index $w\_idx = k \cdot n + i = 2k + i$.
    # So $W[2k+i, r]$.
    # This matches $W[k \cdot n + i, r]$.
    
    # In my analysis:
    # C00: k=0, i=0 -> index 0.
    # C01: k=1, i=0? No.
    # $C_{i,k}$.
    # If $i$ is row, $k$ is col.
    # $W$ index is $k \cdot n + i$.
    # So for fixed $k$, $i$ varies 0,1.
    # $k=0$: indices 0, 1. ($C_{0,0}, C_{1,0}$)
    # $k=1$: indices 2, 3. ($C_{0,1}, C_{1,1}$)
    
    # So:
    # Index 0: $C_{0,0}$
    # Index 1: $C_{1,0}$
    # Index 2: $C_{0,1}$
    # Index 3: $C_{1,1}$
    
    # Let's re-evaluate W_coeffs with this mapping.
    # C00 (idx 0)
    # C10 (idx 1)
    # C01 (idx 2)
    # C11 (idx 3)
    
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # M1: C00(+), C11(+) -> {0: 1, 3: 1}
    # M2: C10(+), C11(-) -> {1: 1, 3: -1}
    # M3: C01(+), C11(+) -> {2: 1, 3: 1}
    # M4: C00(+), C10(+) -> {0: 1, 1: 1}
    # M5: C00(-), C01(+) -> {0: -1, 2: 1}
    # M6: C11(+) -> {3: 1}
    # M7: C00(+) -> {0: 1}
    
    # Comparing with current program:
    # W_coeffs[0] = {0: 1, 3: 1}. Matches M1.
    # W_coeffs[1] = {1: 1, 3: -1}. Matches M2.
    # W_coeffs[2] = {2: 1, 3: 1}. Matches M3.
    # W_coeffs[3] = {0: 1, 1: 1}. Matches M4.
    # W_coeffs[4] = {0: -1, 2: 1}. Matches M5.
    # W_coeffs[5] = {3: 1}. Matches M6.
    # W_coeffs[6] = {0: 1}. Matches M7.
    
    # So the current program is CORRECT.
    # My previous confusion was due to mixing up $C_{i,k}$ and $C_{k,i}$ or index order.
    
    # So the current program is a valid rank-28 decomposition.
    # Since 28 is the theoretical minimum, we cannot improve the rank.
    # We should just return the program as is, or slightly optimized for speed/clarity.
    
    # I will provide the same logic but with a cleaner structure.
