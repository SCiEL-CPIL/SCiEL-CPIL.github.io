import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Split p=10 into 5 blocks of size 2
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        base_rank = b * rank_per_block
        # Local k indices: 0, 1 correspond to global k: 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for j=0,1
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B00)
        # M4 = A11(B01-B00) -- Wait, standard Strassen is for 2x2.
        # Let's use the standard 7-term Strassen for 2x2x2.
        # C00 = M1 - M4 + M6 - M7 ? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B00)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M7
        
        # We need to map these to U, V, W.
        # U[i, j], V[j, k], W[k, i]
        
        # Term 0: M1
        # U: A00, A11 -> U[0,0]=1, U[1,1]=1
        # V: B00, B11 -> V[0,k0]=1, V[1,k1]=1
        # W: C00, C11 -> W[k0,0]=1, W[k1,1]=1
        # But W has signs.
        # Let's define the coefficients.
        
        # It is easier to define the decomposition explicitly.
        # We will add contributions to U, V, W for each of the 11 terms.
        
        # Helper to add a term
        def add_term(u_coeffs, v_coeffs, w_coeffs, r_idx):
            # u_coeffs: list of (idx, val) for U
            # v_coeffs: list of (idx, val) for V
            # w_coeffs: list of (idx, val) for W
            for idx, val in u_coeffs:
                U[idx][r_idx] += val
            for idx, val in v_coeffs:
                V[idx][r_idx] += val
            for idx, val in w_coeffs:
                W[idx][r_idx] += val
        
        # Local indices for A: i in 0,1; j in 0,1,2
        # Local indices for B: j in 0,1,2; k in 0,1
        # Local indices for C: i in 0,1; k in 0,1
        
        # Strassen Terms
        # 0: M1
        add_term(
            [(0*3+0, 1), (1*3+1, 1)], # A00, A11
            [(0*10+k0, 1), (1*10+k1, 1)], # B00, B11
            [(k0*2+0, 1), (k1*2+1, 1)], # C00, C11
            base_rank + 0
        )
        
        # 1: M2
        add_term(
            [(1*3+0, 1), (1*3+1, 1)], # A10, A11
            [(0*10+k0, 1)], # B00
            [(k0*2+1, 1)], # C10
            base_rank + 1
        )
        
        # 2: M3
        add_term(
            [(0*3+0, 1)], # A00
            [(1*10+k1, 1), (0*10+k0, -1)], # B11, -B00
            [(k0*2+0, 1)], # C00 ? No, C01 = M3+M5. So W for C01.
            # Wait, M3 contributes to C01.
            # C01 = M3 + M5.
            # So W[k1, 0] += 1
            base_rank + 2
        )
        # Correction: M3 = A00(B11-B00).
        # V: B11 (j=1, k=1) and -B00 (j=0, k=0).
        # U: A00 (i=0, j=0).
        # W: C01 (i=0, k=1). Index k1*2+0.
        
        # 3: M4
        add_term(
            [(1*3+1, 1)], # A11
            [(0*10+k1, 1), (0*10+k0, -1)], # B01, -B00
            [(k0*2+1, 1), (k0*2+0, -1)], # C10, -C00
            base_rank + 3
        )
        # M4 = A11(B01-B00).
        # Contributes to C10 (+) and C00 (-).
        
        # 4: M5
        add_term(
            [(0*3+0, 1), (0*3+1, 1)], # A00, A01
            [(1*10+k1, 1)], # B11
            [(k1*2+0, 1)], # C01
            base_rank + 4
        )
        
        # 5: M6
        add_term(
            [(1*3+0, 1), (0*3+0, -1)], # A10, -A00
            [(0*10+k0, 1), (1*10+k0, 1)], # B00, B10
            [(k0*2+0, 1)], # C00
            base_rank + 5
        )
        
        # 6: M7
        add_term(
            [(0*3+1, 1), (1*3+1, -1)], # A01, -A11
            [(1*10+k1, 1), (0*10+k1, 1)], # B11, B01
            [(k0*2+0, -1), (k1*2+1, 1)], # -C00, C11
            base_rank + 6
        )
        
        # 7-10: Terms for j=2
        # 7: A02 B20
        add_term(
            [(0*3+2, 1)], # A02
            [(2*10+k0, 1)], # B20
            [(k0*2+0, 1)], # C00
            base_rank + 7
        )
        
        # 8: A02 B21
        add_term(
            [(0*3+2, 1)], # A02
            [(2*10+k1, 1)], # B21
            [(k1*2+0, 1)], # C01
            base_rank + 8
        )
        
        # 9: A12 B20
        add_term(
            [(1*3+2, 1)], # A12
            [(2*10+k0, 1)], # B20
            [(k0*2+1, 1)], # C10
            base_rank + 9
        )
        
        # 10: A12 B21
        add_term(
            [(1*3+2, 1)], # A12
            [(2*10+k1, 1)], # B21
            [(k1*2+1, 1)], # C11
            base_rank + 10
        )
        
    return {"U": U, "V": V, "W": W}
