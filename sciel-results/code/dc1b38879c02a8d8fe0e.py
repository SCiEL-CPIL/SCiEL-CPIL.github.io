"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # For 2x2x8 matrix multiplication, we use a decomposition that exploits
    # the structure better than 4 independent Strassen blocks.
    # 
    # Key insight: C[k,i] = A[0,i]*B[0,k] + A[1,i]*B[1,k]
    # We can use a rank-24 decomposition by grouping the 8 k-values into
    # 4 groups of 2, but using a smarter combination that shares work.
    #
    # Actually, let's use the known optimal decomposition for 2x2xp.
    # For p=8, the minimal rank is known to be 2p = 16 for the trivial case,
    # but Strassen-based gives 28. Let's try to find something in between.
    #
    # Alternative: use rank 24 by combining pairs of Strassen blocks.
    # For each pair of k-values (k, k+1), we compute a 2x2x4 problem.
    # The 2x2x4 tensor has rank 12 (using 2 Strassen blocks worth of sharing).
    # So 2 pairs * 12 = 24.
    
    # Let me try a direct construction with rank 24.
    # For each i (0,1) and k (0..7), C[k,i] = sum_j A[i,j]*B[j,k]
    # 
    # I'll use the following approach: for each of the 4 pairs of consecutive k values,
    # use a 2x2x4 decomposition of rank 6 (if possible) or fall back to 12 per pair.
    
    # Actually, let me just verify the Strassen approach is correct and try rank 24
    # with a different decomposition pattern.
    
    # Use rank 24: 6 terms per group of 2 k-values (4 groups)
    # For a 2x2x4 block (A is 2x2, B is 2x4, C is 4x2), 
    # we can use 6 multiplications if we structure it right.
    
    # Let me construct a rank-24 solution:
    # Group k into pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair, use a 2x2x4 decomposition with 6 terms.
    
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For a 2x2x4 block with B columns (2q, 2q+1):
    # C[2q,0] = A[0,0]*B[0,2q] + A[1,0]*B[1,2q]
    # C[2q,1] = A[0,1]*B[0,2q] + A[1,1]*B[1,2q]
    # C[2q+1,0] = A[0,0]*B[0,2q+1] + A[1,0]*B[1,2q+1]
    # C[2q+1,1] = A[0,1]*B[0,2q+1] + A[1,1]*B[1,2q+1]
    #
    # Using 6 terms:
    # M1 = (A00+A11) * (B0k0+B1k1) -> contributes to multiple C entries
    # This gets complex. Let me use a simpler approach.
    
    # Actually, the minimal rank for 2x2x4 is 8 (trivial) or 7*2=14 (Strassen per 2x2x2).
    # There's no known rank-6 decomposition for 2x2x4.
    
    # Let me stick with rank 28 but verify it's correct, then try to optimize.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A indices: 00->0, 01->1, 10->2, 11->3)
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # Strassen coefficients for V (local B indices: 00->0, 01->1, 10->2, 11->3)
    V_coeffs = [
        {0: 1, 3: 1},       # M1: B00 + B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01 - B11
        {2: 1, 0: -1},      # M4: B10 - B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00 + B01
        {2: 1, 3: 1}        # M7: B10 + B11
    ]
    
    # Strassen coefficients for W (local C indices: 00->0, 10->1, 01->2, 11->3)
    # W index = k*n + i, so local: 0->(k=0,i=0), 1->(k=0,i=1), 2->(k=1,i=0), 3->(k=1,i=1)
    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00 + C11
        {1: 1, 3: -1},      # M2: C01 - C11
        {2: 1, 3: 1},       # M3: C10 + C11
        {0: 1, 1: 1},       # M4: C00 + C01
        {0: -1, 2: 1},      # M5: -C00 + C10
        {3: 1},             # M6: C11
        {0: 1}              # M7: C00
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
                
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val
                
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
