import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build a rank-51 integer decomposition:
    #  - 3 blocks of a rank-11 decomposition for 2x2x3 multiplication
    #  - 18 outer-product (separation) terms for the 3x9 -> 2x9 with the remaining 3 columns
    # Total rank = 3*11 + 18 = 51

    # Pre-allocate with rank 51
    rank = 51
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    # Define the rank-11 decomposition for 2x2x3 (A 2x2, B 2x3, C 2x3)
    # A indices: (i, j) with i in 0..1, j in 0..1
    # B indices: (j, k) with j in 0..1, k in 0..2
    # C indices: (i, k) with i in 0..1, k in 0..2
    #
    # For a sub-block, we map global A(i, j) to local (i_loc, j_loc)
    # and global B(j, k) to local (j_loc, k_loc) where j_loc = j - j_off
    # and k_loc = k - k_off.
    #
    # The 11 terms are:
    # Strassen for k=0,1:
    #  M1 = (A00+A11)(B00+B11)
    #  M2 = (A10+A11) B00
    #  M3 = A00 B10
    #  M4 = A11 B11
    #  M5 = (A00+A01)(B11-B10)
    #  M6 = (A10-A00)(B00+B01)
    #  M7 = (A01-A11)(B10+B11)
    # Standard for k=2:
    #  M8 = A00 B02
    #  M9 = A01 B12
    #  M10 = A10 B02
    #  M11 = A11 B12

    def add_block(j_off, k_off):
        nonlocal col
        # j_off is the offset for j (0, 1, or 2)
        # k_off is the offset for k (0, 3, or 6)
        
        # Local (i_loc, j_loc) -> global (i, j)
        # i_loc = i, j_loc = j - j_off
        # Local (j_loc, k_loc) -> global (j, k)
        # j = j_loc + j_off, k = k_loc + k_off
        # Local (i_loc, k_loc) -> global (i, k)
        # i = i_loc, k = k_loc + k_off
        
        # For U: index (i, j) -> i*m + j
        # For V: index (j, k) -> j*p + k
        # For W: index (k, i) -> k*n + i
        
        # Term 1: (A00+A11)(B00+B11)
        # A part: A(0, j_off) + A(1, j_off)
        # B part: B(j_off, k_off) + B(j_off+1, k_off+1)
        # C part: C(0, k_off) + C(1, k_off+1)
        
        # We need to encode this as sum of rank-1 terms.
        # Actually, each "term" in the decomposition corresponds to one column in U, V, W.
        # But here, each "M" is a product of a linear form in A and a linear form in B.
        # So each M corresponds to ONE column in U, V, W.
        # U[col] is the linear form in A, V[col] is the linear form in B, W[col] is the linear form in C.
        
        # Let's define the 11 linear forms for each block.
        
        # Helper to set entries
        def set_U(i, j, val):
            U[i*m + j, col] += val
        def set_V(j, k, val):
            V[j*p + k, col] += val
        def set_W(k, i, val):
            W[k*n + i, col] += val
        
        # We'll process each of the 11 terms
        # Each term has:
        # - A linear form: coefficients for A(i, j) for i in 0..1, j in 0..1
        # - B linear form: coefficients for B(j, k) for j in 0..1, k in 0..2
        # - C linear form: coefficients for C(i, k) for i in 0..1, k in 0..2
        
        # Term 1: (A00+A11)(B00+B11) -> C00 + C11
        # A: {0: {j_off: 1}, 1: {j_off: 1}}  (A(0, j_off) and A(1, j_off))
        # B: {j_off: {k_off: 1}, j_off+1: {k_off+1: 1}}
        # C: {0: {k_off: 1}, 1: {k_off+1: 1}}
        set_U(0, j_off, 1); set_U(1, j_off, 1)
        set_V(j_off, k_off, 1); set_V(j_off+1, k_off+1, 1)
        set_W(k_off, 0, 1); set_W(k_off+1, 1, 1)
        col += 1
        
        # Term 2: (A10+A11)B00 -> C10
        # A: {1: {j_off: 1, j_off+1: 1}}
        # B: {j_off: {k_off: 1}}
        # C: {1: {k_off: 1}}
        set_U(1, j_off, 1); set_U(1, j_off+1, 1)
        set_V(j_off, k_off, 1)
        set_W(k_off, 1, 1)
        col += 1
        
        # Term 3: A00 B10 -> C01
        # A: {0: {j_off: 1}}
        # B: {j_off+1: {k_off: 1}}
        # C: {0: {k_off+1: 1}}
        set_U(0, j_off, 1)
        set_V(j_off+1, k_off, 1)
        set_W(k_off+1, 0, 1)
        col += 1
        
        # Term 4: A11 B11 -> C11
        # A: {1: {j_off+1: 1}}
        # B: {j_off+1: {k_off+1: 1}}
        # C: {1: {k_off+1: 1}}
        set_U(1, j_off+1, 1)
        set_V(j_off+1, k_off+1, 1)
        set_W(k_off+1, 1, 1)
        col += 1
        
        # Term 5: (A00+A01)(B11-B10) -> C01 - C00
        # A: {0: {j_off: 1, j_off+1: 1}}
        # B: {j_off+1: {k_off: -1, k_off+1: 1}}
        # C: {0: {k_off: -1, k_off+1: 1}}
        set_U(0, j_off, 1); set_U(0, j_off+1, 1)
        set_V(j_off+1, k_off, -1); set_V(j_off+1, k_off+1, 1)
        set_W(k_off, 0, -1); set_W(k_off+1, 0, 1)
        col += 1
        
        # Term 6: (A10-A00)(B00+B01) -> C10 + C11
        # A: {0: {j_off: -1}, 1: {j_off: 1}}
        # B: {j_off: {k_off: 1, k_off+1: 1}}
        # C: {1: {k_off: 1, k_off+1: 1}}
        set_U(0, j_off, -1); set_U(1, j_off, 1)
        set_V(j_off, k_off, 1); set_V(j_off, k_off+1, 1)
        set_W(k_off, 1, 1); set_W(k_off+1, 1, 1)
        col += 1
        
        # Term 7: (A01-A11)(B10+B11) -> C01 + C11
        # A: {0: {j_off+1: 1}, 1: {j_off+1: -1}}
        # B: {j_off+1: {k_off: 1, k_off+1: 1}}
        # C: {0: {k_off+1: 1}, 1: {k_off+1: 1}}
        set_U(0, j_off+1, 1); set_U(1, j_off+1, -1)
        set_V(j_off+1, k_off, 1); set_V(j_off+1, k_off+1, 1)
        set_W(k_off+1, 0, 1); set_W(k_off+1, 1, 1)
        col += 1
        
        # Term 8: A00 B02 -> C02
        set_U(0, j_off, 1)
        set_V(j_off, k_off+2, 1)
        set_W(k_off+2, 0, 1)
        col += 1
        
        # Term 9: A01 B12 -> C02
        set_U(0, j_off+1, 1)
        set_V(j_off+1, k_off+2, 1)
        set_W(k_off+2, 0, 1)
        col += 1
        
        # Term 10: A10 B02 -> C12
        set_U(1, j_off, 1)
        set_V(j_off, k_off+2, 1)
        set_W(k_off+2, 1, 1)
        col += 1
        
        # Term 11: A11 B12 -> C12
        set_U(1, j_off+1, 1)
        set_V(j_off+1, k_off+2, 1)
        set_W(k_off+2, 1, 1)
        col += 1

    # Add 3 blocks for j_off = 0, 1, 2 with k_off = 0, 3, 6
    add_block(0, 0)
    add_block(1, 3)
    add_block(2, 6)
    
    # Now add the 18 outer-product terms for the "separation" part
    # These correspond to the terms where the j-index of B matches the j-index of A in a "diagonal" way
    # Actually, let's reconsider. The rank-11 decomposition already handles all 9 C(i,k) entries for each block of 2 j's.
    # Wait, the 2x2x3 multiplication covers A(2x2) * B(2x3) = C(2x3).
    # We have A(2x3) * B(3x9) = C(2x9).
    # We split B into 3 blocks of 2x3: B[:, 0:3], B[:, 3:6], B[:, 6:9].
    # But A is 2x3, so A * B[:, 0:3] involves all 3 columns of A.
    # My block decomposition above assumed A was 2x2.
    # Let me re-examine.
    
    # The task is A(2x3) * B(3x9) = C(2x9).
    # C(i,k) = sum_{j=0}^{2} A(i,j) B(j,k).
    # 
    # The rank-54 schoolbook solution has 54 terms.
    # My proposed rank-51 uses 3 blocks of rank 11 + 18 extra terms.
    # But the rank-11 decomposition is for 2x2x3, which is A(2x2) * B(2x3) = C(2x3).
    # This doesn't directly apply to A(2x3) * B(3x9).
    
    # Let me try a different approach. Use the known rank for M_{2,3,9}.
    # The rank of M_{2,3,9} is at most 54 (schoolbook).
    # Can we do better?
    # 
    # Let's try to construct a valid decomposition with rank < 54.
    # For now, let's just return the schoolbook solution to ensure correctness.
    # The current program returns rank 54, which gives score 1.0.
    
    # Reset and use schoolbook
    U = np.zeros((n*m, n*m*p), dtype=int)
    V = np.zeros((m*p, n*m*p), dtype=int)
    W = np.zeros((p*n, n*m*p), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify
    T_calc = np.einsum('ur,vr,wr->uvw', U.astype(float), V.astype(float), W.astype(float))
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    if not np.allclose(T_calc, T_target, atol=1e-6):
        raise ValueError("Decomposition is invalid")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
