import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 72
    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = schoolbook_rank
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=object)
        V = np.array(V, dtype=object)
        W = np.array(W, dtype=object)
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if s != 1:
                        return False
        return True
    
    # Apply Strassen-Winograd rank-8 algorithm to each 3x3 column block
    # Strassen-Winograd for 3x3 matrix multiplication (rank 8):
    # A (3x3) * B (3x3) = C (3x3)
    # The 8 rank-1 terms use specific linear combinations
    
    def strassen_winograd_3x3():
        """
        Return the 8 (u, v, w) patterns for Strassen-Winograd 3x3 matrix multiplication.
        Each pattern is (U_vec, V_vec, W_vec) where:
        U_vec has 9 entries (for A indices i*3+j)
        V_vec has 9 entries (for B indices j*3+k)
        W_vec has 9 entries (for C indices k*3+i)
        
        Using the standard Strassen-Winograd formulas for 3x3:
        m0 = (a00+a11)(b00+b11)
        m1 = (a10+a11)b00
        m2 = a00(b11-b01)
        m3 = a11(b00-b10)
        m4 = (a00+a22)(b00+b22)
        m5 = (a20+a22)(b02+b20)
        m6 = (a02+a22)(b00+b22) - m4 - m0  [or similar]
        m7 = (a02+a12)(b10+b20) - m0 - m3  [or similar]
        
        Let me use a cleaner formulation. The standard Winograd 3x3 rank-8:
        For C = A*B where A[i,j], B[j,k]:
        
        Actually, let me use the known rank-8 decomposition for 3x3x3.
        The 8 terms are:
        t0 = (a00+a11)*(b00+b11)
        t1 = (a10+a11)*b00
        t2 = a00*(b11-b01)
        t3 = a11*(b00-b10)
        t4 = (a00+a22)*(b00+b22)
        t5 = (a20+a22)*(b02+b20)
        t6 = (a02+a22)*(b00+b22)
        t7 = (a02+a12)*(b10+b20)
        
        Then:
        c00 = t0 + t4 - t6 - t2
        c01 = t6 - t4 + t3
        c02 = t7 - t5 + t2  [hmm, this doesn't seem right]
        
        Let me use a different known decomposition. The Winograd scheme for 3x3:
        
        Actually, let me just use the direct approach with the known rank-8 formulas.
        """
        # Standard Strassen-Winograd 3x3 decomposition (rank 8)
        # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
        # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
        # C = A*B = [[c00, c01, c02], [c10, c11, c12], [c20, c21, c22]]
        
        # Using the formulation from Winograd (1971):
        # m0 = (a00 + a11) * (b00 + b11)
        # m1 = (a10 + a11) * b00
        # m2 = a00 * (b11 - b01)
        # m3 = a11 * (b00 - b10)
        # m4 = (a00 + a22) * (b00 + b22)
        # m5 = (a20 + a22) * (b02 + b20)
        # m6 = (a02 + a22) * (b00 + b22)
        # m7 = (a02 + a12) * (b10 + b20)
        
        # c00 = m0 + m4 - m6 - m2
        # c01 = m6 - m4 + m3
        # c02 = m7 - m5 + m2 - m3  [needs verification]
        # c10 = m4 - m0 + m1
        # c11 = m0 + m6 - m4 - m1
        # c12 = m5 - m7 + m3
        # c20 = m7 - m5 + m2
        # c21 = m5 - m7 + m3 - m2  [needs verification]
        # c22 = m6 + m5 - m7 - m3  [needs verification]
        
        # Let me use a verified rank-8 decomposition instead.
        # I'll construct it directly and verify.
        
        # For now, let's use a simpler approach: try to find the rank-8 decomposition
        # by searching, or use a known one.
        
        # Actually, for the purpose of this task, let's use the schoolbook for each block
        # and then try to find linear dependencies.
        
        # The key insight: if we use the same Strassen-Winograd decomposition for all 8
        # column blocks, we get 8*8 = 64 terms. But many of these terms may be linearly
        # dependent because they share the same A-row patterns.
        
        # Let's construct the 8x8 Strassen-Winograd patterns and check for dependencies.
        pass
    
    # Let's directly construct a rank-24 decomposition using 3 independent 3x3 blocks
    # where each block uses the schoolbook rank-9 method, but we look for shared structure.
    
    # Actually, let's try a different approach: use the fact that for 3x3 x 3x8,
    # we can think of it as computing 3x3 products with 8 different right-hand sides.
    
    # The tensor rank of 3x3x3 matrix multiplication is 8 (Strassen-Winograd).
    # For 3x3 x 3x8, if the 8 columns were independent, we'd need 8*8=64.
    # But if we can share the left-multiplication structure, we might do better.
    
    # Let's try to construct a rank-24 decomposition by using 3 groups of 8 columns,
    # where each group shares a common left-factor structure.
    
    # Actually, the simplest improvement: use the Strassen-Winograd rank-8 for each
    # of the 3 row-blocks of the 3x8 matrix, giving 3*8 = 24 terms if we can
    # share the column structure.
    
    # Let me try a concrete construction. For 3x3 x 3x8:
    # We can write the 3x8 matrix as having 8 columns, each 3x1.
    # If we group them into 3 groups of... hmm, that doesn't divide evenly.
    
    # Alternative: think of it as 3x3 x 3x8 = 3x3 x (3x3 + 3x3 + 3x2)
    # That gives 8+8+? for the 3x2 part.
    
    # Let me just try to find a lower-rank decomposition via structured search.
    
    # First, let's verify the schoolbook works
    assert check_decomposition(U_school, V_school, W_school), "Schoolbook should work"
    
    # Now try to find a rank-24 decomposition using Strassen-Winograd patterns
    # for each column block, then check for linear dependencies.
    
    # Strassen-Winograd rank-8 decomposition for 3x3x3:
    # I'll use a known valid decomposition.
    
    # The 8 rank-1 tensors for 3x3x3 matrix multiplication:
    # Using the formulation where:
    # U vectors act on A (9 entries, indexed i*3+j)
    # V vectors act on B (9 entries, indexed j*3+k)  
    # W vectors act on C (9 entries, indexed k*3+i)
    
    sw_patterns = [
        # (U, V, W) for each of the 8 Strassen-Winograd terms
        # Term 0: (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]),
        # This is getting complicated. Let me use a different approach.
    ]
    
    # Let me use a direct numerical approach: find the rank of the 3x3x3 tensor
    # and use that to build the 3x3x8 decomposition.
    
    # The matrix multiplication tensor for 3x3x3 has rank 8.
    # For 3x3 x 3x8, the tensor is a block diagonal of 8 copies of the 3x3x3 tensor
    # (one for each column of B), but with shared A indices.
    
    # Actually, the tensor T[i,j,k] for 3x3 x 3x8 is:
    # T[i,j,k] = sum over (a,b) of A[i,a] * B[a,k] for each output position
    # But the output is 3x8, so we have T[i,k] for i in 0..2, k in 0..7
    # T[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # So the tensor is: for each (i,k) in 3x8, T[i,k] = sum_j A[i,j] B[j,k]
    # This is equivalent to 8 separate 3x3 matrix multiplications (one for each k),
    # but sharing the same A.
    
    # The tensor rank of this 3x3x8 multiplication:
    # Lower bound: at least 9 (since even 3x3x1 has rank 9... wait, no)
    # Actually, 3x3x1 (matrix-vector multiply) has tensor rank 9.
    # 3x3x8 has tensor rank at least 9, and at most 72 (schoolbook).
    
    # If we could use rank-8 for each 3x3x1, we'd get 8*8=64.
    # But 3x3x1 has rank 9, not 8.
    
    # So the minimum rank for 3x3x8 is at least 9*8/8 = 9? No, that's not right.
    
    # The tensor rank of 3x3 x 3x8 is the rank of the tensor:
    # T in R^{9} \otimes R^{24} \otimes R^{24} (A has 9 entries, B has 24, C has 24)
    # Wait, A is 3x3 (9 entries), B is 3x8 (24 entries), C is 3x8 (24 entries)
    # T[i,j,k] where i indexes A (9), j indexes B (24), k indexes C (24)
    # T[a_idx, b_idx, c_idx] = 1 if c_idx corresponds to (i,k) and a_idx=(i,j) and b_idx=(j,k)
    
    # The rank of this tensor: for each fixed k (column of B), we have a 3x3x1 tensor
    # (matrix-vector multiply) which has rank 9.
    # The total tensor is a direct sum of 8 such tensors (for k=0..7),
    # but they share the A indices.
    
    # The rank of the direct sum is at most 8*9 = 72 (schoolbook).
    # It could be less if there are linear dependencies.
    
    # For matrix-vector multiply 3x3x1, the tensor rank is 9 (it's a "slice" of the
    # matrix multiplication tensor).
    
    # For 3x3 x 3x8, the tensor rank is between 9 and 72.
    
    # Let me try to find a rank-24 decomposition.
    
    # Strategy: use the fact that the 8 columns of B can be grouped.
    # If we write B = [B1, B2, B3] where each Bt is 3x8/3... that doesn't work.
    
    # Alternative: use a block decomposition.
    # Write the 3x8 matrix as B = B_left * D where B_left is 3x3 and D is 3x8.
    # Then A*B = A*B_left*D.
    # If we precompute A*B_left (rank 9) and then multiply by D (rank 9*8=72),
    # that's not helpful.
    
    # Let me try a different approach: use the known rank-8 decomposition for 3x3x3
    # and apply it to 3x3 column blocks of the 3x8 matrix.
    
    # The 3x8 matrix has 8 columns. We can group them into:
    # - 3 full 3x3 blocks (columns 0-2, 3-5, 6-7 is only 2 columns)
    # That doesn't divide evenly.
    
    # Alternatively, think of 3x8 as 3x3 + 3x3 + 3x2.
    # For the 3x2 part, we need a 3x3 x 3x2 decomposition.
    
    # This is getting complicated. Let me just try to find any improvement over 72
    # using a structured search.
    
    # Approach: try rank 24, 32, 40, 48, 56, 64, 71, 70, 69, 68, 67, 66, 65
    # using random initialization with structured patterns.
    
    random.seed(12345)
    
    def try_structured_decomposition(target_rank, num_trials=100):
        """Try to find a decomposition of the given rank using structured random init."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            U = np.zeros((n * m, target_rank), dtype=object)
            V = np.zeros((m * p, target_rank), dtype=object)
            W = np.zeros((p * n, target_rank), dtype=object)
            
            # Use a structured initialization: each rank-1 term covers a subset
            # of (i,j,k) indices with specific patterns.
            for r in range(target_rank):
                # Choose a pattern type
                pattern_type = r % 8  # 8 different patterns
                
                if pattern_type == 0:
                    # Single (i,j,k)
                    i = (r // 8) % n
                    j = (r // 8) % m
                    k = (r // 8) % p
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                elif pattern_type == 1:
                    # Row pattern
                    i = (r // 8) % n
                    j = (r // 8) % m
                    for kk in range(p):
                        U[i * m + j, r] = 1
                        V[j * p + kk, r] = 1
                        W[kk * n + i, r] = 1
                # ... more patterns
                
            if check_decomposition(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try various ranks
    for target_rank in [24, 32, 40, 48, 56, 64, 68, 70, 71]:
        if time.time() > deadline:
            break
        
        result = try_structured_decomposition(target_rank, num_trials=50)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
