import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1 for the matrix multiplication tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor T."""
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
        return np.max(np.abs(T_recon - T))

    def schoolbook():
        """Schoolbook decomposition with rank n*m*p = 120."""
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def strassen_2x2():
        """
        Strassen rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
        Returns U (4,7), V (4,7), W (4,7) for the 2x2x2 tensor.
        U index: i*2+j, V index: j*2+k, W index: k*2+i
        """
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        
        # Strassen's 7 multiplications for 2x2:
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # M1 = (A10 + A11) * (B00 + B01)
        # M2 = (A10 + A11) * (B10 - B00)
        # M3 = (A00 - A11) * (B10 + B11)
        # M4 = A00 * (B01 - B11)
        # M5 = A11 * (B00 + B11)
        # M6 = (A00 + A01) * (B10 - B11)
        # M7 = (A01 - A10) * (B00 + B10)
        
        # C00 = M5 + M3 - M7 - M1
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 + M4 - M2 - M6
        
        # M1 = (A10 + A11) * (B00 + B01)
        t = 0
        U[1 * 2 + 0, t] = 1  # A10
        U[1 * 2 + 1, t] = 1  # A11
        V[0 * 2 + 0, t] = 1  # B00
        V[0 * 2 + 1, t] = 1  # B01
        W[0 * 2 + 0, t] = -1  # C00 -= M1
        W[1 * 2 + 1, t] = 1   # C11 += M1
        
        # M2 = (A10 + A11) * (B10 - B00)
        t = 1
        U[1 * 2 + 0, t] = 1  # A10
        U[1 * 2 + 1, t] = 1  # A11
        V[1 * 2 + 0, t] = 1  # B10
        V[0 * 2 + 0, t] = -1 # B00
        W[1 * 2 + 0, t] = 1   # C10 += M2
        W[1 * 2 + 1, t] = -1  # C11 -= M2
        
        # M3 = (A00 - A11) * (B10 + B11)
        t = 2
        U[0 * 2 + 0, t] = 1  # A00
        U[1 * 2 + 1, t] = -1 # A11
        V[1 * 2 + 0, t] = 1  # B10
        V[1 * 2 + 1, t] = 1  # B11
        W[0 * 2 + 0, t] = 1   # C00 += M3
        W[1 * 2 + 0, t] = 1   # C10 += M3
        
        # M4 = A00 * (B01 - B11)
        t = 3
        U[0 * 2 + 0, t] = 1  # A00
        V[0 * 2 + 1, t] = 1  # B01
        V[1 * 2 + 1, t] = -1 # B11
        W[0 * 2 + 1, t] = 1   # C01 += M4
        W[1 * 2 + 1, t] = 1   # C11 += M4
        
        # M5 = A11 * (B00 + B11)
        t = 4
        U[1 * 2 + 1, t] = 1  # A11
        V[0 * 2 + 0, t] = 1  # B00
        V[1 * 2 + 1, t] = 1  # B11
        W[0 * 2 + 0, t] = 1   # C00 += M5
        W[0 * 2 + 1, t] = 1   # C01 += M5
        
        # M6 = (A00 + A01) * (B10 - B11)
        t = 5
        U[0 * 2 + 0, t] = 1  # A00
        U[0 * 2 + 1, t] = 1  # A01
        V[1 * 2 + 0, t] = 1  # B10
        V[1 * 2 + 1, t] = -1 # B11
        W[1 * 2 + 1, t] = -1  # C11 -= M6
        
        # M7 = (A01 - A10) * (B00 + B10)
        t = 6
        U[0 * 2 + 1, t] = 1  # A01
        U[1 * 2 + 0, t] = -1 # A10
        V[0 * 2 + 0, t] = 1  # B00
        V[1 * 2 + 0, t] = 1  # B10
        W[0 * 2 + 0, t] = -1  # C00 -= M7
        
        return U, V, W

    def build_block_decomposition():
        """
        Build a decomposition using Strassen for 2x2x2 blocks and schoolbook for strips.
        
        Partition:
        A (4x5) = [A_top(2x5) | A_bottom(2x5)]
        B (5x6) = [B_left(5x3); B_right(5x3)]
        
        We can partition further:
        A (4x5) = [[A11(2x2), A12(2x3)], [A21(2x2), A22(2x3)]]
        B (5x6) = [[B11(2x3), B12(2x3)], [B21(3x3), B22(3x3)]]
        
        Actually, let's use a cleaner partition:
        A (4x5): rows 0-3, cols 0-4
        B (5x6): rows 0-4, cols 0-5
        
        Partition the middle dimension (j) into 2+3:
        A = [A_2 (4x2) | A_3 (4x3)]
        B = [B_2 (2x6); B_3 (3x6)]
        
        C = A_2 * B_2 + A_3 * B_3
        
        For A_2 * B_2: (4x2) * (2x6)
        Partition A_2 into top 2x2 and bottom 2x2:
        A_2 = [A2_top (2x2); A2_bot (2x2)]
        B_2 = [B2_left (2x3) | B2_right (2x3)]
        
        C_2 = A2_top * B2_left + A2_top * B2_right + A2_bot * B2_left + A2_bot * B2_right
        
        Each 2x2 * 2x3 can be decomposed using Strassen on the 2x2 part of B.
        Actually, 2x2 * 2x3: rank at most 2*2*3 = 12 schoolbook, or we can use
        Strassen for the 2x2*2x2 part (rank 7) + 2x2*2x1 (rank 4) = 11.
        
        This is getting complex. Let me try a different approach.
        
        Use the known fact:
        - 2x2*2x2: rank 7 (Strassen)
        - 2x2*2x3: rank at most 12 (schoolbook) or better
        
        Let me just use a hybrid approach:
        1. For the 4x4*4x4 core (first 4 rows of A, first 4 cols of B), use a
           decomposition based on Strassen blocks.
        2. For the remaining strips, use schoolbook.
        
        A (4x5) = [A_core (4x4) | A_strip (4x1)]
        B (5x6) = [B_core (4x6); B_strip (1x6)]
        
        C = A_core * B_core + A_strip * B_strip
        
        A_core * B_core: (4x4) * (4x6)
        Partition B_core into [B_core_left (4x4) | B_core_right (4x2)]
        A_core * B_core = A_core * B_core_left + A_core * B_core_right
        
        A_core * B_core_left: (4x4) * (4x4) -> use Strassen-based rank 28
        A_core * B_core_right: (4x4) * (4x2) -> schoolbook rank 32
        A_strip * B_strip: (4x1) * (1x6) -> schoolbook rank 24
        
        Total: 28 + 32 + 24 = 84
        """
        # Build Strassen-based rank-28 decomposition for 4x4*4x4
        def strassen_4x4():
            """
            Rank-28 Strassen decomposition for 4x4 * 4x4.
            U (16, 28), V (16, 28), W (16, 28)
            U index: i*4+j, V index: j*4+k, W index: k*4+i
            """
            r = 28
            U = np.zeros((16, r))
            V = np.zeros((16, r))
            W = np.zeros((16, r))
            
            # 4 sub-blocks: (rb, cb) in {(0,0), (0,1), (1,0), (1,1)}
            # rb: row block of A (0 or 1), cb: col block of A / row block of B (0 or 1)
            # Each sub-block uses Strassen rank 7
            
            for rb in range(2):
                for cb in range(2):
                    base = (rb * 2 + cb) * 7
                    
                    # A block: rows 2*rb..2*rb+1, cols 2*cb..2*cb+1
                    # B block: rows 2*cb..2*cb+1, cols 2*cb..2*cb+1
                    # C block: rows 2*rb..2*rb+1, cols 2*cb..2*cb+1
                    
                    # Strassen M1 = (A11 + A21)(B11 + B22)
                    t = base + 0
                    U[(2*rb+1)*4 + (2*cb+1), t] = 1  # A[rb+1, cb+1]
                    U[(2*rb+1)*4 + (2*cb), t] = 1     # A[rb+1, cb]
                    V[(2*cb)*4 + (2*cb), t] = 1       # B[cb, cb]
                    V[(2*cb+1)*4 + (2*cb+1), t] = 1   # B[cb+1, cb+1]
                    W[(2*cb)*4 + (2*rb), t] = 1       # C[rb, cb]
                    W[(2*cb+1)*4 + (2*rb+1), t] = 1   # C[rb+1, cb+1]
                    
                    # M2 = A21(B12 - B11)
                    t = base + 1
                    U[(2*rb+1)*4 + (2*cb), t] = 1     # A[rb+1, cb]
                    V[(2*cb)*4 + (2*cb+1), t] = 1     # B[cb, cb+1]
                    V[(2*cb)*4 + (2*cb), t] = -1      # B[cb, cb]
                    W[(2*cb+1)*4 + (2*rb+1), t] = 1   # C[rb+1, cb+1]
                    W[(2*cb+1)*4 + (2*rb), t] = -1    # C[rb, cb+1]
                    
                    # M3 = (A11 + A12)B22
                    t = base + 2
                    U[(2*rb)*4 + (2*cb+1), t] = 1     # A[rb, cb+1]
                    U[(2*rb)*4 + (2*cb), t] = 1       # A[rb, cb]
                    V[(2*cb+1)*4 + (2*cb+1), t] = 1   # B[cb+1, cb+1]
                    W[(2*cb+1)*4 + (2*rb), t] = 1     # C[rb, cb+1]
                    W[(2*cb+1)*4 + (2*rb+1), t] = 1   # C[rb+1, cb+1]
                    
                    # M4 = A11(B21 - B22)
                    t = base + 3
                    U[(2*rb)*4 + (2*cb), t] = 1       # A[rb, cb]
                    V[(2*cb+1)*4 + (2*cb), t] = 1     # B[cb+1, cb]
                    V[(2*cb+1)*4 + (2*cb+1), t] = -1  # B[cb+1, cb+1]
                    W[(2*cb)*4 + (2*rb), t] = 1       # C[rb, cb]
                    W[(2*cb+1)*4 + (2*rb+1), t] = -1  # C[rb+1, cb+1]
                    
                    # M5 = (A11 + A21)B11
                    t = base + 4
                    U[(2*rb)*4 + (2*cb), t] = 1       # A[rb, cb]
                    U[(2*rb+1)*4 + (2*cb), t] = 1     # A[rb+1, cb]
                    V[(2*cb)*4 + (2*cb), t] = 1       # B[cb, cb]
                    W[(2*cb)*4 + (2*rb), t] = -1      # C[rb, cb]
                    W[(2*cb)*4 + (2*rb+1), t] = 1     # C[rb+1, cb]
                    
                    # M6 = (A21 - A11)(B11 + B12)
                    t = base + 5
                    U[(2*rb+1)*4 + (2*cb), t] = 1     # A[rb+1, cb]
                    U[(2*rb)*4 + (2*cb), t] = -1      # A[rb, cb]
                    V[(2*cb)*4 + (2*cb), t] = 1       # B[cb, cb]
                    V[(2*cb)*4 + (2*cb+1), t] = 1     # B[cb, cb+1]
                    # M6 is not directly used in standard Strassen output formulas
                    # for the 4 output blocks in the standard formulation.
                    # In the standard Strassen: C11=M1+M4-M5+M7, C12=M3+M7,
                    # C21=M2+M5, C22=M1-M2+M3-M4
                    # M6 doesn't appear! So we can skip it or set W to 0.
                    # Actually, M6 is needed for the full Strassen but the output
                    # only uses M1-M5 and M7. Let me check...
                    # 
                    # Standard Strassen:
                    # M1 = (A11+A21)(B11+B22)
                    # M2 = A21(B12-B11)
                    # M3 = (A11+A12)B22
                    # M4 = A11(B21-B22)
                    # M5 = (A11+A21)B11
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    #
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M7
                    # C21 = M2 + M5
                    # C22 = M1 - M2 + M3 - M4
                    #
                    # M6 is not used in the output! So we can omit it.
                    # But we need 7 multiplications. Let me re-examine.
                    # Actually, the standard Strassen uses all 7, but M6 might
                    # be used in some formulations. Let me just verify numerically.
                    # For now, set W for M6 to 0 (it won't contribute).
                    
                    # M7 = (A12 - A22)(B21 + B22)
                    t = base + 6
                    U[(2*rb)*4 + (2*cb+1), t] = 1     # A[rb, cb+1]
                    U[(2*rb+1)*4 + (2*cb+1), t] = -1  # A[rb+1, cb+1]
                    V[(2*cb+1)*4 + (2*cb), t] = 1     # B[cb+1, cb]
                    V[(2*cb+1)*4 + (2*cb+1), t] = 1   # B[cb+1, cb+1]
                    W[(2*cb)*4 + (2*rb), t] = 1       # C[rb, cb]
                    W[(2*cb+1)*4 + (2*rb+1), t] = -1  # C[rb+1, cb+1]
            
            return U, V, W
        
        # Build the full decomposition
        # A (4x5) = [A_core (4x4) | A_strip (4x1)]
        # B (5x6) = [B_core (4x6); B_strip (1x6)]
        # C = A_core * B_core + A_strip * B_strip
        
        # A_core * B_core: (4x4) * (4x6)
        # Partition B_core into [B_core_left (4x4) | B_core_right (4x2)]
        # A_core * B_core_left: (4x4) * (4x4) -> Strassen rank 28
        # A_core * B_core_right: (4x4) * (4x2) -> schoolbook rank 32
        # A_strip * B_strip: (4x1) * (1x6) -> schoolbook rank 24
        
        U_s, V_s, W_s = strassen_4x4()
        
        total_rank = 28 + 32 + 24  # 84
        U_full = np.zeros((nm, total_rank))
        V_full = np.zeros((mp, total_rank))
        W_full = np.zeros((pn, total_rank))
        
        # Part 1: A_core * B_core_left (4x4 * 4x4), Strassen rank 28
        # A_core rows 0-3, cols 0-3 -> U indices i*5+j for i in 0..3, j in 0..3
        # B_core_left rows 0-3, cols 0-3 -> V indices j*6+k for j in 0..3, k in 0..3
        # C rows 0-3, cols 0-3 -> W indices k*4+i for k in 0..3, i in 0..3
        
        for t in range(28):
            for a in range(16):  # U_s indices: i*4+j
                i, j = a // 4, a % 4
                if U_s[a, t] != 0:
                    U_full[i * 5 + j, t] += U_s[a, t]
            for b in range(16):  # V_s indices: j*4+k
                j, k = b // 4, b % 4
                if V_s[b, t] != 0:
                    V_full[j * 6 + k, t] += V_s[b, t]
            for c in range(16):  # W_s indices: k*4+i
                k, i = c // 4, c % 4
                if W_s[c, t] != 0:
                    W_full[k * 4 + i, t] += W_s[c, t]
        
        # Part 2: A_core * B_core_right (4x4 * 4x2), schoolbook rank 32
        # A_core: rows 0-3, cols 0-3
        # B_core_right: rows 0-3, cols 4-5 (cols 4 and 5 of B)
        # C: rows 0-3, cols 4-5
        
        offset = 28
        for i in range(4):
            for j in range(4):
                for k in range(2):  # k=0,1 correspond to B cols 4,5
                    t = offset + (i * 4 + j) * 2 + k
                    U_full[i * 5 + j, t] = 1.0
                    V_full[j * 6 + (4 + k), t] = 1.0
                    W_full[(4 + k) * 4 + i, t] = 1.0
        
        # Part 3: A_strip * B_strip (4x1 * 1x6), schoolbook rank 24
        # A_strip: rows 0-3, col 4
        # B_strip: row 4, cols 0-5
        # C: all rows 0-3, all cols 0-5
        
        offset = 60
        for i in range(4):
            for k in range(6):
                t = offset + i * 6 + k
                U_full[i * 5 + 4, t] = 1.0
                V_full[4 * 6 + k, t] = 1.0
                W_full[k * 4 + i, t] = 1.0
        
        return U_full, V_full, W_full

    # Try the block decomposition
    U_block, V_block, W_block = build_block_decomposition()
    err_block = verify_decomposition(U_block, V_block, W_block)
    
    if err_block < 1e-10:
        # Great, we have a valid rank-84 decomposition
        return {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
    
    # If the block decomposition doesn't work, fall back to schoolbook
    U_sb, V_sb, W_sb = schoolbook()
    err_sb = verify_decomposition(U_sb, V_sb, W_sb)
    
    if err_sb < 1e-10:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Should not reach here
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
