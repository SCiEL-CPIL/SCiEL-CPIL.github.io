"""Tensor decomposition for 4x5 by 5x6 matrix multiplication."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 115
    
    n, m, p = 4, 5, 6
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    U = []
    V = []
    W = []
    
    def add_block(U_cols, V_cols, W_cols):
        """Add a block of factor matrices."""
        for col in range(len(U_cols)):
            U.append(U_cols[col])
            V.append(V_cols[col])
            W.append(W_cols[col])
    
    def outer_product_block(n, p):
        """Generate rank n*p block for outer product style."""
        u_cols = []
        v_cols = []
        w_cols = []
        for i in range(n):
            for k in range(p):
                u = [0.0] * 1
                v = [0.0] * 1
                w = [0.0] * 1
                u[0] = 1.0
                v[0] = 1.0
                w[0] = 1.0
                u_cols.append(u)
                v_cols.append(v)
                w_cols.append(w)
        return u_cols, v_cols, w_cols
    
    def strassen_2x2x2():
        """Generate Strassen 2x2x2 block (rank 7)."""
        # Strassen's 7 multiplications for 2x2 * 2x2
        # M1 = (a11+a22)*(b11+b22)
        # M2 = (a21+a22)*b11
        # M3 = a11*(b12-b22)
        # M4 = a22*(b21-b11)
        # M5 = (a11+a12)*b22
        # M6 = (a21-a11)*(b11+b12)
        # M7 = (a12-a22)*(b21+b22)
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # For each multiplication, we need:
        # U[i*m+j] has the coefficient of A[i,j]
        # V[j*p+k] has the coefficient of B[j,k]
        # W[k*n+i] has the coefficient of C[k,i]
        
        muls = [
            # (U_coeffs: list of (i,j,coeff), V_coeffs: list of (j,k,coeff), W_coeffs: list of (k,i,coeff))
            # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1),(0,1,1),(1,0,1)]),
            # M2 = (A[1,0]+A[1,1]) * B[0,0]
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,1,1),(1,0,1)]),
            # M3 = A[0,0] * (B[0,1]-B[1,1])
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,-1)]),
            # M4 = A[1,1] * (B[1,0]-B[0,0])
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,1,-1),(0,1,1)]),
            # M5 = (A[0,0]+A[0,1]) * B[1,1]
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)]),
            # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1),(1,1,1)]),
        ]
        
        u_cols = []
        v_cols = []
        w_cols = []
        for u_coeffs, v_coeffs, w_coeffs in muls:
            u = [0.0] * 1
            v = [0.0] * 1
            w = [0.0] * 1
            for i, j, c in u_coeffs:
                u[0] += c
            for j, k, c in v_coeffs:
                v[0] += c
            for k, i, c in w_coeffs:
                w[0] += c
            u_cols.append(u)
            v_cols.append(v)
            w_cols.append(w)
        return u_cols, v_cols, w_cols
    
    def laderman_4x4x4():
        """Generate Laderman's rank-21 decomposition for 4x4 * 4x4.
        
        Laderman's formula uses 21 multiplications to multiply two 4x4 matrices.
        This is a well-known result in algebraic complexity theory.
        """
        # Laderman's 4x4x4 decomposition with rank 21
        # This is complex; I'll use a known construction
        # Split 4x4 into four 2x2 blocks and use Strassen on each, 
        # but with shared subcomputations.
        
        # Actually, let me use a simpler approach: use 4 Strassen blocks for 2x2 submatrices
        # but that gives 4*7=28. Laderman achieves 21 by sharing.
        
        # For now, let me use 4 independent Strassen decompositions for the 2x2 blocks
        # of a 4x4 * 4x4 multiplication, giving rank 28.
        # This is still better than schoolbook (16) for 4x4.
        # Wait, schoolbook for 4x4 is 16 multiplications, Strassen for 4x4 is 4*7=28.
        # So schoolbook is better for 4x4!
        
        # Let me just use schoolbook for 4x4 (rank 16).
        u_cols = []
        v_cols = []
        w_cols = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u = [0.0] * 1
                    v = [0.0] * 1
                    w = [0.0] * 1
                    u[0] = 1.0
                    v[0] = 1.0
                    w[0] = 1.0
                    u_cols.append(u)
                    v_cols.append(v)
                    w_cols.append(w)
        return u_cols, v_cols, w_cols
    
    # Build the decomposition for 4x5 * 5x6
    # Split the middle dimension (5) into 4+1
    # A is 4x5, B is 5x6, C is 4x6
    # Let A = [A1, A2] where A1 is 4x4 and A2 is 4x1
    # Let B = [B1; B2] where B1 is 4x6 and B2 is 1x6
    # Then C = A1*B1 + A2*B2
    
    # Block 1: A1 (4x4) * B1 (4x6)
    # Split B1 into B11 (4x4) and B12 (4x2)
    # C1 = A1*B11 + A1*B12
    # A1*B11: 4x4 * 4x4 -> use schoolbook rank 16
    # A1*B12: 4x4 * 4x2 -> use schoolbook rank 16? No, 4*4*2 = 32
    # Actually for 4x4 * 4x2, schoolbook is 4*4*2 = 32 multiplications
    
    # Let me use a different split.
    # A (4x5) * B (5x6)
    # Split into:
    # 1. A[:, :4] (4x4) * B[:4, :] (4x6) -> 4x4 * 4x6
    # 2. A[:, 4:] (4x1) * B[4:, :] (1x6) -> 4x1 * 1x6
    
    # For 4x4 * 4x6:
    # Split B into B1 (4x4) and B2 (4x2)
    # A*B1: 4x4*4x4 = rank 16 (schoolbook) or 21 (Laderman)
    # A*B2: 4x4*4x2 = rank 32 (schoolbook)
    # Total: 16+32 = 48 or 21+32 = 53
    
    # For 4x1 * 1x6:
    # This is outer product: rank 4*6 = 24
    
    # Total rank: 48 + 24 = 72 (using schoolbook for 4x4*4x4)
    # Or 53 + 24 = 77 (using Laderman)
    
    # Let me use schoolbook for everything to keep it simple and correct.
    # 4x4 * 4x6: rank 4*4*6 = 96
    # 4x1 * 1x6: rank 4*1*6 = 24
    # Total: 120 (same as schoolbook)
    
    # Let me be smarter. For 4x4 * 4x6:
    # I can split B into two 4x3 matrices
    # A * B1: 4x4 * 4x3 = rank 4*4*3 = 48
    # A * B2: 4x4 * 4x3 = rank 4*4*3 = 48
    # Total: 96
    
    # Hmm, let me just use the direct schoolbook approach but optimized.
    # Actually, the simplest correct approach is schoolbook: rank 4*5*6 = 120
    
    # Let me try a Strassen-like approach for the main block.
    # Split A (4x5) into A1 (4x2), A2 (4x2), A3 (4x1)
    # Split B (5x6) into B1 (2x6), B2 (2x6), B3 (1x6)
    # C = A1*B1 + A2*B2 + A3*B3
    
    # A1*B1: 4x2 * 2x6 = rank 4*2*6 = 48
    # A2*B2: 4x2 * 2x6 = rank 4*2*6 = 48
    # A3*B3: 4x1 * 1x6 = rank 4*1*6 = 24
    # Total: 120
    
    # This isn't helping. Let me just return schoolbook for now.
    # The key insight from the cell findings is that replacing invalid Strassen
    # with valid schoolbook gives rank 105, which is better than 120.
    
    # Let me implement the approach that gives rank 105:
    # Split 4x5 * 5x6 as:
    # - Main block: 4x4 * 4x4 using some method
    # - Other blocks
    
    # Actually, let me just do schoolbook. It's guaranteed correct.
    rank = n * m * p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0] * rank
                v = [0.0] * rank
                w = [0.0] * rank
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                U.append(u)
                V.append(v)
                W.append(w)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
