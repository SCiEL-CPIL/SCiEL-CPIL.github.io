"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = 4, 4, 6
    
    # Build 2x2x2 Strassen (rank 7)
    def strassen_222():
        """Return (U, V, W) for 2x2 * 2x2 -> 2x2, rank 7."""
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        # M0 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        # M1 = A11 * B01
        U[3,1] = 1
        V[1,1] = 1
        W[1,1] = 1; W[3,1] = -1
        # M2 = A10 * (B11-B01)
        U[2,2] = 1
        V[3,2] = 1; V[1,2] = -1
        W[2,2] = 1; W[3,2] = 1
        # M3 = (A00+A10)(B10)
        U[0,3] = 1; U[2,3] = 1
        V[2,3] = 1
        W[0,3] = -1; W[2,3] = 1
        # M4 = (A01-A10)(B00+B10)
        U[0,4] = 1; U[2,4] = -1
        V[0,4] = 1; V[2,4] = 1
        W[0,4] = 1
        # M5 = (A00-A11)(B11+B01)
        U[0,5] = 1; U[3,5] = -1
        V[3,5] = 1; V[1,5] = 1
        W[1,5] = 1
        # M6 = (A10-A00)(B00+A01)
        U[2,6] = 1; U[0,6] = -1
        V[0,6] = 1; V[1,6] = 1
        W[3,6] = 1
        return U, V, W
    
    # Build 2x2x1 schoolbook (rank 4)
    def school_221():
        """Return (U, V, W) for 2x2 * 2x1 -> 2x1, rank 4."""
        rank = 4
        U = np.zeros((4, rank))
        V = np.zeros((2, rank))
        W = np.zeros((2, rank))
        U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
        U[1,1] = 1; V[1,1] = 1; W[0,1] = 1
        U[2,2] = 1; V[0,2] = 1; W[1,2] = 1
        U[3,3] = 1; V[1,3] = 1; W[1,3] = 1
        return U, V, W
    
    # Build 4x4x4 using Strassen (rank 49)
    def build_444():
        """Build 4x4 * 4x4 -> 4x4 decomposition, rank 49."""
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        U2, V2, W2 = strassen_222()
        
        # 7 Strassen terms
        terms = [
            # (A_block, B_block, C_block, coeff)
            # M0 = (A00+A11)(B00+B11) -> C00 += M0, C01 += M0, C10 += M0, C11 += M0
            {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]},
            # M1 = A11 * B01 -> C01 += M1, C11 -= M1
            {'A': [(1,1,1)], 'B': [(0,1,1)], 'C': [(0,1,1), (1,1,-1)]},
            # M2 = A10 * (B11-B01) -> C10 += M2, C11 += M2
            {'A': [(1,0,1)], 'B': [(1,1,1), (0,1,-1)], 'C': [(1,0,1), (1,1,1)]},
            # M3 = (A00+A10) * B10 -> C00 -= M3, C10 += M3
            {'A': [(0,0,1), (1,0,1)], 'B': [(1,0,1)], 'C': [(0,0,-1), (1,0,1)]},
            # M4 = (A01-A10) * (B00+B10) -> C00 += M4
            {'A': [(0,1,1), (1,0,-1)], 'B': [(0,0,1), (1,0,1)], 'C': [(0,0,1)]},
            # M5 = (A00-A11) * (B11+B01) -> C01 += M5
            {'A': [(0,0,1), (1,1,-1)], 'B': [(1,1,1), (0,1,1)], 'C': [(0,1,1)]},
            # M6 = (A10-A00) * (B00+A01) -> C11 += M6
            {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]},
        ]
        
        # Block offsets for 2x2 sub-blocks in 4x4 matrix
        # A[i,j] where i,j in {0,1} for block (bi,bj)
        # Global index: (2*bi + i) * 4 + (2*bj + j) for A
        # For V: (2*bi + i) * 4 + (2*bj + j) where first is row of A col, second is row of B
        # Actually V index: j * 4 + k for 4x4 case
        # A block (ai, aj): rows (2*ai+i, 2*aj+j) for i,j in 0,1
        # B block (bi, bj): rows (2*bi+i, 2*bj+j) for i,j in 0,1
        # C block (ci, cj): rows (2*ci+i, 2*cj+j) for i,j in 0,1
        
        def a_idx(ai, aj, li, lj):
            return (2*ai + li) * 4 + (2*aj + lj)
        
        def v_idx(bi, bj, li, lj):
            return (2*bi + li) * 4 + (2*bj + lj)
        
        def w_idx(ci, cj, li, lj):
            return (2*ci + li) * 4 + (2*cj + lj)
        
        for t in range(7):
            base = t * 7
            for c in range(7):
                col = base + c
                # U part
                for (ai, aj, coeff) in terms[t]['A']:
                    for li in range(2):
                        for lj in range(2):
                            U[a_idx(ai, aj, li, lj), col] += U2[li*2+lj, c] * coeff
                
                # V part
                for (bi, bj, coeff) in terms[t]['B']:
                    for li in range(2):
                        for lj in range(2):
                            V[v_idx(bi, bj, li, lj), col] += V2[li*2+lj, c] * coeff
                
                # W part
                for (ci, cj, coeff) in terms[t]['C']:
                    for li in range(2):
                        for lj in range(2):
                            W[w_idx(ci, cj, li, lj), col] += W2[li*2+lj, c] * coeff
        
        return U, V, W
    
    # Build 4x4x2 (rank 28)
    def build_442():
        """Build 4x4 * 4x2 -> 4x2 decomposition, rank 28."""
        rank = 28
        U = np.zeros((16, rank))
        V = np.zeros((8, rank))  # 4*2
        W = np.zeros((8, rank))  # 2*4
        
        # Use 7 Strassen-like terms, each handling 2x2x2 sub-problems
        # A is 4x4, B is 4x2, C is 4x2
        # Split A into 2x2 blocks: A[(ai,aj)] for ai,aj in {0,1}
        # Split B into 2x2 blocks: B[(bi,bj)] for bi in {0,1}, bj in {0,1} (each 2x1)
        # Actually B is 4x2, so blocks are 2x1: B[(bi)] for bi in {0,1}, each 2x2? No.
        # B is 4 rows, 2 cols. Split into 2 row-blocks of 2x2? No, 2x1 each.
        # B[(bi, bj)] where bi in {0,1} (row block), bj in {0,1} (col within 2x1? No)
        
        # Let me think: B is 4x2. Row index j in 0..3, col index k in 0..1.
        # Split rows into 2 blocks of 2: B_block[bi] has rows 2*bi..2*bi+1, cols 0..1
        # So B_block[bi] is 2x2 for bi in {0,1}
        
        # A is 4x4. Split into 2x2 blocks: A_block[ai, aj] is 2x2
        # C is 4x2. Split into 2x2 blocks: C_block[ci] is 2x2 for ci in {0,1}
        
        # C_block[ci] = sum_{ai,bi} A_block[ai, ...] @ B_block[bi]... 
        # Actually C[2*ci+i, k] = sum_j A[2*ci+i, j] * B[j, k]
        
        # Let's use 7 multipliers. Each multiplier M_t is computed from combinations
        # of 2x2 blocks of A and 2x2 blocks of B, and contributes to C blocks.
        
        # Standard Strassen for 2x2*2x2 gives 7 multipliers. Here we have a "4x2" which
        # can be seen as two independent 2x2*2x2 multiplications? No, that's not right.
        
        # Alternative: 4x4 * 4x2 = (4x4 * 4x1) + (4x4 * 4x1) for the two columns.
        # 4x4*4x1 can be done with rank 16 (schoolbook) or lower.
        # But let's use the 7-term approach.
        
        # I'll use: for each of 7 Strassen terms, the "B" side uses both columns.
        # Term structure similar to 4x4x4 but B only has 2 columns.
        
        U2, V2, W2 = strassen_222()
        
        terms = [
            {'A': [(0,0,1), (1,1,1)], 'B': [(0,1), (1,1)], 'C': [(0,1), (0,2), (1,1), (1,2)]},
            {'A': [(1,1,1)], 'B': [(0,2)], 'C': [(0,2), (1,-1)]},
            {'A': [(1,0,1)], 'B': [(1,2), (0,-1)], 'C': [(1,1), (1,2)]},
            {'A': [(0,0,1), (1,0,1)], 'B': [(1,1)], 'C': [(0,-1), (1,1)]},
            {'A': [(0,1,1), (1,0,-1)], 'B': [(0,1), (1,1)], 'C': [(0,1)]},
            {'A': [(0,0,1), (1,1,-1)], 'B': [(1,2), (0,2)], 'C': [(0,2)]},
            {'A': [(1,0,1), (0,0,-1)], 'B': [(0,1), (0,2)], 'C': [(1,2)]},
        ]
        
        # Hmm, this is getting confused. Let me use a different approach.
        # For 4x4x2, I'll just use schoolbook for each column: rank 16*2 = 32.
        # But we want 28. Let me use a hybrid.
        
        # Actually, let me just verify if rank 28 is achievable and construct it properly.
        # Known: 4x4x2 has rank at most 28 using Strassen-like methods.
        
        # Simpler: use 7 terms where each term multiplies 2x2 submatrices.
        # A blocks: A00, A01, A10, A11 (each 2x2)
        # B blocks: B0, B1 (each 2x2, where B0 is rows 0-1 cols 0-1, B1 is rows 2-3 cols 0-1)
        # C blocks: C0, C1 (each 2x2)
        
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # This is a 2x2 block matrix multiplication where A is 2x2 blocks, B is 2x1 blocks.
        # We can apply Strassen to this: treat it as (2x2) @ (2x1) -> (2x1) at block level.
        
        # Block-level: X (2x2) @ Y (2x1) -> Z (2x1)
        # X = [[A00, A01], [A10, A11]], Y = [[B0], [B1]], Z = [[C0], [C1]]
        
        # Strassen for 2x2 * 2x1:
        # M0 = (X00 + X11) * Y0 = (A00 + A11) * B0
        # M1 = X11 * Y0 = A11 * B0  
        # Wait, that's not standard. For 2x2 * 2x1:
        # Z0 = X00*Y0 + X01*Y1
        # Z1 = X10*Y0 + X11*Y1
        # Rank 4 schoolbook is optimal for 2x2x1.
        
        # So at block level, rank 4 means 4 multipliers, each being 2x2*2x2 = rank 7.
        # Total rank = 4 * 7 = 28. 
        
        # The 4 multipliers:
        # M0 = A00 * B0
        # M1 = A01 * B1
        # M2 = A10 * B0
        # M3 = A11 * B1
        
        # C0 = M0 + M1
        # C1 = M2 + M3
        
        U442 = np.zeros((16, 28))
        V442 = np.zeros((8, 28))
        W442 = np.zeros((8, 28))
        
        U2, V2, W2 = strassen_222()
        
        # 4 multipliers, each rank 7
        # Multiplier 0: A00 * B0 -> contributes to C0
        # Multiplier 1: A01 * B1 -> contributes to C0
        # Multiplier 2: A10 * B0 -> contributes to C1
        # Multiplier 3: A11 * B1 -> contributes to C1
        
        def a_idx_442(ai, aj, li, lj):
            return (2*ai + li) * 4 + (2*aj + lj)
        
        def v_idx_442(bi, k, li, lj):
            # B is 4x2, block bi is rows 2*bi..2*bi+1
            # V index: j * 2 + k where j is row, k is col
            return (2*bi + li) * 2 + lj  # lj is 0 or 1 (column)
        
        def w_idx_442(ci, li, lj):
            # C is 4x2, block ci is rows 2*ci..2*ci+1
            # W index: k * 4 + i where k is col, i is row
            return lj * 4 + (2*ci + li)
        
        mult_defs = [
            {'A': (0,0), 'B': 0, 'C': 0},  # A00 * B0 -> C0
            {'A': (0,1), 'B': 1, 'C': 0},  # A01 * B1 -> C0
            {'A': (1,0), 'B': 0, 'C': 1},  # A10 * B0 -> C1
            {'A': (1,1), 'B': 1, 'C': 1},  # A11 * B1 -> C1
        ]
        
        for mult in range(4):
            base = mult * 7
            ai, aj = mult_defs[mult]['A']
            bi = mult_defs[mult]['B']
            ci = mult_defs[mult]['C']
            
            for c in range(7):
                col = base + c
                for li in range(2):
                    for lj in range(2):
                        U442[a_idx_442(ai, aj, li, lj), col] += U2[li*2+lj, c]
                        V442[v_idx_442(bi, 0, li, lj), col] += V2[li*2+lj, c]
                        W442[w_idx_442(ci, li, lj), col] += W2[li*2+lj, c]
        
        return U442, V442, W442
    
    # Build the full 4x4x6 decomposition
    U_444, V_444, W_444 = build_444()  # 16x49, 16x49, 16x49
    U_442, V_442, W_442 = build_442()  # 16x28, 8x28, 8x28
    
    total_rank = 49 + 28  # 77
    
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))  # 4*6
    W = np.zeros((24, total_rank))  # 6*4
    
    # Copy 4x4x4 part (columns 0-3 of B)
    U[:, 0:49] = U_444
    # V_444 is 16x49, but we need to place it correctly in V (24x49)
    # V index: j * 6 + k. For 4x4x4, k in 0..3.
    # V_444 index: j * 4 + k for k in 0..3
    # So V_444[j*4+k] maps to V[j*6+k]
    for j in range(4):
        for k in range(4):
            V[j*6 + k, 0:49] = V_444[j*4 + k, 0:49]
    
    # W_444 is 16x49. W index: k * 4 + i for k in 0..3.
    # In full W, index is k * 4 + i (same since n=4)
    W[0:16, 0:49] = W_444
    
    # Copy 4x4x2 part (columns 4-5 of B)
    U[:, 49:77] = U_442
    # V_442 is 8x28. V_442 index: j * 2 + k for k in 0..1, j in 0..3
    # In full V, index is j * 6 + (4+k) for k in 0..1
    for j in range(4):
        for k in range(2):
            V[j*6 + 4 + k, 49:77] = V_442[j*2 + k, 0:28]
    
    # W_442 is 8x28. W_442 index: k * 4 + i for k in 0..1
    # In full W, index is (4+k) * 4 + i for k in 0..1
    for k in range(2):
        for i in range(4):
            W[(4+k)*4 + i, 49:77] = W_442[k*4 + i, 0:28]
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(prod - 1.0) > 1e-9:
                        return False, (i, j, k, prod)
        return True, None
    
    valid, info = verify(U, V, W, n, m, p)
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Convert to lists with half-integer precision
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if abs(val * 2 - round(val * 2)) < 1e-9:
                    new_row.append(round(val * 2) / 2)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
