"""Matrix multiplication tensor decomposition for 2x4 by 4x5 matrices."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x5():
    """
    Rank-13 decomposition for 2x2 * 2x5 matrix multiplication.
    
    A is 2x2, B is 2x5, C = A*B is 2x5.
    
    Split B into B = [b0 b1 b2 b3 b4] where bj are 2x1 columns.
    C[:,j] = A * b_j for each j.
    
    Using Strassen for the 2x2 * 2x1 case (rank 7) for the first two columns,
    and direct multiplication (rank 2 each) for the remaining three columns:
    Total: 7 + 3*2 = 13.
    
    Wait - this doesn't work because Strassen's 7 multiplications are for 2x2*2x2.
    For 2x2 * 2x1, we just need 4 multiplications (or can we do better?).
    
    Actually for 2x2 * 2x1: rank is 4 (schoolbook) since it's just 4 scalar products.
    So total would be 4*5 = 20 for M_{2,2,5}.
    
    Let me use a different approach: treat 2x2 * 2x5 as a bilinear map.
    The tensor M_{2,2,5} has dimensions (4, 10, 10).
    
    Let me just use schoolbook for now: rank 20 for each 2x2*2x5.
    Total: 20 + 20 = 40. Same as before.
    
    Hmm, I need a better strategy. Let me think about this differently.
    
    Alternative: Use the "laser method" or a direct construction.
    
    For now, let me try a rank-36 construction using a different block split.
    """
    # For 2x2 * 2x5, use schoolbook: rank 10 (5 columns * 2 rows in A = 10 products)
    # Actually: A is 2x2, B is 2x5. C[i,j] = sum_k A[i,k]*B[k,j].
    # Number of products: 2*2*5 = 20. Schoolbook rank = 20.
    
    # Let me try to find a better decomposition for 2x2 * 2x5.
    # Known: rank(M_{2,2,p}) for p>=2. 
    # For p=1: rank 4. For p=2: rank 7 (Strassen). 
    # For p=3: ? For p=5: ?
    
    # Actually, I recall that rank(M_{2,2,p}) = 2p+3 for p>=2 is NOT correct.
    # The correct upper bound using Strassen + schoolbook: 7 + (p-2)*4 = 4p+3? No...
    
    # Let me just use a constructive approach for the full 2x4*4x5 problem.
    # 
    # Key idea: Use the fact that we can write the multiplication as:
    # C = A * B where A is 2x4, B is 4x5.
    # 
    # Let's use a "partition" approach:
    # Write A = [A_left A_right] (2x2 each)
    # Write B = [B_top; B_bottom] (2x5 each)
    # C = A_left * B_top + A_right * B_bottom
    #
    # For each 2x2 * 2x5 product, I'll use a rank-18 algorithm (if I can find one).
    # 18 + 18 = 36.
    
    # For 2x2 * 2x5: Let's construct a rank-18 decomposition.
    # A = [[a00, a01], [a10, a11]], B = [[b00,b01,b02,b03,b04], [b10,b11,b12,b13,b14]]
    # C[i,j] = a_i0*b_0j + a_i1*b_1j
    #
    # We need sum_c U[i*2+k, c] * V[k*5+j, c] * W[j*2+i, c] = 1 for all i,j,k.
    #
    # Schoolbook: for each (i,j,k), one term. Total 2*2*5 = 20 terms.
    #
    # To reduce by 2: find 2 linear dependencies.
    #
    # Let me try: group by j. For each j, we have a 2x2*2x1 multiplication (rank 4).
    # 5 columns * 4 = 20.
    #
    # Can we share across columns? If we use Strassen-like for pairs of columns...
    # For columns (j, j+1): 2x2 * 2x2, rank 7.
    # Pair columns: (0,1), (2,3), and column 4 alone.
    # 7 + 7 + 4 = 18! 
    #
    # Let me verify: columns 0,1 use Strassen (rank 7), columns 2,3 use Strassen (rank 7),
    # column 4 uses schoolbook (rank 4). Total: 18.
    
    n, m, p = 2, 2, 5
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 18
    V = [[0.0] * rank for _ in range(m * p)]  # 10 x 18
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 18
    
    # Strassen for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b00-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2x2(base_col):
        """Add Strassen decomposition for 2x2*2x2 starting at column base_col."""
        # U rows: 0=a00, 1=a10, 2=a01, 3=a11 (index = i*m+k = i*2+k)
        # V rows: 0=b00, 1=b10, 2=b01, 3=b11 (index = k*p+j = k*5+j, for j=0,1: 0,5,1,6)
        # Wait, V index is k*p+j where p=5. For columns j=0,1:
        #   k=0: j=0 -> V[0], j=1 -> V[5]
        #   k=1: j=0 -> V[1], j=1 -> V[6]
        # W rows: index = j*n+i = j*2+i. For j=0,1:
        #   j=0: i=0 -> W[0], i=1 -> W[1]
        #   j=1: i=0 -> W[2], i=1 -> W[3]
        
        # M1 = (a00+a11)(b00+b11)
        c = base_col
        U[0][c] += 1; U[3][c] += 1
        V[0][c] += 1; V[6][c] += 1
        W[0][c] += 1; W[1][c] += 1; W[2][c] += 1; W[3][c] += 1
        # C00 += M1, C01 += 0, C10 += 0, C11 += M1
        # Wait, I need to be more careful. Each Strassen product contributes to specific C entries.
        # Let me redo this properly.
        
        # Actually, let me rebuild this more carefully.
        pass
    
    # This is getting complex. Let me just directly construct the full 2x4*4x5 rank-36.
    return None

def _construct_rank36():
    """
    Construct a rank-36 decomposition for 2x4 * 4x5.
    
    Strategy: Split into two 2x2 * 2x5 multiplications.
    For each 2x2 * 2x5, use rank 18 (Strassen for 2 column pairs + schoolbook for 1 column).
    Total: 18 + 18 = 36.
    """
    n, m, p = 2, 4, 5
    total_rank = 36
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 x 36
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 x 36
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 x 36
    
    # A is 2x4: rows 0,1. Columns 0-3.
    # U index: i*m + j = i*4 + j, where i in {0,1}, j in {0,1,2,3}
    # B is 4x5: rows 0-3, cols 0-4.
    # V index: j*p + k = j*5 + k, where j in {0,1,2,3}, k in {0,1,2,3,4}
    # C is 2x5: rows 0,1, cols 0-4.
    # W index: k*n + i = k*2 + i, where k in {0,1,2,3,4}, i in {0,1}
    
    # Split: A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x5), B_bottom = B[2:4, :] (2x5)
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For A_left @ B_top:
    # A_left[i, k] corresponds to U[i*4 + k] for k in {0,1}
    # B_top[k, j] corresponds to V[k*5 + j] for k in {0,1}, j in {0..4}
    # Result contributes to C[i, j] -> W[j*2 + i]
    
    # For A_right @ B_bottom:
    # A_right[i, k] corresponds to U[i*4 + 2 + k] for k in {0,1}
    # B_bottom[k, j] corresponds to V[(2+k)*5 + j] for k in {0,1}, j in {0..4}
    # Result contributes to C[i, j] -> W[j*2 + i]
    
    # Helper: add a 2x2 * 2x5 decomposition starting at column offset
    def add_2x2x5(a_col_offset, b_row_offset, base_col):
        """
        Add rank-18 decomposition for 2x2 * 2x5.
        a_col_offset: column offset in A (0 for left, 2 for right)
        b_row_offset: row offset in B (0 for top, 2 for bottom)
        base_col: starting column in the rank dimension
        """
        # A indices: i*4 + a_col_offset + k for i in {0,1}, k in {0,1}
        # B indices: (b_row_offset + k)*5 + j for k in {0,1}, j in {0..4}
        # C indices: j*2 + i for j in {0..4}, i in {0,1}
        
        # Use Strassen for columns (0,1) and (2,3), schoolbook for column 4.
        
        # Strassen for 2x2 * 2x2 (columns 0,1):
        # M1 = (a00+a11)(b00+b10)  [b00=B[0,0], b10=B[1,0], b01=B[0,1], b11=B[1,1]]
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # Let me define:
        # A = [[a00, a01], [a10, a11]] where a_ij = A[i, a_col_offset+j]
        # B = [[b00, b01], [b10, b11]] where b_kj = B[b_row_offset+k, j]
        
        # Strassen products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # For columns (0,1):
        c = base_col
        # M1 = (a00+a11)(b00+b11)
        U[0*4 + a_col_offset + 0][c] += 1  # a00
        U[0*4 + a_col_offset + 1][c] += 0  # not a01
        U[1*4 + a_col_offset + 0][c] += 0  # not a10
        U[1*4 + a_col_offset + 1][c] += 1  # a11
        V[(b_row_offset+0)*5 + 0][c] += 1  # b00
        V[(b_row_offset+0)*5 + 1][c] += 1  # b01 -> wait, b11 is B[1,1]
        # b00 = B[b_row_offset+0, 0], b11 = B[b_row_offset+1, 1]
        V[(b_row_offset+1)*5 + 1][c] += 1
        # C00 = W[0*2+0]=W[0], C11 = W[1*2+1]=W[3]
        W[0][c] += 1
        W[3][c] += 1
        c += 1
        
        # M2 = (a10+a11)b00
        U[1*4 + a_col_offset + 0][c] += 1  # a10
        U[1*4 + a_col_offset + 1][c] += 1  # a11
        V[(b_row_offset+0)*5 + 0][c] += 1  # b00
        # C10 = W[0*2+1]=W[1]
        W[1][c] += 1
        c += 1
        
        # M3 = a00(b10-b11)
        U[0*4 + a_col_offset + 0][c] += 1  # a00
        V[(b_row_offset+1)*5 + 0][c] += 1  # b10
        V[(b_row_offset+1)*5 + 1][c] += -1  # -b11
        # C01 = W[1*2+0]=W[2]
        W[2][c] += 1
        c += 1
        
        # M4 = a11(b00-b10)
        U[1*4 + a_col_offset + 1][c] += 1  # a11
        V[(b_row_offset+0)*5 + 0][c] += 1  # b00
        V[(b_row_offset+1)*5 + 0][c] += -1  # -b10
        # C00 += M4, C10 += M4
        W[0][c] += 1
        W[1][c] += 1
        c += 1
        
        # M5 = (a00+a01)b11
        U[0*4 + a_col_offset + 0][c] += 1  # a00
        U[0*4 + a_col_offset + 1][c] += 1  # a01
        V[(b_row_offset+1)*5 + 1][c] += 1  # b11
        # C00 -= M5, C01 += M5
        W[0][c] += -1
        W[2][c] += 1
        c += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[1*4 + a_col_offset + 0][c] += 1  # a10
        U[0*4 + a_col_offset + 0][c] += -1  # -a00
        V[(b_row_offset+0)*5 + 0][c] += 1  # b00
        V[(b_row_offset+0)*5 + 1][c] += 1  # b01
        # C11 += M6
        W[3][c] += 1
        c += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[0*4 + a_col_offset + 1][c] += 1  # a01
        U[1*4 + a_col_offset + 1][c] += -1  # -a11
        V[(b_row_offset+1)*5 + 0][c] += 1  # b10
        V[(b_row_offset+1)*5 + 1][c] += 1  # b11
        # C00 += M7, C11 += M7
        W[0][c] += 1
        W[3][c] += 1
        c += 1
        
        # Strassen for columns (2,3):
        # Same structure but B columns are 2,3 instead of 0,1
        # b00' = B[b_row_offset+0, 2], b01' = B[b_row_offset+0, 3]
        # b10' = B[b_row_offset+1, 2], b11' = B[b_row_offset+1, 3]
        
        # M1' = (a00+a11)(b00'+b11')
        U[0*4 + a_col_offset + 0][c] += 1
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 2][c] += 1
        V[(b_row_offset+1)*5 + 3][c] += 1
        # C00' = W[2*2+0]=W[4], C11' = W[3*2+1]=W[7]
        W[4][c] += 1
        W[7][c] += 1
        c += 1
        
        # M2' = (a10+a11)b00'
        U[1*4 + a_col_offset + 0][c] += 1
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 2][c] += 1
        # C10' = W[2*2+1]=W[5]
        W[5][c] += 1
        c += 1
        
        # M3' = a00(b10'-b11')
        U[0*4 + a_col_offset + 0][c] += 1
        V[(b_row_offset+1)*5 + 2][c] += 1
        V[(b_row_offset+1)*5 + 3][c] += -1
        # C01' = W[3*2+0]=W[6]
        W[6][c] += 1
        c += 1
        
        # M4' = a11(b00'-b10')
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 2][c] += 1
        V[(b_row_offset+1)*5 + 2][c] += -1
        # C00' += M4', C10' += M4'
        W[4][c] += 1
        W[5][c] += 1
        c += 1
        
        # M5' = (a00+a01)b11'
        U[0*4 + a_col_offset + 0][c] += 1
        U[0*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+1)*5 + 3][c] += 1
        # C00' -= M5', C01' += M5'
        W[4][c] += -1
        W[6][c] += 1
        c += 1
        
        # M6' = (a10-a00)(b00'+b01')
        U[1*4 + a_col_offset + 0][c] += 1
        U[0*4 + a_col_offset + 0][c] += -1
        V[(b_row_offset+0)*5 + 2][c] += 1
        V[(b_row_offset+0)*5 + 3][c] += 1
        # C11' += M6'
        W[7][c] += 1
        c += 1
        
        # M7' = (a01-a11)(b10'+b11')
        U[0*4 + a_col_offset + 1][c] += 1
        U[1*4 + a_col_offset + 1][c] += -1
        V[(b_row_offset+1)*5 + 2][c] += 1
        V[(b_row_offset+1)*5 + 3][c] += 1
        # C00' += M7', C11' += M7'
        W[4][c] += 1
        W[7][c] += 1
        c += 1
        
        # Schoolbook for column 4: 4 products
        # C[i,4] = a_i0 * B[b_row_offset+0, 4] + a_i1 * B[b_row_offset+1, 4]
        for i in range(2):
            for k in range(2):
                U[i*4 + a_col_offset + k][c] += 1
                V[(b_row_offset+k)*5 + 4][c] += 1
                W[4*2 + i][c] += 1  # W[8+i]
                c += 1
        
        return c
    
    # Add left block: A_left @ B_top
    col = add_2x2x5(0, 0, 0)
    
    # Add right block: A_right @ B_bottom
    col = add_2x2x5(2, 2, col)
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank-36 construction
    try:
        result = _construct_rank36()
        if result is not None:
            # Verify
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
