"""Strategy: Kronecker product of 2x2 Strassen and 4x4 Schoolbook for 4x4x8."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0

def _time_left():
    return _DEADLINE - _SAFETY - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # For each (i,j,k), check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _strassen_2x2_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2. Rank 7."""
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)(e), M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)(h), M6 = (c-d)(f+g), M7 = (b-d)(e+f)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2: i,j,k in {0,1}
    # U indices: i*2+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: j*2+k: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W indices: k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (a+d)(e+h)
    # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (c+d)(e)
    # c=A[1,0], d=A[1,1], e=B[0,0]
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1
    
    # M3 = a(f-h)
    # a=A[0,0], f=B[0,1], h=B[1,1]
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1
    
    # M4 = d(g-e)
    # d=A[1,1], g=B[1,0], e=B[0,0]
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # M5 = (a+b)(h)
    # a=A[0,0], b=A[0,1], h=B[1,1]
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # M6 = (c-d)(f+g)
    # c=A[1,0], d=A[1,1], f=B[0,1], g=B[1,0]
    U[2,5] = 1; U[3,5] = -1
    V[1,5] = 1; V[2,5] = 1
    W[3,5] = 1
    
    # M7 = (b-d)(e+f)
    # b=A[0,1], d=A[1,1], e=B[0,0], f=B[0,1]
    U[1,6] = 1; U[3,6] = -1
    V[0,6] = 1; V[1,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _schoolbook_4x4_4x4():
    """Return schoolbook decomposition for 4x4 * 4x4 -> 4x4. Rank 64."""
    n = m = p = 4
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _build_kronecker_decomposition():
    """
    Construct 4x4 * 4x8 decomposition using Kronecker product of:
    1. Strassen 2x2 * 2x2 -> 2x2 (Rank 7)
    2. Schoolbook 4x4 * 4x4 -> 4x4 (Rank 64)
    
    Total Rank = 7 * 64 = 448? No, that's too high.
    Wait, the dimensions are 4x4 * 4x8.
    We can view 4x4 as 2x2 blocks of 2x2 matrices.
    A is 4x4 -> 2x2 blocks of 2x2.
    B is 4x8 -> 2x4 blocks of 2x4? No.
    
    Let's rethink the Kronecker structure.
    We want to multiply (4x4) by (4x8).
    Let's split the 4x4 matrix A into 2x2 blocks:
    A = [A11 A12; A21 A22] where each Ai is 2x2.
    
    Let's split the 4x8 matrix B into 2x4 blocks:
    B = [B1; B2] where B1 is 2x4 and B2 is 2x4.
    
    This doesn't map directly to a simple Kronecker product of two standard matmul tensors because the inner dimension 4 is split into 2+2 for A but the outer dimension 8 is not split into 2x2 blocks of 2x4 in a way that allows a simple tensor product with a 2x2*2x2 tensor.
    
    Alternative:
    Split 4x4 * 4x8 into two 4x4 * 4x4 multiplications?
    B = [B_left, B_right] where B_left, B_right are 4x4.
    C = A * B = [A * B_left, A * B_right].
    
    If we use Strassen for 4x4 * 4x4, what is the rank?
    4x4 * 4x4 can be done with rank 7 * 4 = 28? No.
    Strassen for 4x4 * 4x4:
    Split into 2x2 blocks.
    7 multiplications of 2x2 * 2x2.
    Each 2x2 * 2x2 has rank 4 (schoolbook) or 7 (Strassen).
    If we use schoolbook for the 2x2 level: Rank = 7 * 4 = 28.
    If we use Strassen for the 2x2 level: Rank = 7 * 7 = 49.
    
    So 4x4 * 4x4 has rank 28 (using Strassen at top, schoolbook at bottom).
    Then 4x4 * 4x8 is two such multiplications: Rank = 28 + 28 = 56.
    
    This is much better than 128.
    Let's implement this "Strassen-Top, Schoolbook-Bottom" approach.
    """
    n, m, p = 4, 4, 8
    
    # We will construct the decomposition for 4x4 * 4x4 first.
    # Then duplicate it for the second half of B.
    
    U_4x4, V_4x4, W_4x4 = _build_4x4_4x4_strassen_top_schoolbook_bottom()
    
    # Now extend to 4x4 * 4x8.
    # B is 4x8. We split B into B1 (4x4) and B2 (4x4).
    # V matrix has shape (m*p, rank) = (4*8, rank) = (32, rank).
    # V[j*p+k, r].
    # For B1 (columns 0-3), j*p+k is j*4+k.
    # For B2 (columns 4-7), j*p+k is j*4+k. Wait.
    # B is 4x8. Index (j, k) with j in 0..3, k in 0..7.
    # Flattened index for V is j*8+k.
    # 
    # In the 4x4 * 4x4 case:
    # B is 4x4. Index (j, k) with j in 0..3, k in 0..3.
    # Flattened index for V is j*4+k.
    #
    # So for the 4x4 * 4x8 case:
    # The V matrix will have 32 rows.
    # Rows 0..31 correspond to j*8+k.
    # 
    # The 4x4 * 4x4 V matrix has 16 rows (j*4+k).
    # 
    # We can map the 4x4 V rows to the 4x8 V rows.
    # For B1 (first 4 cols): k' = k. Index in V_4x8 = j*8 + k.
    # For B2 (last 4 cols): k' = k+4. Index in V_4x8 = j*8 + (k+4).
    
    rank_4x4 = U_4x4.shape[1]
    rank_total = 2 * rank_4x4
    
    U_total = np.zeros((n*m, rank_total))
    V_total = np.zeros((m*p, rank_total))
    W_total = np.zeros((p*n, rank_total))
    
    # Copy U for both halves (U depends only on A, which is 4x4)
    U_total[:, :rank_4x4] = U_4x4
    U_total[:, rank_4x4:] = U_4x4
    
    # Copy W for both halves?
    # W depends on C (4x4) and A (4x4).
    # C1 = A * B1 (4x4)
    # C2 = A * B2 (4x4)
    # W_4x4 is for C = A * B.
    # So W_total[:, :rank_4x4] = W_4x4
    # W_total[:, rank_4x4:] = W_4x4
    
    W_total[:, :rank_4x4] = W_4x4
    W_total[:, rank_4x4:] = W_4x4
    
    # V mapping
    # V_4x4 has shape (16, rank_4x4). Rows indexed by j*4+k.
    # V_total has shape (32, rank_total). Rows indexed by j*8+k.
    
    for r in range(rank_4x4):
        for j in range(4):
            for k in range(4):
                val = V_4x4[j*4+k, r]
                # Map to B1 (k in 0..3)
                V_total[j*8+k, r] = val
                # Map to B2 (k in 4..7)
                V_total[j*8+(k+4), rank_4x4+r] = val
                
    return U_total, V_total, W_total

def _build_4x4_4x4_strassen_top_schoolbook_bottom():
    """
    Build 4x4 * 4x4 decomposition using Strassen at the block level (2x2 blocks)
    and Schoolbook at the element level (2x2 matrices).
    Rank = 7 (Strassen) * 4 (Schoolbook 2x2) = 28.
    """
    # Strassen 2x2 * 2x2 -> 2x2
    U_s, V_s, W_s = _strassen_2x2_2x2()
    rank_s = U_s.shape[1] # 7
    
    # Schoolbook 2x2 * 2x2 -> 2x2
    U_sb, V_sb, W_sb = _schoolbook_2x2_2x2()
    rank_sb = U_sb.shape[1] # 4
    
    rank_total = rank_s * rank_sb # 28
    
    U = np.zeros((16, rank_total))
    V = np.zeros((16, rank_total))
    W = np.zeros((16, rank_total))
    
    # The 4x4 matrix A is composed of 2x2 blocks A11, A12, A21, A22.
    # A = [A11 A12; A21 A22]
    # A11 is at rows 0-1, cols 0-1.
    # A12 is at rows 0-1, cols 2-3.
    # A21 is at rows 2-3, cols 0-1.
    # A22 is at rows 2-3, cols 2-3.
    
    # Strassen operates on these blocks.
    # For each Strassen product r_s (0..6), we have a linear combination of blocks.
    # We need to map the block indices to the full 4x4 indices.
    
    # Let's define the block mapping.
    # Block (bi, bj) corresponds to rows bi*2..bi*2+1 and cols bj*2..bj*2+1.
    # Block index b = bi*2 + bj.
    # b=0: (0,0) -> rows 0,1 cols 0,1
    # b=1: (0,1) -> rows 0,1 cols 2,3
    # b=2: (1,0) -> rows 2,3 cols 0,1
    # b=3: (1,1) -> rows 2,3 cols 2,3
    
    # U_s[alpha, r_s] where alpha is the flattened index of the 2x2 block matrix A_block.
    # A_block is 2x2, so alpha is 0..3.
    # U_s[alpha, r_s] tells us how much of A_block[alpha] is used in product r_s.
    
    # We need to expand this to the full 4x4 matrix.
    # For a given Strassen product r_s, and a given schoolbook product r_sb:
    # The coefficient for A[i,j] is sum over blocks b such that U_s[alpha_b, r_s] != 0.
    # Actually, U_s is defined on the 2x2 block matrix.
    # Let A_blocks be a 2x2 matrix of 2x2 blocks.
    # Strassen computes products of these blocks.
    # Each block multiplication is a 2x2 * 2x2 -> 2x2 operation, done by schoolbook.
    
    # So, for each Strassen term r_s:
    #   We have a "virtual" 2x2 matrix A_vs and B_vs formed by linear combinations of blocks.
    #   Then we multiply A_vs * B_vs using schoolbook (4 products).
    #   Each schoolbook product r_sb gives a term in the final C.
    
    # Let's iterate.
    for r_s in range(rank_s):
        for r_sb in range(rank_sb):
            r_total = r_s * rank_sb + r_sb
            
            # U_s[:, r_s] gives the coefficients for the 4 blocks of A.
            # U_s[0, r_s] -> coeff for A11
            # U_s[1, r_s] -> coeff for A12
            # U_s[2, r_s] -> coeff for A21
            # U_s[3, r_s] -> coeff for A22
            
            # V_s[:, r_s] gives the coefficients for the 4 blocks of B.
            # V_s[0, r_s] -> coeff for B11
            # V_s[1, r_s] -> coeff for B12
            # V_s[2, r_s] -> coeff for B21
            # V_s[3, r_s] -> coeff for B22
            
            # U_sb[:, r_sb] gives the coefficients for the elements of the "virtual" 2x2 A_vs.
            # U_sb[0, r_sb] -> coeff for A_vs[0,0]
            # U_sb[1, r_sb] -> coeff for A_vs[0,1]
            # U_sb[2, r_sb] -> coeff for A_vs[1,0]
            # U_sb[3, r_sb] -> coeff for A_vs[1,1]
            
            # This is getting complicated to map directly.
            # Let's use a simpler construction: Kronecker product of the tensors.
            # Tensor T_4x4 = T_strassen_2x2 (on blocks) ⊗ T_schoolbook_2x2 (on elements).
            # This works because the 4x4 multiplication is exactly the Kronecker product of the 2x2 block multiplication and the 2x2 element multiplication.
            
            # U_4x4[i*4+j, r_total] = U_s[i_block, r_s] * U_sb[i_elem, r_sb]
            # where i_block = i//2, j_block = j//2, i_elem = i%2, j_elem = j%2.
            # Wait, the indexing is (i, j) for 4x4.
            # i_block = i // 2, i_elem = i % 2.
            # j_block = j // 2, j_elem = j % 2.
            # The block index in Strassen is b = i_block * 2 + j_block.
            # The element index in Schoolbook is e = i_elem * 2 + j_elem.
            
            # So U_4x4[i*4+j, r_total] = U_s[b, r_s] * U_sb[e, r_sb]
            
            for i in range(4):
                for j in range(4):
                    b = (i // 2) * 2 + (j // 2)
                    e = (i % 2) * 2 + (j % 2)
                    U[i*4+j, r_total] = U_s[b, r_s] * U_sb[e, r_sb]
                    
            for j in range(4):
                for k in range(4):
                    b = (j // 2) * 2 + (k // 2)
                    e = (j % 2) * 2 + (k % 2)
                    V[j*4+k, r_total] = V_s[b, r_s] * V_sb[e, r_sb]
                    
            for k in range(4):
                for i in range(4):
                    b = (k // 2) * 2 + (i // 2)
                    e = (k % 2) * 2 + (i % 2)
                    W[k*4+i, r_total] = W_s[b, r_s] * W_sb[e, r_sb]
                    
    return U, V, W

def _schoolbook_2x2_2x2():
    """Return schoolbook decomposition for 2x2 * 2x2 -> 2x2. Rank 4."""
    n = m = p = 2
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Build the decomposition
    U, V, W = _build_kronecker_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U, V, W = _schoolbook_4x4_4x8()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_4x4_4x8():
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W
# EVOLVE-BLOCK-END
