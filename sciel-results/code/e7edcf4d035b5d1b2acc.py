"""Rewritten program for matrix multiplication tensor decomposition (4x5, 5x6)."""
import time
import random
import math
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """Return rank-49 Strassen decomposition for 4x4x4 matrix multiplication.
    Uses standard Strassen applied to 2x2 blocks, giving 49 = 7^2 multiplications.
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r] for i,j,k in [0,4).
    """
    # Strassen for 2x2: 7 multiplications
    # For 4x4: split into 2x2 blocks, apply Strassen recursively
    # Level 1: 4x4 -> 4 sub-blocks of 2x2, 7 mults at this level
    # Level 2: each 2x2 mult uses 7 rank-1 terms
    # Total: 7 * 7 = 49 rank-1 terms
    
    # Strassen 2x2: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 4x4, split A into [[A00,A01],[A10,A11]] where each is 2x2
    # Same for B and C.
    # Apply Strassen at the 4x4 level (7 multiplications of 2x2 matrices)
    # Each 2x2 multiplication uses Strassen (7 rank-1 terms)
    # Total: 49 rank-1 terms
    
    # Let's build this explicitly.
    # First, define the Strassen 2x2 decomposition as a list of (coeff_U, coeff_V, coeff_W) for each rank-1 term.
    # For 2x2: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen 2x2 rank-1 terms (7 terms):
    # Each term r has U, V, W entries for i,j,k in [0,2)
    
    strassen_2x2 = []
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1; V: B00=1, B11=1; W: C00+=1, C11+=1
    strassen_2x2.append(
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    )
    # M2 = (A10+A11)B00
    strassen_2x2.append(
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,0,0),(1,0,1)])  # C10 += M2
    )
    # M3 = A00(B10-B11)
    strassen_2x2.append(
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,0,1)])  # C01 += M3, C10 += M3
    )
    # M4 = A11(B01-B00)
    strassen_2x2.append(
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)])  # C00 += M4, C10 += M4
    )
    # M5 = (A00+A01)B11
    strassen_2x2.append(
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])  # C00 -= M5, C01 += M5
    )
    # M6 = (A10-A00)(B00+B01)
    strassen_2x2.append(
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])  # C11 += M6
    )
    # M7 = (A01-A11)(B10+B11)
    strassen_2x2.append(
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)])  # C00 += M7, C11 += M7
    )
    
    # Now for 4x4: we have 7 "super-multiplications" of 2x2 matrices.
    # Each super-multiplication produces a 2x2 result that contributes to C blocks.
    # The 7 super-multiplications are:
    # S1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # S2 = (A10+A11)B00 -> contributes to C10
    # S3 = A00(B10-B11) -> contributes to C01, C10
    # S4 = A11(B01-B00) -> contributes to C00, C10
    # S5 = (A00+A01)B11 -> contributes to C00, C01
    # S6 = (A10-A00)(B00+B01) -> contributes to C11
    # S7 = (A01-A11)(B10+B11) -> contributes to C00, C11
    
    # For each super-multiplication, we need to expand using Strassen 2x2.
    # The U, V, W for each super-multiplication are 2x2 matrices (in the 4x4 indexing).
    
    # Let's build the full 4x4 decomposition.
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    # U index: i*4+j for i,j in [0,4)
    # V index: j*4+k for j,k in [0,4)
    # W index: k*4+i for k,i in [0,4)
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    col = 0
    for s in range(7):
        # s-th super-multiplication
        # For each of the 7 Strassen 2x2 rank-1 terms:
        for t in range(7):
            r = s * 7 + t
            # Get the Strassen 2x2 term
            u_terms, v_terms, w_terms = strassen_2x2[t]
            
            # Now map the 2x2 indices to 4x4 indices based on which super-multiplication
            # Super-multiplication s involves specific 2x2 blocks of A, B, C.
            # We need to define for each s:
            #   Which 2x2 blocks of A are combined (with coefficients) to form the "A" of the 2x2 mult
            #   Which 2x2 blocks of B are combined to form the "B"
            #   Which 2x2 blocks of C receive the result
            
            # Let me define the block structure:
            # A = [[A00, A01], [A10, A11]] where Aij is 2x2
            # In 4x4 indexing: A00 = rows 0-1, cols 0-1; A01 = rows 0-1, cols 2-3
            #                   A10 = rows 2-3, cols 0-1; A11 = rows 2-3, cols 2-3
            # Similarly for B and C.
            
            # For super-multiplication s, the "effective" 2x2 A matrix is a linear combination of A blocks.
            # The Strassen 2x2 term t has u_terms giving coefficients for the 2x2 A matrix.
            # Each entry (i2, j2) in the 2x2 A corresponds to a specific 2x2 block in the 4x4 A.
            
            # This is getting complex. Let me use a different approach:
            # Directly enumerate the 49 rank-1 terms by expanding Strassen.
            
            pass
    
    # This approach is getting too complex. Let me use a known explicit Strassen 4x4.
    # Actually, let me just use schoolbook for now and return rank 120,
    # but structure the code for the block decomposition.
    
    # For now, return None to indicate this function is not yet implemented.
    return None

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication.
    For each (i,k), the (i,k)-entry of C = A*B must equal sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] ... 
    Actually the tensor T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] must equal delta? 
    No: the multiplication tensor has T[i,j,k] = 1 if we are computing C[i,k] = sum_j A[i,j]B[j,k].
    The decomposition requires: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if (i,j,k) is a "1" in the tensor.
    But the matrix multiplication tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]? No.
    
    Let me reconsider. The matrix multiplication tensor for multiplying (n x m) by (m x p) is a 3-tensor T of shape (n*m, m*p, p*n) where
    T[i*m + j, j'*p + k, k'*n + i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'} ... 
    
    Actually, the standard tensor decomposition for matrix multiplication:
    We want C = A B where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k].
    
    The tensor T is indexed by (i,j,k) with T[i,j,k] = 1. The decomposition is:
    T[i,j,k] = sum_{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The convention in the task: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    
    So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    And for all other combinations of the three indices that don't correspond to valid (i,j,k) triples... 
    Wait, the indices of U, V, W are coupled: U index is i*m+j, V index is j*p+k, W index is k*n+i.
    So the three indices (a, b, c) where a = i*m+j, b = j*p+k, c = k*n+i must satisfy:
    j = a % m, i = a // m, j = b // p, k = b % p, k = c // n, i = c % n.
    
    So the consistency conditions are:
    a % m == b // p  (j matches)
    b % p == c // n  (k matches)
    c % n == a // m  (i matches)
    
    For valid (i,j,k), the condition is sum_r U[a,r]V[b,r]W[c,r] = 1.
    For invalid combinations, there's no constraint? Actually, the tensor is only defined on the "support" where the indices are consistent.
    
    Hmm, but actually in tensor decomposition for matrix multiplication, the tensor is a 3-tensor with entries T[a,b,c] where a indexes A, b indexes B, c indexes C. The entry T[a,b,c] is 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j', i=i', k=k', and 0 otherwise. But since a,b,c uniquely determine i,j,k (with the consistency), T[a,b,c] = 1 if consistent, 0 otherwise.
    
    Wait no. Let me think again. The matrix multiplication can be written as:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This is a multilinear map. The associated tensor T has entries T[i,j,k] = 1 (for all i,j,k) but that's not right either.
    
    Actually, the correct way: the tensor T of matrix multiplication (n x m) * (m x p) -> (n x p) is a 3-tensor with indices (i, j, k) where i in [n], j in [m], k in [p], and T[i,j,k] = 1. The decomposition is:
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    So for ALL i,j,k, the sum must equal 1. That's the condition.
    
    Let me verify with the schoolbook: U[i*m+j, r] = 1 only when r = i*m*p + j*p + k, V[j*p+k, r] = 1 only when r = i*m*p + j*p + k, W[k*n+i, r] = 1 only when r = i*m*p + j*p + k. So the product is 1 only when all three are 1, which happens only for the specific r. So sum_r = 1. Correct.
    
    So the verification is: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = n, m, p
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r] for i,j,k in [0,2).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    
    # M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1; V[3][2] = -1  # B10, -B11
    W[1][2] = 1; W[2][2] = 1  # C01, C10
    
    # M4 = A11(B01-B00)
    U[3][3] = 1  # A11
    V[1][3] = 1; V[0][3] = -1  # B01, -B00
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1; W[1][4] = 1  # -C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, -A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, -A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = 1; W[3][6] = 1  # C00, C11
    
    return U, V, W

def _block_decomposition():
    """Block decomposition for 4x5x6 using Strassen on 4x4x4 sub-block.
    Split m=5 into 4+1, p=6 into 4+2.
    Blocks: (4x4x4), (4x4x2), (4x1x4), (4x1x2)
    Use Strassen (rank 49) for 4x4x4, schoolbook for others.
    Total rank: 49 + 32 + 16 + 8 = 105
    """
    n, m, p = 4, 5, 6
    
    # Block boundaries
    m1, m2 = 4, 1  # m = m1 + m2
    p1, p2 = 4, 2  # p = p1 + p2
    
    # Sub-block ranks
    rank_444 = 49  # Strassen
    rank_442 = 4*4*2  # schoolbook
    rank_414 = 4*1*4  # schoolbook
    rank_412 = 4*1*2  # schoolbook
    total_rank = rank_444 + rank_442 + rank_414 + rank_412
    
    U = [[0]*total_rank for _ in range(n*m)]  # 20 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 30 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 24 rows
    
    col = 0
    
    # Block 1: (4x4x4) - Strassen
    # i in [0,4), j in [0,4), k in [0,4)
    # U index: i*5+j, V index: j*6+k, W index: k*4+i
    sU, sV, sW = _strassen_4x4x4()
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*m + j][col + r] = sU[i*4 + j][r]
        for j in range(4):
            for k in range(4):
                V[j*p + k][col + r] = sV[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k*n + i][col + r] = sW[k*4 + i][r]
    col += rank_444
    
    # Block 2: (4x4x2) - schoolbook
    # i in [0,4), j in [0,4), k in [4,6)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Block 3: (4x1x4) - schoolbook
    # i in [0,4), j in [4,5), k in [0,4)
    for i in range(4):
        for j in range(4, 5):
            for k in range(4):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Block 4: (4x1x2) - schoolbook
    # i in [0,4), j in [4,5), k in [4,6)
    for i in range(4):
        for j in range(4, 5):
            for k in range(4, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """Return rank-49 Strassen decomposition for 4x4x4.
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r] for i,j,k in [0,4).
    """
    # Build from two levels of Strassen 2x2.
    # Level 1: 4x4 split into 2x2 blocks, 7 super-multiplications
    # Level 2: each super-multiplication is a 2x2 mult, use Strassen (7 terms)
    # Total: 49 terms
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Strassen 2x2 base terms
    s2U, s2V, s2W = _strassen_2x2()
    
    # Define the 7 super-multiplications for 4x4:
    # Each is defined by:
    #   a_blocks: list of (block_row, block_col, coeff) for A
    #   b_blocks: list of (block_row, block_col, coeff) for B
    #   c_blocks: list of (block_row, block_col, coeff) for C (where result goes)
    # Block indices: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
    
    # S1 = (A00+A11)(B00+B11) -> C00 += S1, C11 += S1
    # S2 = (A10+A11)B00 -> C10 += S2
    # S3 = A00(B10-B11) -> C01 += S3, C10 += S3
    # S4 = A11(B01-B00) -> C00 += S4, C10 += S4
    # S5 = (A00+A01)B11 -> C00 -= S5, C01 += S5
    # S6 = (A10-A00)(B00+B01) -> C11 += S6
    # S7 = (A01-A11)(B10+B11) -> C00 += S7, C11 += S7
    
    super_mults = [
        # (a_blocks, b_blocks, c_blocks)
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),   # S1
        ([((1,0,1),(1,1,1))], [((0,0,1))], [((1,0,1))]),                      # S2
        ([((0,0,1))], [((1,0,1),(1,1,-1))], [((0,1,1),(1,0,1))]),             # S3
        ([((1,1,1))], [((0,1,1),(0,0,-1))], [((0,0,1),(1,0,1))]),             # S4
        ([((0,0,1),(0,1,1))], [((1,1,1))], [((0,0,-1),(0,1,1))]),             # S5
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [((1,1,1))]),             # S6
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),     # S7
    ]
    
    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(super_mults):
        for t_idx in range(7):
            r = s_idx * 7 + t_idx
            # For each Strassen 2x2 term t_idx, we have:
            # s2U[i2*2+j2, t_idx] for i2,j2 in [0,2)
            # s2V[j2*2+k2, t_idx] for j2,k2 in [0,2)
            # s2W[k2*2+i2, t_idx] for k2,i2 in [0,2)
            
            # Map 2x2 indices to 4x4 block indices:
            # i2 in [0,2) maps to block row i2, j2 in [0,2) maps to block col j2
            # So the 2x2 "A" matrix has entries at block positions (i2, j2)
            # Each block position (br, bc) corresponds to 4x4 indices:
            #   rows: br*2 to br*2+1, cols: bc*2 to bc*2+1
            
            # For the "A" part:
            # s2U[i2*2+j2, t_idx] gives coefficient for A[i2*2: i2*2+2, j2*2: j2*2+2]
            # But we need to scale by the super-multiplication coefficients.
            
            # For each (i2, j2) pair where s2U[i2*2+j2, t_idx] != 0:
            #   The corresponding 2x2 block of A is at block position (i2, j2)
            #   We need to find which super-multiplication block this maps to.
            
            # Actually, the Strassen 2x2 decomposition already handles the 2x2 structure.
            # The super-multiplication tells us which 2x2 blocks of the 4x4 matrix to use.
            # So for super-multiplication s with a_blocks = [(br, bc, coeff), ...]:
            #   The "effective" 2x2 A matrix has entry (i2, j2) = sum of coeff for blocks where (br, bc) = (i2, j2)
            #   Then we apply Strassen 2x2 to this effective 2x2 matrix.
            
            # Let me build the effective 2x2 matrices:
            eff_a = [[0]*2 for _ in range(2)]
            for (br, bc, coeff) in a_blocks:
                eff_a[br][bc] += coeff
            
            eff_b = [[0]*2 for _ in range(2)]
            for (br, bc, coeff) in b_blocks:
                eff_b[br][bc] += coeff
            
            eff_c = [[0]*2 for _ in range(2)]
            for (br, bc, coeff) in c_blocks:
                eff_c[br][bc] += coeff
            
            # Now apply Strassen 2x2 to the effective matrices:
            # For each rank-1 term t_idx:
            for i2 in range(2):
                for j2 in range(2):
                    coeff_a = s2U[i2*2+j2][t_idx] * eff_a[i2][j2]
                    if coeff_a != 0:
                        # Map to 4x4: rows i2*2 to i2*2+1, cols j2*2 to j2*2+1
                        for di in range(2):
                            for dj in range(2):
                                i4 = i2*2 + di
                                j4 = j2*2 + dj
                                U[i4*4 + j4][r] += coeff_a
            
            for j2 in range(2):
                for k2 in range(2):
                    coeff_b = s2V[j2*2+k2][t_idx] * eff_b[j2][k2]
                    if coeff_b != 0:
                        for dj in range(2):
                            for dk in range(2):
                                j4 = j2*2 + dj
                                k4 = k2*2 + dk
                                V[j4*4 + k4][r] += coeff_b
            
            for k2 in range(2):
                for i2 in range(2):
                    coeff_c = s2W[k2*2+i2][t_idx] * eff_c[k2][i2]
                    if coeff_c != 0:
                        for dk in range(2):
                            for di in range(2):
                                k4 = k2*2 + dk
                                i4 = i2*2 + di
                                W[k4*4 + i4][r] += coeff_c
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try block decomposition with Strassen
    try:
        result = _block_decomposition()
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
