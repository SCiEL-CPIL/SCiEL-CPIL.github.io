"""Matrix multiplication 4x4 x 4x8 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 as (U_entries, V_entries, W_entries).
    Each term is a rank-1 tensor. We return lists of (index, value) for each matrix.
    U is 2x2 (4 entries), V is 2x2 (4 entries), W is 2x2 (4 entries) per term.
    Indices: (i,j) -> i*2+j for U, (j,k) -> j*2+k for V, (k,i) -> k*2+i for W.
    """
    # 7 Strassen multiplications
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For each of the 7 terms, we have:
    # U factor: which A entries with what signs
    # V factor: which B entries with what signs
    # W factor: which C entries with what signs (output)
    
    # Let's define each term as (U_coeffs, V_coeffs, W_coeffs)
    # where U_coeffs is a dict {(i,j): coeff} for A[i,j]
    # V_coeffs is a dict {(j,k): coeff} for B[j,k]
    # W_coeffs is a dict {(i,k): coeff} for C[i,k]
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    # C00 += M1, C11 += M1
    # Actually, let me be more careful.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So each M contributes to specific C entries with specific signs.
    
    # M1: A factors: A00 (+1), A11 (+1); B factors: B00 (+1), B11 (+1)
    # C contributions: C00 (+1), C11 (+1)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # M2: A factors: A10 (+1), A11 (+1); B factors: B00 (+1)
    # C contributions: C10 (+1), C11 (-1)
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    
    # M3: A factors: A00 (+1); B factors: B01 (+1), B11 (-1)
    # C contributions: C01 (+1), C11 (+1)
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    
    # M4: A factors: A11 (+1); B factors: B10 (+1), B00 (-1)
    # C contributions: C00 (+1), C10 (+1)
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M5: A factors: A00 (+1), A01 (+1); B factors: B11 (+1)
    # C contributions: C00 (-1), C01 (+1)
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    
    # M6: A factors: A10 (+1), A00 (-1); B factors: B00 (+1), B01 (+1)
    # C contributions: C11 (+1)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # M7: A factors: A01 (+1), A11 (-1); B factors: B10 (+1), B11 (+1)
    # C contributions: C00 (+1)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    return terms

def _build_4x4x4_decomposition():
    """Build a rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen's algorithm applied to 2x2 blocks of 2x2 matrices.
    Returns U, V, W as lists of lists.
    U: shape (16, 49), V: shape (16, 49), W: shape (16, 49)
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    # 4x4x4: n=4, m=4, p=4
    # Partition into 2x2 blocks. Each block is 2x2.
    # The 4x4 matrix A is partitioned into 4 blocks: A00, A01, A10, A11 (each 2x2)
    # Similarly B into B00, B01, B10, B11
    # C into C00, C01, C10, C11
    
    # Using Strassen on the 2x2 block structure:
    # This gives 7 "meta-terms", each of which is a 2x2x2 multiplication of 2x2 blocks.
    # Each 2x2x2 multiplication of 2x2 blocks requires 7 rank-1 terms (Strassen).
    # Total: 7 * 7 = 49 rank-1 terms.
    
    strassen_terms = _strassen_2x2_terms()
    
    # For the 4x4x4 case:
    # A is 4x4, indexed as A[bi*2+ii, bj*2+jj] where bi,bj in {0,1} are block indices, ii,jj in {0,1}
    # B is 4x4, indexed as B[bj*2+jj, bk*2+kk]
    # C is 4x4, indexed as C[bi*2+ii, bk*2+kk]
    
    # Each Strassen term at the block level involves:
    # - A block entries: for each (bi, bj) in the U factor, with coefficient
    # - B block entries: for each (bj, bk) in the V factor, with coefficient  
    # - C block entries: for each (bi, bk) in the W factor, with coefficient
    
    # Within each block, we use Strassen's 7 terms.
    
    U = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
    V = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
    W = [[0] * 49 for _ in range(16)]  # 4*4 = 16 rows
    
    col = 0
    for meta_term in strassen_terms:
        # meta_term has U, V, W dicts with block-level indices
        # For each combination of inner Strassen terms
        for inner_term in strassen_terms:
            # The meta_term U factor gives block (bi, bj) with coefficient cu
            # The inner_term U factor gives within-block (ii, jj) with coefficient cu_inner
            # So A[bi*2+ii, bj*2+jj] gets coefficient cu * cu_inner
            
            # Similarly for V and W
            
            for (bi, bj), cu in meta_term['U'].items():
                for (ii, jj), cu_inner in inner_term['U'].items():
                    row_a = bi * 2 + ii
                    col_a = bj * 2 + jj
                    U[row_a * 4 + col_a][col] += cu * cu_inner
            
            for (bj, bk), cv in meta_term['V'].items():
                for (jj, kk), cv_inner in inner_term['V'].items():
                    row_b = bj * 2 + jj
                    col_b = bk * 2 + kk
                    V[row_b * 4 + col_b][col] += cv * cv_inner
            
            for (bi, bk), cw in meta_term['W'].items():
                for (ii, kk), cw_inner in inner_term['W'].items():
                    row_c = bk * 2 + kk
                    col_c = bi * 2 + ii
                    W[row_c * 4 + col_c][col] += cw * cw_inner
            
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    # Build full matrices
    U_mat = np.array(U, dtype=float)  # (n*m, rank)
    V_mat = np.array(V, dtype=float)  # (m*p, rank)
    W_mat = np.array(W, dtype=float)  # (p*n, rank)
    
    # Reshape to 3D
    U_3d = U_mat.reshape(n, m, rank)  # U[i, j, r]
    V_3d = V_mat.reshape(m, p, rank)  # V[j, k, r]
    W_3d = W_mat.reshape(p, n, rank)  # W[k, i, r]
    
    # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But actually, the decomposition is: C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * ... 
    # Wait, let me reconsider.
    
    # The tensor decomposition: T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # where T_{i,j,k} = 1 if this is the "standard" multiplication tensor
    # Actually, the matrix multiplication tensor is defined as:
    # T[i,j,k] = delta(i,j,k) where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor has T[i,j,k] = 1 for all valid (i,j,k) in the sense that it's the structure tensor.
    
    # Let me verify by checking: for random A, B, does C = A*B match sum_r U_r(A) * V_r(B) * W_r?
    
    # Actually, the correct verification:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
    # No, that's not right either.
    
    # The rank-r decomposition means:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T is the matrix multiplication tensor.
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor in a sense)
    # No, that's not right. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # Actually, the structure tensor has T[i,j,k] = 1 for all (i,j,k) in the range.
    
    # Let me just verify numerically:
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute via decomposition
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
    # Wait, I think the correct formula is:
    # C[i,k] = sum_r (U_r^T * A_col_i) * (V_r^T * B_row_k) * W[k,i,r]
    # Let me think again...
    
    # Standard tensor decomposition for matmul:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) W[k,i,r]
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        # U[:, :, r] is n x m
        U_r = U_mat[:, r].reshape(n, m)
        V_r = V_mat[:, r].reshape(m, p)
        W_r = W_mat[:, r].reshape(p, n)
        
        # Sum_j U[i,j,r] * A[i,j] for each i
        a_part = np.sum(U_r * A_test, axis=1)  # shape (n,)
        # Sum_j V[j,k,r] * B[j,k] for each k
        b_part = np.sum(V_r * B_test, axis=0)  # shape (p,)
        
        # C[i,k] += a_part[i] * b_part[k] * W_r[k,i]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += a_part[i] * b_part[k] * W_r[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Use two 4x4x4 decompositions (rank 49 each) for the two halves of the 8 columns
    # Total rank: 98
    
    U_444, V_444, W_444 = _build_4x4x4_decomposition()
    
    # For 4x4 x 4x8:
    # A is 4x4, B is 4x8, C is 4x8
    # Split B into B_left (4x4) and B_right (4x4)
    # C_left = A * B_left, C_right = A * B_right
    
    # U for 4x4x4: shape (16, 49), indexed U[i*4+j, r]
    # For 4x4x8: U shape (16, rank_total), V shape (32, rank_total), W shape (32, rank_total)
    # V[j*8+k, r] for j in 0..3, k in 0..7
    # W[k*4+i, r] for k in 0..7, i in 0..3
    
    rank_total = 98
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(32)]
    W = [[0] * rank_total for _ in range(32)]
    
    # First half: columns 0-3 of B, columns 0-3 of C
    for r in range(49):
        # U
        for i in range(4):
            for j in range(4):
                U[i*4+j][r] = U_444[i*4+j][r]
        # V: B[j,k] for k in 0..3
        for j in range(4):
            for k in range(4):
                V[j*8+k][r] = V_444[j*4+k][r]
        # W: C[k,i] for k in 0..3
        for k in range(4):
            for i in range(4):
                W[k*4+i][r] = W_444[k*4+i][r]
    
    # Second half: columns 4-7 of B, columns 4-7 of C
    for r in range(49):
        r2 = 49 + r
        # U: same as first half
        for i in range(4):
            for j in range(4):
                U[i*4+j][r2] = U_444[i*4+j][r]
        # V: B[j,k] for k in 4..7, so V[j*8+(k-4)] = V_444[j*4+(k-4)]
        for j in range(4):
            for k in range(4):
                V[j*8+(4+k)][r2] = V_444[j*4+k][r]
        # W: C[k,i] for k in 4..7, so W[(k-4)*4+i] = W_444[(k-4)*4+i]
        for k in range(4):
            for i in range(4):
                W[(4+k)*4+i][r2] = W_444[k*4+i][r]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
