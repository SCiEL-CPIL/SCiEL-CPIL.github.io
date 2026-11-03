"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the factors exactly decompose the MM tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    # Reshape
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        Uc = U_r[:, :, c]  # (n, m)
        Vc = V_r[:, :, c]  # (m, p)
        Wc = W_r[:, :, c]  # (p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        tmp = np.einsum('ij,jk->ijk', Uc, Vc)
        T_computed += tmp * Wc.T[:, np.newaxis, :]
    
    T_expected = np.ones((n, m, p), dtype=np.float64)
    return np.max(np.abs(T_computed - T_expected)) < tol


def _construct_strassen_2x2():
    """Return U, V, W factors for 2x2 x 2x2 matrix multiplication (rank 7)."""
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Index: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # A indices: (i,j) -> i*m+j
    # B indices: (j,k) -> j*p+k
    # C indices: (k,i) -> k*n+i (output factor uses trace convention W[k*n+i])
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=U[0,0], A11=U[1,1]; V: B00=V[0,0], B11=V[1,1]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += M1, C11 += M1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += M2, C11 -= M2
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0] and C[1,1]
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C01 += M3, C11 += M3
    W[0, 2] = 1; W[3, 2] = 1  # C[0,1] and C[1,1]
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C00 += M4, C10 += M4
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0] and C[1,0]
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 -= M5, C01 += M5
    W[0, 4] = -1; W[0, 4] = 1  # C[0,0] and C[0,1]... wait
    # C00 = M1 + M4 - M5 + M7, C01 = M3 + M5
    W[0, 4] = -1  # C[0,0] -= M5
    W[0, 4] = 1   # C[0,1] += M5 -- conflict!
    # Let me redo: W[k*n+i, c] where output is C[i,k]
    # C00: k=0, i=0 -> W[0*2+0, c] = W[0, c]
    # C01: k=1, i=0 -> W[1*2+0, c] = W[2, c]
    # C10: k=0, i=1 -> W[0*2+1, c] = W[1, c]
    # C11: k=1, i=1 -> W[1*2+1, c] = W[3, c]
    
    # Redo with correct W indexing:
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C10
    W[3, 1] = -1  # C11
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00
    W[1, 3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00
    W[2, 4] = 1   # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1   # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C00
    
    return U, V, W


def _construct_strassen_4x4():
    """Construct rank-49 decomposition for 4x4 x 4x4 using block Strassen."""
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Split 4x4 into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    #
    # Block Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product, so we need 7 scalar products per Mi
    # Total: 7 * 7 = 49 scalar products
    
    # Block indexing:
    # A block (bi, bj) where bi, bj in {0,1}: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block (bj, bk): rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # C block (bk, bi): rows 2*bk..2*bk+1, cols 2*bi..2*bi+1
    
    # For each block-level Strassen term (7 terms), we need a 2x2x2 Strassen (7 terms)
    # Total 49 terms
    
    # Block-level coefficients:
    # term_block: (A_blocks, B_blocks, C_blocks, coeff)
    # A_blocks: list of (bi, bj) with sign
    # B_blocks: list of (bj, bk) with sign
    # C_blocks: list of (bk, bi) with sign
    
    block_terms = [
        # (A_blocks, B_blocks, C_blocks)
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),   # M1
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1), (1,1,-1))]),              # M2
        ([((0,0,1))], [((0,1,1), (1,1,-1))], [((0,1,1), (1,1,1))]),              # M3
        ([((1,1,1))], [((1,0,1), (0,0,-1))], [((0,0,1), (1,0,1))]),              # M4
        ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,0,-1), (0,1,1))]),              # M5
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),              # M6
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1))]),              # M7
    ]
    
    # 2x2 Strassen scalar factors
    s2U, s2V, s2W = _construct_strassen_2x2()
    
    col = 0
    for a_blocks, b_blocks, c_blocks in block_terms:
        for s2c in range(7):
            # For this scalar product within the 2x2 Strassen:
            # U contributes: for each (bi,bj,sign) in a_blocks, for each (u_i, u_j) in s2U nonzero
            # V contributes: for each (bj,bk,sign) in b_blocks, for each (v_j, v_k) in s2V nonzero
            # W contributes: for each (bk,bi,sign) in c_blocks, for each (w_k, w_i) in s2W nonzero
            
            for (bi, bj, sa) in a_blocks:
                for (bj2, bk, sb) in b_blocks:
                    if bj != bj2:
                        continue
                    for (bk2, bi2, sc) in c_blocks:
                        if bk != bk2 or bi != bi2:
                            continue
                        
                        # Now add the scalar Strassen contribution
                        for ui in range(2):
                            for uj in range(2):
                                val_u = s2U[ui * 2 + uj, s2c]
                                if val_u == 0:
                                    continue
                                i = 2 * bi + ui
                                j = 2 * bj + uj
                                U[i * m + j, col] += sa * val_u
                        
                        for uj2 in range(2):
                            for uk in range(2):
                                val_v = s2V[uj2 * 2 + uk, s2c]
                                if val_v == 0:
                                    continue
                                j = 2 * bj + uj2
                                k = 2 * bk + uk
                                V[j * p + k, col] += sb * val_v
                        
                        for uk2 in range(2):
                            for ui2 in range(2):
                                val_w = s2W[uk2 * 2 + ui2, s2c]
                                if val_w == 0:
                                    continue
                                k = 2 * bk + uk2
                                i = 2 * bi + ui2
                                W[k * n + i, col] += sc * val_w
            
            col += 1
    
    return U, V, W, rank


def _construct_4x4x5():
    """Construct a decomposition for 4x4 x 4x5."""
    n, m, p = 4, 4, 5
    
    # Start with 4x4 x 4x4 Strassen (rank 49)
    U4, V4, W4, rank4 = _construct_strassen_4x4()
    
    # We need to handle the 5th column of B (index k=4)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # For k=0..3, the 4x4x4 decomposition handles it.
    # For k=4, we need additional terms.
    #
    # The 4x4x4 decomposition has V entries only for k=0..3 (since it's 4x4).
    # So for k=4, V[j*p+4, c] = 0 for all c in the first 49 terms.
    # We need new terms where V[j*4+4, c] != 0 and W[4*n+i, c] != 0.
    
    # We need: sum_c U[i*m+j, c] * V[j*4+4, c] * W[4*n+i, c] = 1 for all i, j
    
    # Simple approach: use 4 new terms, one for each j
    # Term for j: U[i*4+j, c_j] = 1, V[j*4+4, c_j] = 1, W[4*4+i, c_j] = 1
    # Then for fixed (i,j): sum over c gives 1 from term c_j (since U[i*4+j, c_j]=1, V[j*4+4, c_j]=1, W[16+i, c_j]=1)
    # And 0 from all other terms.
    
    rank_total = rank4 + m  # 49 + 4 = 53
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Copy the 4x4x4 part
    U[:, :rank4] = U4
    V[:, :rank4] = V4
    W[:, :rank4] = W4
    
    # Add terms for k=4
    for j in range(m):
        c = rank4 + j
        for i in range(n):
            U[i * m + j, c] = 1
            W[4 * n + i, c] = 1
        V[j * p + 4, c] = 1
    
    return U, V, W, rank_total


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank


def _matricize_tensor(T, mode):
    """Matricize tensor T (shape n,m,p) along given mode."""
    n, m, p = T.shape
    if mode == 0:
        return T.reshape(n, m * p)
    elif mode == 1:
        return T.transpose(1, 0, 2).reshape(m, n * p)
    elif mode == 2:
        return T.transpose(2, 0, 1).reshape(p, n * m)


def _rank_revealing_qr(A, max_rank=20, tol=1e-10):
    """Compute a low-rank approximation via QR with column pivoting."""
    n, p = A.shape
    rank = min(n, p, max_rank)
    
    # QR with column pivoting
    Q, R, P = np.linalg.qr(A, mode='complete')
    
    # Determine numerical rank
    diag_R = np.abs(np.diag(R))
    if rank == 0:
        return A[:, :0], np.zeros(n)
    
    num_rank = np.sum(diag_R > tol * diag_R[0])
    num_rank = max(1, min(num_rank, rank))
    
    # Get the first num_rank columns of Q and corresponding rows of R
    Qr = Q[:, :num_rank]
    Rr = R[:num_rank, :]
    
    return Qr, Rr


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # First, try the deterministic construction (rank 53)
    try:
        U, V, W, rank = _construct_4x4x5()
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W, rank = _construct_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
