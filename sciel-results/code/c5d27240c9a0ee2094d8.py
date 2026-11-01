import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 115.0  # safety margin before 150s kill

    n, m, p = 2, 3, 8
    # Decompose into 4 blocks of M_{2,3,2}
    # Each block: A(2x3) * B(3x2) = C(2x2)
    # Rank per block: 11 (Strassen-7 on 2x2x2 sub-block + 4 rank-1 for extra column)
    # Total rank: 4 * 11 = 44

    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44

    U = np.zeros((n*m, total_rank))  # (6, 44)
    V = np.zeros((m*p, total_rank))  # (24, 44)
    W = np.zeros((p*n, total_rank))  # (16, 44)

    for blk in range(num_blocks):
        base = blk * rank_per_block
        # For block blk, B columns are 2*blk and 2*blk+1
        # C columns (output) are 2*blk and 2*blk+1

        # Split A(2x3) into A0(2x2) [cols 0,1] and A1(2x1) [col 2]
        # Split B(3x2) into B0(3x2) [rows 0,1] and B1(3x1) [row 2]

        # Part 1: A0(2x2) * B0(2x2) = C00(2x2) -- use Strassen rank-7
        # Part 2: A1(2x1) * B1(3x1->actually just row 2) * ... 
        # Actually: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        # Strassen handles the first two terms (rank 7)
        # Extra: A[i,2]*B[2,k] for i in {0,1}, k in {2*blk, 2*blk+1} -> 4 rank-1 terms
        # Total: 7 + 4 = 11 per block.

        # Strassen for 2x2 * 2x2:
        # A_s = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B_s = [[B[0,2*blk], B[0,2*blk+1]], [B[1,2*blk], B[1,2*blk+1]]]
        # C_s[i,k] = A_s[i,0]*B_s[0,k] + A_s[i,1]*B_s[1,k]
        
        # Strassen 7 products for 2x2*2x2:
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B01-B11)
        # m4 = A11(B10-B00)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B01)
        # m7 = (A01-A11)(B10+B11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # In our indexing for block blk:
        # A[i,0] -> U index i*3+0, A[i,1] -> U index i*3+1, A[i,2] -> U index i*3+2
        # B[0,k] -> V index 0*p + k = k, B[1,k] -> V index 1*p+k = p+k, B[2,k] -> V index 2*p+k = 2p+k
        # C[i,k] -> W index k*n + i = 2k+i (for block blk: k is 2*blk or 2*blk+1)
        # W index for C[i, 2*blk] = (2*blk)*n + i = 4*blk + i
        # W index for C[i, 2*blk+1] = (2*blk+1)*n + i = 4*blk + 2 + i

        # Strassen part (indices 0-6 in block):
        # m1 = (A00+A11)(B00+B11)
        U[0*3+0, base+0] = 1   # A[0,0]
        U[1*3+1, base+0] = 1   # A[1,1]
        V[0*p + 2*blk, base+0] = 1     # B[0, 2*blk]
        V[1*p + (2*blk+1), base+0] = 1 # B[1, 2*blk+1]
        W[4*blk + 0, base+0] = 1       # C[0, 2*blk]
        W[4*blk + 3, base+0] = 1       # C[1, 2*blk+1]
        
        # m2 = (A10+A11)B00
        U[1*3+0, base+1] = 1   # A[1,0]
        U[1*3+1, base+1] = 1   # A[1,1]
        V[0*p + 2*blk, base+1] = 1     # B[0, 2*blk]
        W[4*blk + 1, base+1] = 1       # C[1, 2*blk]
        W[4*blk + 3, base+1] = -1      # C[1, 2*blk+1]
        
        # m3 = A00(B01-B11)
        U[0*3+0, base+2] = 1   # A[0,0]
        V[0*p + (2*blk+1), base+2] = 1   # B[0, 2*blk+1]
        V[1*p + (2*blk+1), base+2] = -1  # B[1, 2*blk+1]
        W[4*blk + 2, base+2] = 1         # C[0, 2*blk+1]
        W[4*blk + 3, base+2] = 1         # C[1, 2*blk+1]
        
        # m4 = A11(B10-B00)
        U[1*3+1, base+3] = 1   # A[1,1]
        V[1*p + 2*blk, base+3] = 1     # B[1, 2*blk]
        V[0*p + 2*blk, base+3] = -1    # B[0, 2*blk]
        W[4*blk + 0, base+3] = 1       # C[0, 2*blk]
        W[4*blk + 1, base+3] = 1       # C[1, 2*blk]
        
        # m5 = (A00+A01)B11
        U[0*3+0, base+4] = 1   # A[0,0]
        U[0*3+1, base+4] = 1   # A[0,1]
        V[1*p + (2*blk+1), base+4] = 1  # B[1, 2*blk+1]
        W[4*blk + 0, base+4] = -1       # C[0, 2*blk]
        W[4*blk + 2, base+4] = 1        # C[0, 2*blk+1]
        
        # m6 = (A10-A00)(B00+B01)
        U[1*3+0, base+5] = 1   # A[1,0]
        U[0*3+0, base+5] = -1  # A[0,0]
        V[0*p + 2*blk, base+5] = 1        # B[0, 2*blk]
        V[0*p + (2*blk+1), base+5] = 1    # B[0, 2*blk+1]
        W[4*blk + 3, base+5] = 1          # C[1, 2*blk+1]
        
        # m7 = (A01-A11)(B10+B11)
        U[0*3+1, base+6] = 1   # A[0,1]
        U[1*3+1, base+6] = -1  # A[1,1]
        V[1*p + 2*blk, base+6] = 1        # B[1, 2*blk]
        V[1*p + (2*blk+1), base+6] = 1    # B[1, 2*blk+1]
        W[4*blk + 0, base+6] = 1          # C[0, 2*blk]
        
        # Extra part: A[i,2] * B[2,k] for i in {0,1}, k in {2*blk, 2*blk+1}
        # 4 rank-1 terms (indices 7-10 in block):
        # C[0, 2*blk] += A[0,2]*B[2, 2*blk]
        U[0*3+2, base+7] = 1
        V[2*p + 2*blk, base+7] = 1
        W[4*blk + 0, base+7] = 1
        
        # C[0, 2*blk+1] += A[0,2]*B[2, 2*blk+1]
        U[0*3+2, base+8] = 1
        V[2*p + (2*blk+1), base+8] = 1
        W[4*blk + 2, base+8] = 1
        
        # C[1, 2*blk] += A[1,2]*B[2, 2*blk]
        U[1*3+2, base+9] = 1
        V[2*p + 2*blk, base+9] = 1
        W[4*blk + 1, base+9] = 1
        
        # C[1, 2*blk+1] += A[1,2]*B[2, 2*blk+1]
        U[1*3+2, base+10] = 1
        V[2*p + (2*blk+1), base+10] = 1
        W[4*blk + 3, base+10] = 1

    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Check that U @ diag @ V^T reconstructs the multiplication tensor
        # The tensor T[i,j,k] = delta[i,k] for matrix mult A(2x3)*B(3x8)
        # Actually T[i,j,k] where i is row of A (0..1), j is col of A/row of B (0..2), k is col of B (0..7)
        # T[i,j,k] = 1 if i==k else 0 ... wait no.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor T[i,j,k] such that C = sum_r U[:,r] * V[:,r] * W[:,r]
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] (with appropriate indexing)
        
        # Let's construct the target tensor
        # U has shape (n*m, rank) -> index (i*m+j, r)
        # V has shape (m*p, rank) -> index (j*p+k, r)
        # W has shape (p*n, rank) -> index (k*n+i, r)
        
        # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} ? No.
        # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is T[i,j,k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k]
        # Actually T[i,j,k] = delta_{i,k} is wrong.
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
        # T[i,j,k] is 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
        # It contributes to C[i,k] always (for fixed i,j,k).
        # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
        
        # Let's think again. The tensor M_{n,m,p} has entries T[i,j,k] = delta_{i,k}? No.
        # M_{n,m,p} is defined such that <T, A \otimes B \otimes C> = Tr(C^T (A B))
        # So T[i,j,k] = 1 if i==k, else 0? 
        # <T, A \otimes B \otimes C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
        # Tr(C^T A B) = sum_{i,k} C[k,i] (A B)[i,k] = sum_{i,k} C[k,i] sum_j A[i,j] B[j,k]
        # = sum_{i,j,k} A[i,j] B[j,k] C[k,i] delta_{i,k}? No, just sum_{i,j,k} A[i,j] B[j,k] C[k,i]
        # So T[i,j,k] = 1 for all i,j,k.
        
        # Wait, let's re-verify with the schoolbook.
        # Schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
        # sum_col U[i*m+j, col] V[j*p+k, col] W[k*n+i, col]
        # For a given (i,j,k), the term is 1 if there exists col such that
        # col corresponds to (i,j,k). In schoolbook, each (i,j,k) has its own col.
        # So sum = 1 for each (i,j,k).
        # So T[i,j,k] = 1 for all i,j,k.
        
        # Let's verify numerically.
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Reconstruct C from U, V, W
        C_recon = np.zeros((n, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # C[i,k] += sum_j u[i,j] v[j,k] * w[k,i]? 
            # Actually the decomposition is C = sum_r (U_r \otimes V_r) \cdot W_r
            # Where U_r is n x m, V_r is m x p, W_r is p x n (diagonal for outer product)
            # C[i,k] = sum_r U_r[i,:] @ V_r[:,k] * W_r[k,i]
            for i in range(n):
                for k in range(p):
                    C_recon[i, k] += np.dot(u[i, :], v[:, k]) * w[k, i]
        
        if not np.allclose(C, C_recon, atol=1e-10):
            return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook))
        V_sb = np.zeros((m*p, rank_schoolbook))
        W_sb = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
