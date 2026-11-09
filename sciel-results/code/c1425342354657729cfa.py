"""Rewritten program: Strassen-block decomposition for 3x5x6 matrix multiplication."""
import time
import numpy as np
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a low-rank decomposition of the 3x5 @ 5x6 matrix multiplication tensor.
    
    Uses partitioning into blocks and Strassen's algorithm for 2x2 sub-multiplications.
    Total rank = 84 (vs schoolbook 90).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # Returns (U_block, V_block, W_block) each of shape (4, 7) for the local indices
    def strassen_2x2():
        """Strassen decomposition for 2x2 * 2x2 -> 2x2.
        
        Local indices: 
          A: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3
          B: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3
          C: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3
        
        7 products:
          M1 = (A00+A11)*(B00+B11)
          M2 = (A10+A11)*B00
          M3 = A00*(B01-B11)
          M4 = A11*(B10-B00)
          M5 = (A00+A01)*B11
          M6 = (A10-A00)*(B00+B01)
          M7 = (A01-A11)*(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (A00+A11)*(B00+B11)
        # A: indices 0 (A00), 3 (A11); B: indices 0 (B00), 3 (B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C00 += M1, C11 += M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)*B00
        # A: indices 2 (A10), 3 (A11); B: index 0 (B00)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C10 += M2, C11 -= M2
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00*(B01-B11)
        # A: index 0 (A00); B: indices 1 (B01), 3 (B11) with signs +1, -1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C01 += M3, C11 += M3
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11*(B10-B00)
        # A: index 3 (A11); B: indices 2 (B10), 0 (B00) with signs +1, -1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C00 += M4, C10 += M4
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)*B11
        # A: indices 0 (A00), 1 (A01); B: index 3 (B11)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C00 -= M5, C01 += M5
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        # A: indices 2 (A10), 0 (A00) with signs +1, -1; B: indices 0 (B00), 1 (B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C11 += M6
        W[3][5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        # A: indices 1 (A01), 3 (A11) with signs +1, -1; B: indices 2 (B10), 3 (B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C00 += M7
        W[0][6] = 1
        
        return U, V, W
    
    # Simple schoolbook for small blocks (size_a x size_b * size_b x size_c)
    def schoolbook_block(sa, sb, sc):
        """Simple schoolbook for sa x sb * sb x sc matrix multiplication.
        
        Local indices:
          A: sa rows, sb cols -> flat index = row*sb + col (total sa*sb)
          B: sb rows, sc cols -> flat index = row*sc + col (total sb*sc)
          C: sa rows, sc cols -> flat index = row*sc + col (total sa*sc)
        
        Returns U (sa*sb, sa*sb*sc), V (sb*sc, sa*sb*sc), W (sa*sc, sa*sb*sc)
        """
        rank = sa * sb * sc
        U = [[0]*rank for _ in range(sa*sb)]
        V = [[0]*rank for _ in range(sb*sc)]
        W = [[0]*rank for _ in range(sa*sc)]
        col = 0
        for i in range(sa):
            for j in range(sb):
                for k in range(sc):
                    U[i*sb + j][col] = 1
                    V[j*sc + k][col] = 1
                    W[i*sc + k][col] = 1
                    col += 1
        return U, V, W
    
    # Define partitions
    # I (rows of A, rows of C): split into {0,1} and {2}
    # J (cols of A, rows of B): split into {0,1}, {2,3}, {4}
    # K (cols of B, cols of C): split into {0,1}, {2,3}, {4,5}
    
    I_parts = [[0, 1], [2]]       # sizes 2, 1
    J_parts = [[0, 1], [2, 3], [4]]  # sizes 2, 2, 1
    K_parts = [[0, 1], [2, 3], [4, 5]]  # sizes 2, 2, 2
    
    # Build the global decomposition
    # We'll accumulate contributions to U, V, W
    # Global dimensions: U is (n*m=15, rank), V is (m*p=30, rank), W is (p*n=18, rank)
    
    # First, let's compute the total rank needed
    total_rank = 0
    sub_ranks = []
    for ii in range(len(I_parts)):
        for jj in range(len(J_parts)):
            for kk in range(len(K_parts)):
                sa = len(I_parts[ii])
                sb = len(J_parts[jj])
                sc = len(K_parts[kk])
                if sa == 2 and sb == 2 and sc == 2:
                    sub_ranks.append(7)  # Strassen
                else:
                    sub_ranks.append(sa * sb * sc)  # Schoolbook
                total_rank += sub_ranks[-1]
    
    # Allocate global matrices
    U_global = [[0] * total_rank for _ in range(n * m)]
    V_global = [[0] * total_rank for _ in range(m * p)]
    W_global = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in the blocks
    col_offset = 0
    for ii in range(len(I_parts)):
        for jj in range(len(J_parts)):
            for kk in range(len(K_parts)):
                sa = len(I_parts[ii])
                sb = len(J_parts[jj])
                sc = len(K_parts[kk])
                
                if sa == 2 and sb == 2 and sc == 2:
                    # Use Strassen
                    U_block, V_block, W_block = strassen_2x2()
                    r = 7
                else:
                    # Use schoolbook
                    U_block, V_block, W_block = schoolbook_block(sa, sb, sc)
                    r = sa * sb * sc
                
                # Map local indices to global indices
                I_idx = I_parts[ii]
                J_idx = J_parts[jj]
                K_idx = K_parts[kk]
                
                for local_col in range(r):
                    # U: local (i_local, j_local) -> global (I_idx[i_local]*m + J_idx[j_local])
                    for li in range(sa):
                        for lj in range(sb):
                            local_u_idx = li * sb + lj
                            global_u_idx = I_idx[li] * m + J_idx[lj]
                            U_global[global_u_idx][col_offset + local_col] += U_block[local_u_idx][local_col]
                    
                    # V: local (j_local, k_local) -> global (J_idx[j_local]*p + K_idx[k_local])
                    for lj in range(sb):
                        for lk in range(sc):
                            local_v_idx = lj * sc + lk
                            global_v_idx = J_idx[llj] * p + K_idx[lk]
                            V_global[global_v_idx][col_offset + local_col] += V_block[local_v_idx][local_col]
                    
                    # W: local (k_local, i_local) -> global (K_idx[k_local]*n + I_idx[i_local])
                    for lk in range(sc):
                        for li in range(sa):
                            local_w_idx = lk * sa + li
                            global_w_idx = K_idx[lk] * n + I_idx[li]
                            W_global[global_w_idx][col_offset + local_col] += W_block[local_w_idx][local_col]
                
                col_offset += r
    
    # Verify: check that the decomposition is correct
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Our decomposition: C[i,k] = sum_r U[i*m+j_col, r] * V[j_row*p+k, r] * W[k*n+i, r]
    # Actually: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the tensor decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    
    # Let's verify with a simple check using numpy
    try:
        U_np = np.array(U_global, dtype=float)
        V_np = np.array(V_global, dtype=float)
        W_np = np.array(W_global, dtype=float)
        
        # The matrix multiplication tensor T has T[i,j,k] = delta_{i,k} in the sense that
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor decomposition says: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
        # should equal 1 if (i,j,k) is a valid triple (it always is), and the structure must be correct.
        
        # Actually, the correct verification: for random A (n x m) and B (m x p),
        # C = A @ B should equal sum_r outer(U[:,r] reshaped to (n,m)) * outer(V[:,r] reshaped to (m,p)) 
        # applied via the W structure.
        
        # More precisely: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
        # No, that's not right either.
        
        # The standard tensor decomposition for matrix multiplication:
        # (A*B)[i,k] = sum_r U_{i,r}^{(A)} V_{k,r}^{(B)} ... 
        # 
        # Actually in this formulation:
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The decomposition says: for each r, we have a rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # where U[:,r] is indexed by (i,j), V[:,r] by (j,k), W[:,r] by (k,i)
        # 
        # The matrix multiplication tensor T[i,j,k] = 1 (for the identity structure)
        # satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # 
        # Wait, that's not quite right. Let me think again.
        # 
        # For matrix multiplication C = AB where A is n×m, B is m×p, C is n×p:
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        # 
        # The bilinear (actually trilinear in the tensor sense) decomposition:
        # We need: for all i, j, k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = δ (some structure)
        # 
        # Actually, the correct statement is that the "multiplication tensor" μ satisfies:
        # μ[i,j,k] = 1 if we're computing the (i,k) entry using A[i,j] and B[j,k]
        # But each (i,j,k) triple corresponds to exactly one product term.
        # 
        # The tensor rank decomposition means:
        # μ[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # where μ[i,j,k] = 1 for all valid (i,j,k) triples.
        # 
        # Wait no. μ[i,j,k] should be 1 because each term A[i,j]*B[j,k] appears exactly once.
        # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        
        # Let's verify this
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        error += abs(s - 1.0)
        
        if error > 1e-8:
            # Decomposition is incorrect, fall back to schoolbook
            return schoolbook()
        
    except Exception:
        return schoolbook()
    
    # Check if we're close to deadline
    if time.time() > deadline - 5.0:
        return {"U": U_global, "V": V_global, "W": W_global}
    
    return {"U": U_global, "V": V_global, "W": W_global}
# EVOLVE-BLOCK-END
