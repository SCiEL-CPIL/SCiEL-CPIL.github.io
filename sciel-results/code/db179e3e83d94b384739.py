"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    def strassen_2x2x2(U, V, W, i0, j0, k0, col_offset):
        """Add Strassen rank-7 decomposition for 2x2x2 block at (i0,j0,k0)."""
        # 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U indices: [i0+i, j0+j] -> global row i0*5 + j0 + j (wait, U is indexed i*m+j)
        # U row index: (i0+i)*m + (j0+j)
        # V row index: (j0+j)*p + (k0+k)
        # W row index: (k0+k)*n + (i0+i)
        
        c = col_offset
        # Term 0: M1 = (A00+A11)(B00+B11)
        U[(i0+0)*m + (j0+0)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+0)*p + (k0+0)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[(k0+0)*n + (i0+0)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        
        # Term 1: M2 = (A10+A11)B00
        U[(i0+1)*m + (j0+0)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+0)*p + (k0+0)][c] += 1
        W[(k0+0)*n + (i0+1)][c] += 1
        c += 1
        
        # Term 2: M3 = A00(B01-B11)
        U[(i0+0)*m + (j0+0)][c] += 1
        V[(j0+0)*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + (i0+0)][c] += 1
        c += 1
        
        # Term 3: M4 = A11(B10-B00)
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+0)][c] += 1
        V[(j0+0)*p + (k0+0)][c] += -1
        W[(k0+0)*n + (i0+0)][c] += 1
        W[(k0+0)*n + (i0+1)][c] += 1
        c += 1
        
        # Term 4: M5 = (A00+A10)B11
        U[(i0+0)*m + (j0+0)][c] += 1
        U[(i0+1)*m + (j0+0)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[(k0+0)*n + (i0+0)][c] += -1
        W[(k0+1)*n + (i0+0)][c] += 1
        c += 1
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + (j0+0)][c] += 1
        U[(i0+0)*m + (j0+0)][c] += -1
        V[(j0+0)*p + (k0+0)][c] += 1
        V[(j0+0)*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        U[(i0+0)*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + (k0+0)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[(k0+0)*n + (i0+0)][c] += 1
        c += 1
        
        return c
    
    def schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, col_offset):
        """Add schoolbook decomposition for ni x nj x nk block at (i0,j0,k0)."""
        c = col_offset
        for di in range(ni):
            for dj in range(nj):
                for dk in range(nk):
                    U[(i0+di)*m + (j0+dj)][c] += 1
                    V[(j0+dj)*p + (k0+dk)][c] += 1
                    W[(k0+dk)*n + (i0+di)][c] += 1
                    c += 1
        return c
    
    # Build the decomposition
    # Partition: i: [0,1] (one block of size 2)
    #             j: [0,1], [2,3], [4] 
    #             k: [0,1], [2,3], [4,5]
    
    # For 2x2x2 blocks use Strassen (rank 7)
    # For 2x1x2 blocks use schoolbook (rank 4)
    
    # Determine which blocks to use
    j_blocks = [(0, 2), (2, 2), (4, 1)]  # (start, size)
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    total_rank = 0
    for j0, js in j_blocks:
        for k0, ks in k_blocks:
            if js == 2 and ks == 2:
                total_rank += 7  # Strassen
            else:
                total_rank += 2 * js * ks  # schoolbook for 2 x js x ks
    
    # Allocate matrices
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, js in j_blocks:
        for k0, ks in k_blocks:
            if js == 2 and ks == 2:
                col = strassen_2x2x2(U, V, W, 0, j0, k0, col)
            else:
                col = schoolbook_block(U, V, W, 0, j0, k0, 2, js, ks, col)
    
    # Verify
    def verify_exact(U_mat, V_mat, W_mat, rank_val):
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank_val):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_exact(U, V, W, rank):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][c] = 1
                    V_s[j * p + k][c] = 1
                    W_s[k * n + i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
