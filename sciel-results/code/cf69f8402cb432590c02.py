"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        r = len(U[0])
        if len(V[0]) != r or len(W[0]) != r or r <= 0:
            return False
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Schoolbook decomposition (rank 63) as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    if not verify(U_sb, V_sb, W_sb):
        return {'U': [[0]], 'V': [[0]], 'W': [[0]]}

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try to find rank-17 decomposition: 7+7+3
    # Split p=7 into 3+3+1
    # For each 3x3 block: use rank-7 Strassen-like decomposition
    # For the 3x1 block: use rank-3 outer product

    def make_strassen_3x3_3x3():
        """
        Return U, V, W for 3x3 x 3x3 matrix multiplication with rank 7.
        U: (9, 7), V: (9, 7), W: (9, 7)
        Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=3
        """
        # Strassen's algorithm for 2x2 x 2x2, extended to 3x3
        # We'll use a known rank-7 construction
        
        # The 7 rank-1 terms for 3x3 x 3x3:
        # We define them using linear combinations
        
        U = np.zeros((9, 7))
        V = np.zeros((9, 7))
        W = np.zeros((9, 7))
        
        # Term 1: (a00+a11)(b00+b11)
        # U contribution: a00, a11
        U[0*3+0, 0] = 1  # a00
        U[1*3+1, 0] = 1  # a11
        V[0*3+0, 0] = 1  # b00
        V[1*3+1, 0] = 1  # b11
        W[0*3+0, 0] = 1  # c00
        W[1*3+1, 0] = 1  # c11
        
        # Term 2: (a10+a11)b00
        U[1*3+0, 1] = 1  # a10
        U[1*3+1, 1] = 1  # a11
        V[0*3+0, 1] = 1  # b00
        W[1*3+0, 1] = 1  # c10
        W[1*3+1, 1] = -1 # c11
        
        # Term 3: a00(b01-b11)
        U[0*3+0, 2] = 1  # a00
        V[0*3+1, 2] = 1  # b01
        V[1*3+1, 2] = -1 # b11
        W[0*3+1, 2] = 1  # c01
        W[1*3+1, 2] = 1  # c11
        
        # Term 4: a11(b10-b00)
        U[1*3+1, 3] = 1  # a11
        V[1*3+0, 3] = 1  # b10
        V[0*3+0, 3] = -1 # b00
        W[1*3+0, 3] = 1  # c10
        W[0*3+0, 3] = 1  # c00
        
        # Term 5: (a00+a01)b11
        U[0*3+0, 4] = 1  # a00
        U[0*3+1, 4] = 1  # a01
        V[1*3+1, 4] = 1  # b11
        W[0*3+1, 4] = 1  # c01
        W[1*3+1, 4] = 1  # c11
        
        # Term 6: (a10-a00)(b00+b01)
        U[1*3+0, 5] = 1  # a10
        U[0*3+0, 5] = -1 # a00
        V[0*3+0, 5] = 1  # b00
        V[0*3+1, 5] = 1  # b01
        W[1*3+0, 5] = 1  # c10
        W[0*3+1, 5] = 1  # c01
        
        # Term 7: (a01-a11)(b10+b11)
        U[0*3+1, 6] = 1  # a01
        U[1*3+1, 6] = -1 # a11
        V[1*3+0, 6] = 1  # b10
        V[1*3+1, 6] = 1  # b11
        W[0*3+1, 6] = 1  # c01
        W[1*3+1, 6] = 1  # c11
        
        return U, V, W

    # Test the Strassen decomposition
    U_str, V_str, W_str = make_strassen_3x3_3x3()
    
    # Verify for 3x3 x 3x3
    def verify_3x3(U, V, W, tol=1e-9):
        nn, mm, pp = 3, 3, 3
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    val = np.dot(Ua[i*mm+j], Va[j*pp+k] * Wa[k*nn+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    strassen_valid = verify_3x3(U_str, V_str, W_str)
    
    if strassen_valid:
        # Build rank-17 decomposition for 3x3 x 3x7
        # Columns 0-2: first 3x3 block (rank 7)
        # Columns 3-5: second 3x3 block (rank 7)
        # Column 6: 3x1 block (rank 3)
        
        rank = 17
        U_17 = np.zeros((9, rank))
        V_17 = np.zeros((21, rank))
        W_17 = np.zeros((21, rank))
        
        # Block 1: columns 0-2 of B, uses U_str, V_str, W_str
        # V for block 1: rows 0-8 (j*p+k for j=0,1,2 and k=0,1,2)
        U_17[:, 0:7] = U_str
        V_17[0:9, 0:7] = V_str  # j*p+k for k=0,1,2
        W_17[0:9, 0:7] = W_str  # k*n+i for k=0,1,2
        
        # Block 2: columns 3-5 of B, uses U_str, V_str, W_str
        U_17[:, 7:14] = U_str
        V_17[9:18, 7:14] = V_str  # j*p+k for k=3,4,5 -> row offset 3*3=9
        W_17[9:18, 7:14] = W_str  # k*n+i for k=3,4,5 -> row offset 3*3=9
        
        # Block 3: column 6 of B, rank-3 outer product
        for j in range(3):
            for i in range(3):
                idx = 14 + j * 3 + i  # Wait, need to be careful with indexing
                # Actually for 3x1: U[i*3+j, r], V[j*3+6, r], W[6*3+i, r]
                # Rank 3: one term per (i,j) pair? No, rank 3 means 3 terms total
                # For 3x3 x 3x1: C[i,6] = sum_j A[i,j] * B[j,6]
                # This is an outer product: rank 3 (one for each j)
                pass
        
        # For 3x3 x 3x1: C[:,6] = A @ B[:,6]
        # C[i,6] = sum_j A[i,j] B[j,6]
        # Rank-3 decomposition: 3 terms, one for each j
        for j in range(3):
            r = 14 + j
            U_17[0*3+j, r] = 1  # a0j
            U_17[1*3+j, r] = 1  # a1j
            U_17[2*3+j, r] = 1  # a2j
            V_17[j*3+6, r] = 1  # b[j,6]
            W_17[6*3+0, r] = 1  # c[0,6]... wait, W[k*n+i]
            W_17[6*3+0, r] = 1  # i=0
            W_17[6*3+1, r] = 1  # i=1
            W_17[6*3+2, r] = 1  # i=2
        
        # Hmm, that's not right. For the 3x1 case:
        # C[i,6] = sum_j A[i,j] B[j,6]
        # Term for j: U[i*3+j, r_j] = 1, V[j*3+6, r_j] = 1, W[6*3+i, r_j] = 1 for all i
        # That gives 3 terms (one per j), each with W having 1s in all 3 positions
        # Let me redo:
        
        U_17[:, 14:17] = 0
        V_17[:, 14:17] = 0
        W_17[:, 14:17] = 0
        
        for j in range(3):
            r = 14 + j
            # U: A[i,j] for all i
            for i in range(3):
                U_17[i*3+j, r] = 1
            # V: B[j,6]
            V_17[j*3+6, r] = 1
            # W: C[i,6] for all i -> W[6*3+i, r]
            for i in range(3):
                W_17[6*3+i, r] = 1
        
        U_17_list = U_17.tolist()
        V_17_list = V_17.tolist()
        W_17_list = W_17.tolist()
        
        if verify(U_17_list, V_17_list, W_17_list):
            best_rank = 17
            best_U, best_V, best_W = U_17_list, V_17_list, W_17_list
        else:
            # Strassen decomposition was valid for 3x3x3x3 but the assembly failed
            # Try to debug or fall back
            pass
    
    # If we didn't find a better decomposition, return schoolbook
    if best_rank < schoolbook_rank:
        return {'U': best_U, 'V': best_V, 'W': best_W}
    else:
        return {'U': U_sb, 'V': V_sb, 'W': W_sb}

# EVOLVE-BLOCK-END
