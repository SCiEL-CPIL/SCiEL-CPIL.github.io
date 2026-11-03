import numpy as np
import random
import time

DIMENSIONS = [3, 4, 6]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                
                if j1 != j2 or k1 != k2 or i1 != i2:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err = abs(val)
                    max_err = max(max_err, err)
    
    return max_err <= tol

def _strassen_decompose_2x2x2():
    """Return U, V, W matrices for 2x2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 2: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Product 3: M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Product 4: M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Product 5: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Product 6: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Product 7: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_structured_decomposition():
    """Construct a rank-66 decomposition for 3x4x6 using block Strassen."""
    n, m, p = 3, 4, 6
    
    # Dimensions of factor matrices
    # U: (n*m, rank) = (12, 66)
    # V: (m*p, rank) = (24, 66)
    # W: (p*n, rank) = (18, 66)
    
    U = [[0] * 66 for _ in range(12)]
    V = [[0] * 66 for _ in range(24)]
    W = [[0] * 66 for _ in range(18)]
    
    col = 0
    
    # Get Strassen factors
    sU, sV, sW = _strassen_decompose_2x2x2()
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for:
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        """
        nonlocal col
        for r in range(7):
            # U indices: (i0+ui//2, j0+ui%2) -> row index (i0+ui//2)*4 + (j0+ui%2)
            for ui in range(4):
                val = sU[ui, r]
                if val != 0:
                    i_idx = i0 + ui // 2
                    j_idx = j0 + ui % 2
                    u_row = i_idx * 4 + j_idx
                    U[u_row][col] = val
            
            # V indices: (j0+vi//2, k0+vi%2) -> row index (j0+vi//2)*6 + (k0+vi%2)
            for vi in range(4):
                val = sV[vi, r]
                if val != 0:
                    j_idx = j0 + vi // 2
                    k_idx = k0 + vi % 2
                    v_row = j_idx * 6 + k_idx
                    V[v_row][col] = val
            
            # W indices: (k0+wi//2, i0+wi%2) -> row index (k0+wi//2)*3 + (i0+wi%2)
            for wi in range(4):
                val = sW[wi, r]
                if val != 0:
                    k_idx = k0 + wi // 2
                    i_idx = i0 + wi % 2
                    w_row = k_idx * 3 + i_idx
                    W[w_row][col] = val
            
            col += 1
    
    def add_simple_block(i0, j0, k0, n_rows, n_cols_b, n_cols_c):
        """
        Add schoolbook decomposition for:
        A[i0, j0:j0+n_cols_b] * B[j0:j0+n_cols_b, k0:k0+n_cols_c] -> C[i0, k0:k0+n_cols_c]
        Here n_rows is always 1 for the bottom block.
        """
        nonlocal col
        for j in range(n_cols_b):
            for k in range(n_cols_c):
                u_row = i0 * 4 + (j0 + j)
                v_row = (j0 + j) * 6 + (k0 + k)
                w_row = (k0 + k) * 3 + i0
                
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1

    # Top part: A[0:2, :] (2x4) * B[:, :] (4x6)
    # Split B into 3 blocks of width 2: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # For each block k (0, 1, 2):
    #   A_top (2x4) * B_block (4x2)
    #   Split A_top into A[0:2, 0:2] and A[0:2, 2:4]
    #   Split B_block into B[0:2, k*2:k*2+2] and B[2:4, k*2:k*2+2]
    
    for k in range(3):
        k0 = k * 2
        # Block 1: A[0:2, 0:2] * B[0:2, k0:k0+2]
        add_strassen_block(0, 0, k0)
        # Block 2: A[0:2, 2:4] * B[2:4, k0:k0+2]
        add_strassen_block(0, 2, k0)
        
    # Bottom part: A[2:3, :] (1x4) * B[:, :] (4x6)
    # Split B into 3 blocks of width 2
    for k in range(3):
        k0 = k * 2
        # A[2, 0:4] * B[0:4, k0:k0+2]
        # This is 1x4 * 4x2, rank 8
        add_simple_block(2, 0, k0, 1, 4, 2)
        
    # Verify col is 66
    # 3 k-blocks * (2 Strassen blocks * 7 rank) = 42
    # 3 k-blocks * (1 Simple block * 8 rank) = 24
    # Total 66
    
    if col != 66:
        # Fallback if something went wrong with counting
        pass
        
    return {"U": U, "V": V, "W": W}

def _find_low_rank_decomposition(max_rank=72, deadline=None):
    n, m, p = DIMENSIONS
    best_result = _construct_schoolbook()
    best_rank = n * m * p
    
    # Try structured decomposition first
    structured = _construct_structured_decomposition()
    if _verify_decomposition(structured["U"], structured["V"], structured["W"]):
        best_result = structured
        best_rank = len(structured["U"][0])
    
    return best_result

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Try structured decomposition
    structured = _construct_structured_decomposition()
    if _verify_decomposition(structured["U"], structured["V"], structured["W"]):
        return structured
    
    # Fallback to schoolbook
    return _construct_schoolbook()
