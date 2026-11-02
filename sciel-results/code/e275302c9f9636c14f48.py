import numpy as np
import time

def construct():
    n, m, p = 5, 5, 5
    rank = 99
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Block structure: split indices into [0,1,2] (block 0, size 3) and [3,4] (block 1, size 2)
    # A[i,j], B[j,k], C[i,k]
    # A block (row i-block, col j-block): A[i,j] where i in row-block, j in col-block
    # B block (col j-block, row k-block): B[j,k]
    # C block (row i-block, col k-block): C[i,k]
    
    def add_schoolbook(U_base, V_base, W_base, 
                       i_indices, j_indices, k_indices,
                       U_row_fn, V_col_fn, W_out_fn):
        """Add schoolbook decomposition for product of blocks.
        U_row_fn(j, i): returns row index in U for A[i,j]
        V_col_fn(j, k): returns col index in V for B[j,k]  
        W_out_fn(k, i): returns output index in W for C[i,k]
        """
        nonlocal col
        for j in j_indices:
            for i in i_indices:
                for k in k_indices:
                    U[U_row_fn(i, j), U_base + col - 0] = 1  # placeholder
                    col += 1
        # This is getting complicated. Let me do it more directly.
        pass
    
    # Let me just do it directly for each block product
    
    # Helper: add a schoolbook term for A[i_sub, j_sub] * B[j_sub, k_sub] -> C[i_sub, k_sub]
    # where i_sub is a list of i indices, j_sub list of j, k_sub list of k
    def add_term(i_list, j_list, k_list):
        """Add schoolbook decomposition for the product of blocks with given index ranges.
        This computes sum over j in j_list of A[i,j] * B[j,k] for i in i_list, k in k_list.
        Each (i,j,k) triple contributes one rank-1 term.
        Returns number of columns used.
        """
        nonlocal col
        count = 0
        for j in j_list:
            for i in i_list:
                for k in k_list:
                    # U index: A[i,j] -> row i*m + j
                    U[i * m + j, col] = 1
                    # V index: B[j,k] -> col j*p + k
                    V[j * p + k, col] = 1
                    # W index: C[i,k] -> row k*n + i (trace convention W[k*n+i, r])
                    W[k * n + i, col] = 1
                    col += 1
                    count += 1
        return count
    
    # Helper: add Strassen for 2x2 block product
    # A is 2x2 with rows i0,i1 and cols j0,j1
    # B is 2x2 with cols j0,j1 and rows k0,k1
    # C is 2x2 with rows i0,i1 and cols k0,k1
    def add_strassen_2x2(i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2 * 2x2 -> 2x2.
        Returns number of columns used (7).
        """
        nonlocal col
        # M1 = (A11 + A22)(B11 + B22)
        U[i0 * m + j0, col] = 1
        U[i1 * m + j1, col] = 1
        V[j0 * p + k0, col] = 1
        V[j1 * p + k1, col] = 1
        W[k0 * n + i0, col] = 1
        W[k0 * n + i1, col] = 1
        W[k1 * n + i0, col] = 1
        W[k1 * n + i1, col] = 1
        col += 1
        
        # M2 = (A21 + A22)B11
        U[i1 * m + j0, col] = 1
        U[i1 * m + j1, col] = 1
        V[j0 * p + k0, col] = 1
        W[k0 * n + i1, col] = 1
        W[k1 * n + i1, col] = 1
        col += 1
        
        # M3 = A11(B12 - B22)
        U[i0 * m + j0, col] = 1
        V[j0 * p + k1, col] = 1
        V[j1 * p + k1, col] = -1
        W[k1 * n + i0, col] = 1
        W[k1 * n + i1, col] = 1
        col += 1
        
        # M4 = A22(B21 - B11)
        U[i1 * m + j1, col] = 1
        V[j0 * p + k0, col] = 1
        V[j1 * p + k0, col] = -1
        W[k0 * n + i0, col] = 1
        W[k0 * n + i1, col] = 1
        col += 1
        
        # M5 = (A11 + A12)B22
        U[i0 * m + j0, col] = 1
        U[i0 * m + j1, col] = 1
        V[j1 * p + k1, col] = 1
        W[k1 * n + i0, col] = 1
        W[k1 * n + i1, col] = 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[i1 * m + j0, col] = 1
        U[i0 * m + j0, col] = -1
        V[j0 * p + k0, col] = 1
        V[j0 * p + k1, col] = 1
        W[k0 * n + i0, col] = 1
        W[k1 * n + i0, col] = 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[i0 * m + j1, col] = 1
        U[i1 * m + j1, col] = -1
        V[j1 * p + k0, col] = 1
        V[j1 * p + k1, col] = 1
        W[k0 * n + i1, col] = 1
        W[k1 * n + i1, col] = 1
        col += 1
        
        return 7
    
    # Block 0: indices [0,1,2], Block 1: indices [3,4]
    b0 = [0, 1, 2]
    b1 = [3, 4]
    
    # C11 = A11*B11 + A12*B21  (3x3 output)
    # A11: i in b0, j in b0 -> rows 0-2, cols 0-2
    # B11: j in b0, k in b0 -> cols 0-2, rows 0-2
    add_term(b0, b0, b0)  # A11*B11, rank 18
    
    # A12: i in b0, j in b1 -> rows 0-2, cols 3-4
    # B21: j in b1, k in b0 -> cols 3-4, rows 0-2
    add_term(b0, b1, b0)  # A12*B21, rank 18
    
    # C12 = A11*B12 + A12*B22  (3x2 output)
    # B12: j in b0, k in b1 -> cols 0-2, rows 3-4
    add_term(b0, b0, b1)  # A11*B12, rank 12
    
    # B22: j in b1, k in b1 -> cols 3-4, rows 3-4
    add_term(b0, b1, b1)  # A12*B22, rank 12
    
    # C21 = A21*B11 + A22*B21  (2x3 output)
    # A21: i in b1, j in b0 -> rows 3-4, cols 0-2
    add_term(b1, b0, b0)  # A21*B11, rank 12
    
    # A22: i in b1, j in b1 -> rows 3-4, cols 3-4
    add_term(b1, b1, b0)  # A22*B21, rank 12
    
    # C22 = A21*B12 + A22*B22  (2x2 output)
    # A21*B12: i in b1, j in b0, k in b1
    add_term(b1, b0, b1)  # A21*B12, rank 8
    
    # A22*B22: i in b1, j in b1, k in b1 -> use Strassen
    add_strassen_2x2(3, 4, 3, 4, 3, 4)  # A22*B22, rank 7
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
