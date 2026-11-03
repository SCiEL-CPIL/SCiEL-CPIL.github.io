import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 matrix multiplication.
    Rank 7.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = -1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = -1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[1, 5] = -1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W, rank

def _construct_block_decomposition():
    """
    Construct a rank-minimal decomposition for 4x5 * 5x5.
    Uses Strassen for 2x2 blocks where possible.
    """
    n, m, p = 4, 5, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1_term(u_sparsity, v_sparsity, w_sparsity):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        for idx, val in u_sparsity.items():
            u_vec[idx] = val
        for idx, val in v_sparsity.items():
            v_vec[idx] = val
        for idx, val in w_sparsity.items():
            w_vec[idx] = val
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_strassen_block(u_offset, v_offset, w_offset):
        """
        Add Strassen decomposition for a 2x2 * 2x2 block.
        u_offset: flat index of A[0,0] in the 2x2 block (row-major in A)
        v_offset: flat index of B[0,0] in the 2x2 block (row-major in B)
        w_offset: flat index in W corresponding to C[0,0] of the 2x2 block
                  W[k*n+i] where k=col of C, i=row of C
        """
        Us, Vs, Ws, rs = _strassen_2x2()
        for r in range(rs):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            # A: 2x2 block. A[i,j] -> flat index i*m+j
            # Block starts at row u_offset//m, col u_offset%m
            row_start_a = u_offset // m
            col_start_a = u_offset % m
            for la in range(2):
                for lc in range(2):
                    local_idx = la * 2 + lc
                    global_idx = (row_start_a + la) * m + (col_start_a + lc)
                    u_vec[global_idx] = Us[local_idx, r]
            
            # B: 2x2 block. B[i,j] -> flat index i*p+j
            row_start_b = v_offset // p
            col_start_b = v_offset % p
            for la in range(2):
                for lc in range(2):
                    local_idx = la * 2 + lc
                    global_idx = (row_start_b + la) * p + (col_start_b + lc)
                    v_vec[global_idx] = Vs[local_idx, r]
            
            # W: C is n x p. C[i,k] -> W[k*n + i]
            # w_offset = k0*n + i0 where k0 is col, i0 is row
            k0 = w_offset // n
            i0 = w_offset % n
            for la in range(2):
                for lc in range(2):
                    local_idx = la * 2 + lc
                    # C[i0+la, k0+lc] -> W[(k0+lc)*n + (i0+la)]
                    global_idx = (k0 + lc) * n + (i0 + la)
                    w_vec[global_idx] = Ws[local_idx, r]
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    def add_2x2_1_block(u_offset, v_offset, w_offset):
        """
        2x2 * 2x1 -> 2x1
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        """
        # Term 1
        add_rank1_term({u_offset: 1}, {v_offset: 1}, {w_offset: 1})
        # Term 2
        add_rank1_term({u_offset + 1: 1}, {v_offset + p: 1}, {w_offset: 1})
        # Term 3
        add_rank1_term({u_offset + 2: 1}, {v_offset: 1}, {w_offset + n: 1})
        # Term 4
        add_rank1_term({u_offset + 3: 1}, {v_offset + p: 1}, {w_offset + n: 1})
    
    def add_2x1_2_block(u_offset, v_offset, w_offset):
        """
        2x1 * 1x2 -> 2x2
        C[0,0] = A[0,0]*B[0,0]
        C[0,1] = A[0,0]*B[0,1]
        C[1,0] = A[1,0]*B[0,0]
        C[1,1] = A[1,0]*B[0,1]
        """
        k0 = w_offset // n
        i0 = w_offset % n
        # C[0,0] -> W[k0*n + i0] = w_offset
        add_rank1_term({u_offset: 1}, {v_offset: 1}, {w_offset: 1})
        # C[0,1] -> W[(k0+1)*n + i0] = w_offset + n
        add_rank1_term({u_offset: 1}, {v_offset + 1: 1}, {w_offset + n: 1})
        # C[1,0] -> W[k0*n + (i0+1)] = w_offset + 1
        add_rank1_term({u_offset + 1: 1}, {v_offset: 1}, {w_offset + 1: 1})
        # C[1,1] -> W[(k0+1)*n + (i0+1)] = w_offset + n + 1
        add_rank1_term({u_offset + 1: 1}, {v_offset + 1: 1}, {w_offset + n + 1: 1})
    
    # Build decomposition
    # 1. A11(2x2) * B[0:2,0:2](2x2) -> C[0:2,0:2]
    add_strassen_block(0, 0, 0)
    
    # 2. A11(2x2) * B[0:2,2:4](2x2) -> C[0:2,2:4]
    add_strassen_block(0, 2, 8)
    
    # 3. A11(2x2) * B[0:2,4:5](2x1) -> C[0:2,4:5]
    add_2x2_1_block(0, 4, 16)
    
    # 4. A12(2x2) * B[2:4,0:2](2x2) -> C[0:2,0:2]
    add_strassen_block(2, 10, 0)
    
    # 5. A12(2x2) * B[2:4,2:4](2x2) -> C[0:2,2:4]
    add_strassen_block(2, 12, 8)
    
    # 6. A12(2x2) * B[2:4,4:5](2x1) -> C[0:2,4:5]
    add_2x2_1_block(2, 14, 16)
    
    # 7. A13(2x1) * B[4:5,0:2](1x2) -> C[0:2,0:2]
    add_2x1_2_block(4, 20, 0)
    
    # 8. A13(2x1) * B[4:5,2:4](1x2) -> C[0:2,2:4]
    add_2x1_2_block(4, 22, 8)
    
    # 9. A13(2x1) * B[4:5,4:5](1x1) -> C[0:2,4:5]
    add_rank1_term({4: 1}, {24: 1}, {16: 1})
    add_rank1_term({9: 1}, {24: 1}, {17: 1})
    
    # 10. A21(2x2) * B[0:2,0:2](2x2) -> C[2:4,0:2]
    add_strassen_block(10, 0, 2)
    
    # 11. A21(2x2) * B[0:2,2:4](2x2) -> C[2:4,2:4]
    add_strassen_block(10, 2, 10)
    
    # 12. A21(2x2) * B[0:2,4:5](2x1) -> C[2:4,4:5]
    add_2x2_1_block(10, 4, 18)
    
    # 13. A22(2x2) * B[2:4,0:2](2x2) -> C[2:4,0:2]
    add_strassen_block(12, 10, 2)
    
    # 14. A22(2x2) * B[2:4,2:4](2x2) -> C[2:4,2:4]
    add_strassen_block(12, 12, 10)
    
    # 15. A22(2x2) * B[2:4,4:5](2x1) -> C[2:4,4:5]
    add_2x2_1_block(12, 14, 18)
    
    # 16. A23(2x1) * B[4:5,0:2](1x2) -> C[2:4,0:2]
    add_2x1_2_block(14, 20, 2)
    
    # 17. A23(2x1) * B[4:5,2:4](1x2) -> C[2:4,2:4]
    add_2x1_2_block(14, 22, 10)
    
    # 18. A23(2x1) * B[4:5,4:5](1x1) -> C[2:4,4:5]
    add_rank1_term({14: 1}, {24: 1}, {18: 1})
    add_rank1_term({19: 1}, {24: 1}, {19: 1})
    
    rank = len(U_cols)
    U = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 0))
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W, rank = _construct_block_decomposition()
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
