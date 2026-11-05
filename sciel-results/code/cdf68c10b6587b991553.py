import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _strassen_2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    A is 2x2, B is 2x2, C is 2x2.
    A_flat = [A[0,0], A[0,1], A[1,0], A[1,1]]
    B_flat = [B[0,0], B[0,1], B[1,0], B[1,1]]
    C_flat = [C[0,0], C[0,1], C[1,0], C[1,1]]
    
    Strassen formulas:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U rows correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # V rows correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # W rows correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    
    U = [
        [1, 1, 1, 1, 1, -1, -1],  # A[0,0]
        [0, 0, 1, 0, 1, 0, -1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
        [1, 1, 0, 1, 0, 0, 1],    # A[1,1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
        [1, 0, 1, 0, 0, 1, 1],    # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],    # B[1,0]
        [1, 0, -1, 0, 1, 0, 1],   # B[1,1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1]
    ]
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Schoolbook decomposition for n x m times m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
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
    return U, V, W, rank

def _construct_block_decomposition():
    """
    Construct a rank-117 decomposition for 5x5x5 using 2+2+1 partition
    with Strassen for 2x2x2 sub-problems.
    """
    n, m, p = 5, 5, 5
    
    # Partition sizes
    part_n = [2, 2, 1]  # for A rows and C rows
    part_m = [2, 2, 1]  # for A cols and B rows
    part_p = [2, 2, 1]  # for B cols and C cols
    
    # Get Strassen decomposition
    strassen_U, strassen_V, strassen_W = _strassen_2x2()
    
    # Build full U, V, W matrices
    # We'll collect all terms first
    terms = []  # list of (U_block, V_block, W_block, rank)
    
    for i in range(3):  # block row index for A/C
        for j in range(3):  # block col index for A / block row index for B
            for k in range(3):  # block col index for B/C
                ni = part_n[i]
                mj = part_m[j]
                pk = part_p[k]
                
                # A block is ni x mj
                # B block is mj x pk
                # C block is ni x pk
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    U_block, V_block, W_block, rank = strassen_U, strassen_V, strassen_W, 7
                else:
                    # Use schoolbook
                    U_block, V_block, W_block, rank = _schoolbook_block(ni, mj, pk)
                
                terms.append((U_block, V_block, W_block, rank))
    
    # Calculate total rank
    total_rank = sum(t[3] for t in terms)
    
    # Build full matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    row_offset_A = 0  # offset in A rows
    row_offset_B = 0  # offset in B rows (same as A cols)
    row_offset_C = 0  # offset in C rows (same as A rows)
    col_offset_B = 0  # offset in B cols
    col_offset_C = 0  # offset in C cols
    
    # We need to track offsets for each block
    # A row offsets: cumulative sum of part_n
    A_row_offsets = [0]
    for sz in part_n:
        A_row_offsets.append(A_row_offsets[-1] + sz)
    
    # A col offsets (same as B row offsets): cumulative sum of part_m
    A_col_offsets = [0]
    for sz in part_m:
        A_col_offsets.append(A_col_offsets[-1] + sz)
    
    # B col offsets (same as C col offsets): cumulative sum of part_p
    B_col_offsets = [0]
    for sz in part_p:
        B_col_offsets.append(B_col_offsets[-1] + sz)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = part_n[i]
                mj = part_m[j]
                pk = part_p[k]
                
                U_block, V_block, W_block, rank = terms[i*9 + j*3 + k]
                
                # A block starts at row A_row_offsets[i], col A_col_offsets[j]
                a_row_start = A_row_offsets[i]
                a_col_start = A_col_offsets[j]
                
                # B block starts at row A_col_offsets[j] (since A cols = B rows), col B_col_offsets[k]
                b_row_start = A_col_offsets[j]
                b_col_start = B_col_offsets[k]
                
                # C block starts at row A_row_offsets[i], col B_col_offsets[k]
                c_row_start = A_row_offsets[i]
                c_col_start = B_col_offsets[k]
                
                # Fill U
                for r in range(ni * mj):
                    u_row = a_row_start + r // mj
                    u_col = a_col_start + r % mj
                    for c in range(rank):
                        U[u_row * m + u_col][col_offset + c] = U_block[r][c]
                
                # Fill V
                for r in range(mj * pk):
                    v_row = b_row_start + r // pk
                    v_col = b_col_start + r % pk
                    for c in range(rank):
                        V[v_row * p + v_col][col_offset + c] = V_block[r][c]
                
                # Fill W
                # W is indexed as W[k*n + i, r] per the prompt
                # C block is ni x pk, so C[row, col] where row in [0,ni), col in [0,pk)
                # The prompt says W[k*n+i, r] where k is col index, i is row index
                for r in range(ni * pk):
                    w_row = c_row_start + r // pk
                    w_col = c_col_start + r % pk
                    # W index: w_col * n + w_row
                    for c in range(rank):
                        W[w_col * n + w_row][col_offset + c] = W_block[r][c]
                
                col_offset += rank
    
    return {"U": U, "V": V, "W": W}, total_rank

def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify the decomposition using the W[k*n+i, r] convention.
    C[i,j] = sum_r W[j*n+i, r] * L_r * R_r
    where L_r = sum_{i',k'} U[i'*m+k', r] * A[i',k']
    and R_r = sum_{k',j'} V[k'*p+j', r] * B[k',j']
    """
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    L = U_arr @ A_flat  # Shape (rank,)
    R = V_arr @ B_flat  # Shape (rank,)
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            # W[j*n + i, r] per the prompt convention
            idx = j * n + i
            C_computed[i, j] = np.sum(W_arr[idx, :] * L * R)
    
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Try the block decomposition with Strassen
    try:
        result, rank = _construct_block_decomposition()
        
        # Verify before returning
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook_block(n, m, p)
    return {"U": U, "V": V, "W": W}
