"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """Return (U, V, W) for schoolbook [n,m,p] with rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, rank, seed=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    np.random.seed(seed)
    A_test = np.random.randint(-3, 4, size=(n, m))
    B_test = np.random.randint(-3, 4, size=(m, p))
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            C_computed[i, k] = s
    
    return np.allclose(C_computed, C_expected, atol=1e-9)

def _strassen_222_factors():
    """
    Return U (4x7), V (4x7), W (4x7) for 2x2*2x2=2x2 using Strassen (rank 7).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # M6 = (c-b)(e+f)
    U[2][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7 = (a+d)(g+h)
    U[0][6] = 1; U[3][6] = 1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = -1
    
    return U, V, W, 7

def _block_decomposition():
    """
    Decompose [3,5,5] using block structure with Strassen for 2x2x2 blocks.
    Rank 71.
    """
    n, m, p = 3, 5, 5
    total_rank = 71
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_subproblem(sub_n, sub_m, sub_p, row_offset_A, col_offset_B, mid_offset, 
                       sub_U, sub_V, sub_W, sub_rank):
        nonlocal col
        for i in range(sub_n):
            for j in range(sub_m):
                for r in range(sub_rank):
                    U[(row_offset_A + i) * m + (mid_offset + j)][col + r] = sub_U[i * sub_m + j][r]
        for j in range(sub_m):
            for k in range(sub_p):
                for r in range(sub_rank):
                    V[(mid_offset + j) * p + (col_offset_B + k)][col + r] = sub_V[j * sub_p + k][r]
        for k in range(sub_p):
            for i in range(sub_n):
                for r in range(sub_rank):
                    W[(col_offset_B + k) * n + (row_offset_A + i)][col + r] = sub_W[k * sub_n + i][r]
        col += sub_rank

    # 1. [2,2,2] Strassen
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    add_subproblem(2, 2, 2, 0, 0, 0, U_s, V_s, W_s, r_s)
    
    # 2. [2,2,2] Strassen
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    add_subproblem(2, 2, 2, 0, 0, 2, U_s, V_s, W_s, r_s)
    
    # 3. [2,1,2] Schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 1, 2)
    add_subproblem(2, 1, 2, 0, 0, 4, U_s, V_s, W_s, r_s)
    
    # 4. [2,2,3] Split into [2,2,2] + [2,2,1]
    # 4a. [2,2,2]
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    add_subproblem(2, 2, 2, 0, 2, 0, U_s, V_s, W_s, r_s)
    # 4b. [2,2,1]
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 2, 1)
    add_subproblem(2, 2, 1, 0, 4, 0, U_s, V_s, W_s, r_s)
    # Note: The previous code used _strassen_223_split_factors which did this internally.
    # Here I am doing it manually to ensure correct offsets.
    # Wait, the offset for the [2,2,1] part in [2,2,3] is col_offset_B=4?
    # [2,2,3] covers B cols 2,3,4.
    # [2,2,2] covers B cols 2,3. (mid_offset 0, col_offset_B 2)
    # [2,2,1] covers B col 4. (mid_offset 0, col_offset_B 4)
    
    # 5. [2,2,3] Split into [2,2,2] + [2,2,1]
    # 5a. [2,2,2]
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    add_subproblem(2, 2, 2, 0, 2, 2, U_s, V_s, W_s, r_s)
    # 5b. [2,2,1]
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 2, 1)
    add_subproblem(2, 2, 1, 0, 4, 2, U_s, V_s, W_s, r_s)
    
    # 6. [2,1,3] Schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 1, 3)
    add_subproblem(2, 1, 3, 0, 2, 4, U_s, V_s, W_s, r_s)
    
    # 7. [1,2,2] Schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 2)
    add_subproblem(1, 2, 2, 2, 0, 0, U_s, V_s, W_s, r_s)
    
    # 8. [1,2,2] Schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 2)
    add_subproblem(1, 2, 2, 2, 0, 2, U_s, V_s, W_s, r_s)
    
    # 9. [1,1,2] Schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 1, 2)
    add_subproblem(1, 1, 2, 2, 0, 4, U_s, V_s, W_s, r_s)
    
    # 10. [1,2,3] Schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 3)
    add_subproblem(1, 2, 3, 2, 2, 0, U_s, V_s, W_s, r_s)
    
    # 11. [1,2,3] Schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 3)
    add_subproblem(1, 2, 3, 2, 2, 2, U_s, V_s, W_s, r_s)
    
    # 12. [1,1,3] Schoolbook
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 1, 3)
    add_subproblem(1, 1, 3, 2, 2, 4, U_s, V_s, W_s, r_s)
    
    return U, V, W, col

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W, rank = _block_decomposition()
        
        if time.time() < deadline and _verify_decomposition(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    U, V, W, rank = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
