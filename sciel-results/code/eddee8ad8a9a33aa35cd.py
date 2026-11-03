"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe (150s wall-clock)
    
    n, m, p = DIMENSIONS
    
    # Laderman's 3x3x3 decomposition (rank 23)
    # The 23 multiplications for 3x3 matrix multiplication
    laderman_U = [
        # Each entry: (i,j) -> list of (mul_idx, coeff)
        [(0,1),(1,1),(2,1)],       # U[0,0]
        [(3,1),(4,1)],              # U[0,1]
        [(5,1),(6,1)],              # U[0,2]
        [(7,1),(8,1)],              # U[1,0]
        [(9,1),(10,1),(11,1)],     # U[1,1]
        [(12,1),(13,1)],            # U[1,2]
        [(14,1),(15,1)],            # U[2,0]
        [(16,1),(17,1)],            # U[2,1]
        [(18,1),(19,1),(20,1)],    # U[2,2]
    ]
    laderman_V = [
        [(0,1),(1,1),(2,1)],       # V[0,0]
        [(3,1),(4,1)],              # V[0,1]
        [(5,1),(6,1)],              # V[0,2]
        [(7,1),(8,1)],              # V[1,0]
        [(9,1),(10,1),(11,1)],     # V[1,1]
        [(12,1),(13,1)],            # V[1,2]
        [(14,1),(15,1)],            # V[2,0]
        [(16,1),(17,1)],            # V[2,1]
        [(18,1),(19,1),(20,1)],    # V[2,2]
    ]
    laderman_W = [
        [(0,1),(1,1),(2,1)],       # W[0,0]
        [(3,1),(4,1)],              # W[0,1]
        [(5,1),(6,1)],              # W[0,2]
        [(7,1),(8,1)],              # W[1,0]
        [(9,1),(10,1),(11,1)],     # W[1,1]
        [(12,1),(13,1)],            # W[1,2]
        [(14,1),(15,1)],            # W[2,0]
        [(16,1),(17,1)],            # W[2,1]
        [(18,1),(19,1),(20,1)],    # W[2,2]
    ]
    
    # Actually, let me use a known correct Laderman decomposition
    # Laderman 1976: rank 23 for 3x3x3
    # Using the standard formulation:
    # The 23 terms are indexed 0..22
    
    # Let me use a verified Laderman decomposition
    # U[i,j] coefficients for each of 23 multiplications
    L_U = np.zeros((3,3,23), dtype=object)
    L_V = np.zeros((3,3,23), dtype=object)
    L_W = np.zeros((3,3,23), dtype=object)
    
    # Laderman's algorithm - standard form
    # Multiplication indices:
    # 0: A00*B00
    # 1: A00*B01
    # 2: A01*B00
    # 3: A01*B01
    # 4: A02*B00
    # 5: A02*B01
    # 6: A00*B02
    # 7: A01*B02
    # 8: A02*B02
    # 9: A10*B10
    # 10: A10*B11
    # 11: A11*B10
    # 12: A11*B11
    # 13: A12*B10
    # 14: A12*B11
    # 15: A10*B12
    # 16: A11*B12
    # 17: A12*B12
    # 18: A20*B20
    # 19: A20*B21
    # 20: A21*B20
    # 21: A21*B21
    # 22: A22*B20 + A20*B22
    
    # Actually this isn't quite right. Let me use a proper Laderman.
    # The correct Laderman 3x3x3 rank-23 decomposition:
    
    # I'll use a different approach: just use the known structure
    # For 3x3x3, the tensor is sum over i,j,k of e_i ⊗ e_j ⊗ e_k * T[i,j,k]
    # where T[i,j,k] = 1 if we're computing C[i,j] += A[i,k]*B[k,j]
    
    # Let me just build a good base decomposition and reduce it.
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Build candidate: Strassen tiling
    # i_blocks: [0,1], [2]
    # j_blocks: [0,1], [2,3], [4]
    # k_blocks: [0,1], [2,3], [4,5], [6]
    
    max_rank = 120
    U1 = [[0] * max_rank for _ in range(n * m)]
    V1 = [[0] * max_rank for _ in range(m * p)]
    W1 = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    i_blocks = [(0, 1), (2, 2)]
    j_blocks = [(0, 1), (2, 3), (4, 4)]
    k_blocks = [(0, 1), (2, 3), (4, 5), (6, 6)]
    
    for (i0, i1) in i_blocks:
        for (j0, j1) in j_blocks:
            for (k0, k1) in k_blocks:
                i_size = i1 - i0 + 1
                j_size = j1 - j0 + 1
                k_size = k1 - k0 + 1
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    col = add_strassen(U1, V1, W1, i0, j0, k0, col)
                else:
                    for i in range(i0, i1 + 1):
                        for j in range(j0, j1 + 1):
                            for k in range(k0, k1 + 1):
                                U1[i * m + j][col] = 1
                                V1[j * p + k][col] = 1
                                W1[k * n + i][col] = 1
                                col += 1
    
    U1 = [row[:col] for row in U1]
    V1 = [row[:col] for row in V1]
    W1 = [row[:col] for row in W1]
    
    # Build candidate 2: different tiling
    # i_blocks: [0,1,2] (use 3x3 Laderman-like)
    # Actually let's try: partition into 3+2 for i, 3+2 for j, 3+4 for k
    # Use Strassen where we have 2x2x2, schoolbook otherwise
    # i_blocks: [0,1], [2]  (sizes 2,1)
    # j_blocks: [0,1], [2,3], [4]  (sizes 2,2,1)
    # k_blocks: [0,1,2], [3,4], [5,6]  (sizes 3,2,2)
    
    max_rank2 = 120
    U2 = [[0] * max_rank2 for _ in range(n * m)]
    V2 = [[0] * max_rank2 for _ in range(m * p)]
    W2 = [[0] * max_rank2 for _ in range(p * n)]
    
    col = 0
    i_blocks2 = [(0, 1), (2, 2)]
    j_blocks2 = [(0, 1), (2, 3), (4, 4)]
    k_blocks2 = [(0, 2), (3, 4), (5, 6)]
    
    for (i0, i1) in i_blocks2:
        for (j0, j1) in j_blocks2:
            for (k0, k1) in k_blocks2:
                i_size = i1 - i0 + 1
                j_size = j1 - j0 + 1
                k_size = k1 - k0 + 1
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    col = add_strassen(U2, V2, W2, i0, j0, k0, col)
                else:
                    for i in range(i0, i1 + 1):
                        for j in range(j0, j1 + 1):
                            for k in range(k0, k1 + 1):
                                U2[i * m + j][col] = 1
                                V2[j * p + k][col] = 1
                                W2[k * n + i][col] = 1
                                col += 1
    
    U2 = [row[:col] for row in U2]
    V2 = [row[:col] for row in V2]
    W2 = [row[:col] for row in W2]
    
    # Candidate 3: Another tiling
    # i: [0,1],[2] ; j: [0,1],[2,3],[4] ; k: [0,1],[2,3,4],[5,6]
    max_rank3 = 120
    U3 = [[0] * max_rank3 for _ in range(n * m)]
    V3 = [[0] * max_rank3 for _ in range(m * p)]
    W3 = [[0] * max_rank3 for _ in range(p * n)]
    
    col = 0
    i_blocks3 = [(0, 1), (2, 2)]
    j_blocks3 = [(0, 1), (2, 3), (4, 4)]
    k_blocks3 = [(0, 1), (2, 4), (5, 6)]
    
    for (i0, i1) in i_blocks3:
        for (j0, j1) in j_blocks3:
            for (k0, k1) in k_blocks3:
                i_size = i1 - i0 + 1
                j_size = j1 - j0 + 1
                k_size = k1 - k0 + 1
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    col = add_strassen(U3, V3, W3, i0, j0, k0, col)
                else:
                    for i in range(i0, i1 + 1):
                        for j in range(j0, j1 + 1):
                            for k in range(k0, k1 + 1):
                                U3[i * m + j][col] = 1
                                V3[j * p + k][col] = 1
                                W3[k * n + i][col] = 1
                                col += 1
    
    U3 = [row[:col] for row in U3]
    V3 = [row[:col] for row in V3]
    W3 = [row[:col] for row in W3]
    
    # Candidate 4: Pure schoolbook as fallback
    U4 = [[0] * (n * m * p) for _ in range(n * m)]
    V4 = [[0] * (n * m * p) for _ in range(m * p)]
    W4 = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U4[i * m + j][col] = 1
                V4[j * p + k][col] = 1
                W4[k * n + i][col] = 1
                col += 1
    
    candidates = [(U1, V1, W1), (U2, V2, W2), (U3, V3, W3), (U4, V4, W4)]
    
    def build_tensor_matrix(U, V, W, n, m, p):
        """Build the (n*m*p) x rank matrix where entry (i,j,k; r) = U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]"""
        r = len(U[0])
        M = np.zeros((n * m * p, r), dtype=object)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        M[idx, r_idx] = U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    idx += 1
        return M
    
    def exact_rank_and_pivots(M, deadline):
        """Exact Gaussian elimination to find rank and pivot columns"""
        rows = M.shape[0]
        cols = M.shape[1]
        A = [list(M[row]) for row in range(rows)]
        pivot_cols = []
        current_row = 0
        
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                break
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1
        
        return pivot_cols
    
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(len(U[0])):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True
    
    best_result = None
    best_rank = float('inf')
    
    for U_c, V_c, W_c in candidates:
        if time.time() > deadline:
            break
        
        r = len(U_c[0])
        
        # First verify the base decomposition is correct
        if not verify(U_c, V_c, W_c, n, m, p):
            continue
        
        if r < best_rank:
            best_result = {"U": U_c, "V": V_c, "W": W_c}
            best_rank = r
        
        # Try to reduce rank
        M = build_tensor_matrix(U_c, V_c, W_c, n, m, p)
        
        # Use float SVD to estimate if reduction is possible
        M_float = np.array(M, dtype=float)
        try:
            sv = np.linalg.svd(M_float, compute_uv=False)
            tol = max(M_float.shape) * np.finfo(float).eps * (sv[0] if sv[0] > 0 else 1.0)
            num_rank = int(np.sum(sv > tol))
        except:
            num_rank = r
        
        if num_rank < r:
            # Try exact reduction
            pivot_cols = exact_rank_and_pivots(M, deadline)
            exact_rank = len(pivot_cols)
            
            if exact_rank < r and exact_rank < best_rank:
                pivot_set = set(pivot_cols)
                keep_cols = [c for c in range(r) if c not in pivot_set]
                U_new = [[U_c[i][c] for c in keep_cols] for i in range(n * m)]
                V_new = [[V_c[i][c] for c in keep_cols] for i in range(m * p)]
                W_new = [[W_c[i][c] for c in keep_cols] for i in range(p * n)]
                
                if verify(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = len(keep_cols)
    
    if best_result is not None:
        return best_result
    
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
