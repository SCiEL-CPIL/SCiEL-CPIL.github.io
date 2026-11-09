import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    N_U = n * m  # 6
    N_V = m * p  # 15
    N_W = p * n  # 10
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Precompute indices for verification
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        terms = U_arr[u_indices] * V_arr[v_indices] * W_arr[w_indices]
        sums = terms.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-9)
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build 28-rank decomposition: Strassen(7) + Strassen(7) + schoolbook_2x2_2x1(4) + schoolbook_2x1_1x5(10)
    R = 28
    U = [[0] * R for _ in range(N_U)]
    V = [[0] * R for _ in range(N_V)]
    W = [[0] * R for _ in range(N_W)]
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
        """
        Strassen for 2x2 * 2x2.
        a_rows, a_cols: row/col indices into A (2x3 matrix)
        b_rows, b_cols: row/col indices into B (3x5 matrix)
        c_rows, c_cols: row/col indices into C (2x5 matrix)
        """
        a11 = a_rows[0] * m + a_cols[0]
        a12 = a_rows[0] * m + a_cols[1]
        a21 = a_rows[1] * m + a_cols[0]
        a22 = a_rows[1] * m + a_cols[1]
        
        b11 = b_rows[0] * p + b_cols[0]
        b12 = b_rows[0] * p + b_cols[1]
        b21 = b_rows[1] * p + b_cols[0]
        b22 = b_rows[1] * p + b_cols[1]
        
        c11 = c_cols[0] * n + c_rows[0]
        c12 = c_cols[1] * n + c_rows[0]
        c21 = c_cols[0] * n + c_rows[1]
        c22 = c_cols[1] * n + c_rows[1]
        
        c = start_col
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        U[a11][c] += 1; U[a22][c] += 1
        V[b11][c] += 1; V[b22][c] += 1
        W[c11][c] += 1; W[c22][c] += 1
        c += 1
        
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        U[a21][c] += 1; U[a22][c] += 1
        V[b11][c] += 1
        W[c21][c] += 1; W[c22][c] -= 1
        c += 1
        
        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        U[a11][c] += 1
        V[b12][c] += 1; V[b22][c] -= 1
        W[c12][c] += 1; W[c22][c] += 1
        c += 1
        
        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        U[a22][c] += 1
        V[b21][c] += 1; V[b11][c] -= 1
        W[c11][c] += 1; W[c21][c] += 1
        c += 1
        
        # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
        U[a11][c] += 1; U[a12][c] += 1
        V[b22][c] += 1
        W[c11][c] -= 1; W[c12][c] += 1
        c += 1
        
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        U[a21][c] += 1; U[a11][c] -= 1
        V[b11][c] += 1; V[b12][c] += 1
        W[c22][c] += 1
        c += 1
        
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        U[a12][c] += 1; U[a22][c] -= 1
        V[b21][c] += 1; V[b22][c] += 1
        W[c11][c] += 1
        c += 1
        
        return c
    
    def add_sb_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
        """Schoolbook 2x2 * 2x1 -> 2x1, rank 4."""
        c = start_col
        for i in range(2):
            for k in range(2):
                ui = a_rows[i] * m + a_cols[k]
                vi = b_rows[k] * p + b_col
                wi = c_col * n + c_rows[i]
                U[ui][c] = 1
                V[vi][c] = 1
                W[wi][c] = 1
                c += 1
        return c
    
    def add_sb_2x1_1x5(a_rows, a_col, b_row, b_cols, c_rows, start_col):
        """Schoolbook 2x1 * 1x5 -> 2x5, rank 10."""
        c = start_col
        for i in range(2):
            for k in range(5):
                ui = a_rows[i] * m + a_col
                vi = b_row * p + b_cols[k]
                wi = b_cols[k] * n + c_rows[i]
                U[ui][c] = 1
                V[vi][c] = 1
                W[wi][c] = 1
                c += 1
        return c
    
    # Decomposition: C = A[:,0:2]*B[0:2,:] + A[:,2]*B[2,:]
    # A[:,0:2]*B[0:2,:] split B[0:2,:] into B[0:2,0:2], B[0:2,2:4], B[0:2,4]
    
    # Part 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] via Strassen
    col = add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    
    # Part 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] via Strassen
    col = add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], col)
    
    # Part 3: A[0:2,0:2] * B[0:2,4] -> C[0:2,4] via schoolbook
    col = add_sb_2x2_2x1([0,1], [0,1], [0,1], 4, [0,1], 4, col)
    
    # Part 4: A[0:2,2] * B[2,0:5] -> C[0:2,0:5] via schoolbook
    col = add_sb_2x1_1x5([0,1], 2, 2, [0,1,2,3,4], [0,1], col)
    
    assert col == R, f"Expected {R}, got {col}"
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return get_schoolbook()
