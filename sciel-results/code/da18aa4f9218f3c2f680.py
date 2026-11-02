"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]
_START_TIME = time.time()
_DEADLINE = 145.0  # seconds, with 5s safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition factors
    # Indices: U(i,j), V(j,k), W(k,i) where i,j in {0,1}, k in {0,1}
    # Global indices:
    # U: i*m + j
    # V: j*p + k_global
    # W: k_global*n + i
    # k_global = 2*t + k_local
    
    # Define the 7 Strassen terms
    # Each term is a list of (u_idx, v_idx, w_idx, coeff)
    # But since it's a tensor decomposition, each term contributes to one rank dimension.
    # For a specific term s and block t:
    # col = t * 7 + s
    # U entries: for (i, j, c) in u_entries: U[i*m+j][col] += c
    # V entries: for (j, k_local, c) in v_entries: V[j*p + 2*t + k_local][col] += c
    # W entries: for (i, k_local, c) in w_entries: W[(2*t + k_local)*n + i][col] += c
    
    # Strassen factors (U, V, W contributions)
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B01-B11) -> C01, C11
    # M4 = A11(B10-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    strassen = [
        # (U_entries, V_entries, W_entries)
        # U: (i, j, coeff)
        # V: (j, k_local, coeff)
        # W: (i, k_local, coeff)
        
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_entries, v_entries, w_entries = strassen[s]
            
            for i, j, c in u_entries:
                U[i * m + j][col] = c
                
            for j, k_local, c in v_entries:
                k_global = 2 * t + k_local
                V[j * p + k_global][col] = c
                
            for i, k_local, c in w_entries:
                k_global = 2 * t + k_local
                W[k_global * n + i][col] = c
    
    # Verify correctness with random test
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # C[i,k] = sum_s W[k*n+i, s] * (sum_j A[i,j] * U[i*m+j, s]) * (sum_j B[j,k] * V[j*p+k, s])
    C_decomp = np.zeros((n, p))
    for s in range(rank):
        A_proj = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_proj[i, j] = A_test[i, j] * U_arr[i*m+j, s]
        B_proj = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_proj[j, k] = B_test[j, k] * V_arr[j*p+k, s]
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += W_arr[k*n+i, s] * np.sum(A_proj[i, :] * B_proj[:, k])
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        # Fallback: return schoolbook decomposition (rank 128)
        rank_school = n * m * p  # 64
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
