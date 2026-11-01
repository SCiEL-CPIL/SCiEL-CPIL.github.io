"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We assume n=m=p=4 for this specific task
    # Generate Strassen decomposition for 4x4
    
    # Base case: 1x1
    # U, V, W are lists of 1D arrays (flattened matrices)
    # Or just scalars for 1x1
    U_base = [np.array([1])]
    V_base = [np.array([1])]
    W_base = [np.array([1])]
    
    # Strassen coefficients for A, B, C blocks
    # A_k = sum alpha[k][r][c] * A_rc
    # B_k = sum beta[k][r][c] * B_rc
    # C_rc = sum gamma[k][r][c] * M_k  (Note: gamma is indexed by k and block rc)
    
    # Indices: k in 0..6, r,c in 0..1
    alpha = [
        [[1, 0], [0, 1]],  # k=0: A00 + A11
        [[0, 0], [1, 1]],  # k=1: A10 + A11
        [[1, 0], [0, 0]],  # k=2: A00
        [[0, 0], [0, 1]],  # k=3: A11
        [[1, 1], [0, 0]],  # k=4: A00 + A01
        [[0, 0], [1, -1]], # k=5: A10 - A00
        [[0, 1], [0, -1]]  # k=6: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # k=0: B00 + B11
        [[1, 0], [0, 0]],  # k=1: B00
        [[0, 0], [1, -1]], # k=2: B10 - B11
        [[0, 1], [0, -1]], # k=3: B01 - B00
        [[0, 0], [0, 1]],  # k=4: B11
        [[1, 1], [0, 0]],  # k=5: B00 + B01
        [[0, 0], [1, 1]]   # k=6: B10 + B11
    ]
    
    # gamma[k][r][c] is the coefficient of M_k in C_rc
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    gamma = [
        [[1, 0], [0, 1]],  # k=0: C00 (1), C11 (1)
        [[0, 0], [1, -1]], # k=1: C10 (1), C11 (-1)
        [[0, 1], [0, 1]],  # k=2: C01 (1), C11 (1)
        [[1, 0], [1, 0]],  # k=3: C00 (1), C10 (1)
        [[-1, 1], [0, 0]], # k=4: C00 (-1), C01 (1)
        [[0, 0], [0, 1]],  # k=5: C11 (1)
        [[-1, 0], [0, 0]]  # k=6: C00 (-1)
    ]
    
    # Recursive function to generate U, V, W for size N
    def generate_strassen(N):
        if N == 1:
            return U_base, V_base, W_base
        
        # Generate for N/2
        U_half, V_half, W_half = generate_strassen(N // 2)
        
        U_full = []
        V_full = []
        W_full = []
        
        for k in range(7):
            for t in range(len(U_half)):
                u_half = U_half[t]
                v_half = V_half[t]
                w_half = W_half[t]
                
                u_full = np.zeros((N, N))
                v_full = np.zeros((N, N))
                w_full = np.zeros((N, N))
                
                for r in range(2):
                    for c in range(2):
                        block_size = N // 2
                        r_start = r * block_size
                        c_start = c * block_size
                        r_end = r_start + block_size
                        c_end = c_start + block_size
                        
                        u_full[r_start:r_end, c_start:c_end] = alpha[k][r][c] * u_half
                        v_full[r_start:r_end, c_start:c_end] = beta[k][r][c] * v_half
                        w_full[r_start:r_end, c_start:c_end] = gamma[k][r][c] * w_half
                
                U_full.append(u_full)
                V_full.append(v_full)
                W_full.append(w_full)
                
        return U_full, V_full, W_full
    
    U_list, V_list, W_list = generate_strassen(4)
    
    rank = len(U_list)
    
    # Convert to the required format
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U_mat = np.zeros((n * m, rank))
    V_mat = np.zeros((m * p, rank))
    W_mat = np.zeros((p * n, rank))
    
    for t in range(rank):
        # Flatten U_list[t]
        # Row index in U_mat is i*m + j
        # U_list[t] is N x N. i, j in 0..N-1
        for i in range(n):
            for j in range(m):
                U_mat[i * m + j, t] = U_list[t][i, j]
                
        for j in range(m):
            for k in range(p):
                V_mat[j * p + k, t] = V_list[t][j, k]
                
        for k in range(p):
            for i in range(n):
                W_mat[k * n + i, t] = W_list[t][k, i] # Note: W convention W[k*n+i]
                
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
