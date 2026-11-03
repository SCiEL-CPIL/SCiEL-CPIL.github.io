"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 90. We aim for 84.
    # Blocks:
    # I1=[0,1], I2=[2]
    # J1=[0,1], J2=[2,3], J3=[4]
    # K1=[0,1], K2=[2,3], K3=[4,5]
    
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    # Strassen coefficients for output C_uv
    # M1, M2, M3, M4, M5, M6, M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Map (u, v) to list of (m_index, coeff)
    c_coeffs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # i_off, j_off, k_off are top-left indices of the 2x2x2 block
        
        # M1: (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 0:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M2: (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 1:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M3: A[i0,j0] * (B[j0,k1] - B[j1,k1])
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 2:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M4: A[i1,j1] * (B[j1,k0] - B[j0,k0])
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 3:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M5: (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 4:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M6: (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 5:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1
        
        # M7: (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        for (u, v), terms in c_coeffs.items():
            for (idx, coeff) in terms:
                if idx == 6:
                    i_out = i_off + u
                    k_out = k_off + v
                    W[k_out * n + i_out][col] += coeff
        col += 1

    # Iterate through blocks
    # I1=[0,1], I2=[2]
    # J1=[0,1], J2=[2,3], J3=[4]
    # K1=[0,1], K2=[2,3], K3=[4,5]
    
    # Block I1, K1
    add_strassen(0, 0, 0) # J1
    add_strassen(0, 2, 0) # J2
    for i in [0, 1]:
        for k in [0, 1]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I1, K2
    add_strassen(0, 0, 2) # J1
    add_strassen(0, 2, 2) # J2
    for i in [0, 1]:
        for k in [2, 3]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I1, K3
    add_strassen(0, 0, 4) # J1
    add_strassen(0, 2, 4) # J2
    for i in [0, 1]:
        for k in [4, 5]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K1
    for i in [2]:
        for j in [0, 1]:
            for k in [0, 1]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [0, 1]:
                add_schoolbook(i, j, k) # J2
        for k in [0, 1]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K2
    for i in [2]:
        for j in [0, 1]:
            for k in [2, 3]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [2, 3]:
                add_schoolbook(i, j, k) # J2
        for k in [2, 3]:
            add_schoolbook(i, 4, k) # J3
            
    # Block I2, K3
    for i in [2]:
        for j in [0, 1]:
            for k in [4, 5]:
                add_schoolbook(i, j, k) # J1
        for j in [2, 3]:
            for k in [4, 5]:
                add_schoolbook(i, j, k) # J2
        for k in [4, 5]:
            add_schoolbook(i, 4, k) # J3
            
    # Verify the decomposition using exact integer arithmetic
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    # Check: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = int(np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :]))
                if val != 1:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 0:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col_s] = 1
                    V_s[j*p+k][col_s] = 1
                    W_s[k*n+i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
