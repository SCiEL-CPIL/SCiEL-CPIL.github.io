import time
import numpy as np

# Dimensions for the matrix multiplication: (n x m) * (m x p)
N, M, P = 2, 4, 8

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Total rank using Strassen on 2x2 blocks
    # 8 blocks (2 in middle dim, 4 in last dim) * 7 multiplications per Strassen
    RANK = 56
    
    # Initialize factor matrices as lists of lists of zeros
    # U: (N*M) x RANK
    # V: (M*P) x RANK
    # W: (P*N) x RANK
    U = [[0.0] * RANK for _ in range(N * M)]
    V = [[0.0] * RANK for _ in range(M * P)]
    W = [[0.0] * RANK for _ in range(P * N)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We iterate over the 8 blocks
    # u_idx: index in the middle dimension block (0 or 1)
    # v_idx: index in the last dimension block (0, 1, 2, or 3)
    
    for u_idx in range(2):
        for v_idx in range(4):
            base_r = (u_idx * 4 + v_idx) * 7
            
            # Indices in A (N x M)
            # A[i, j] where i in 0..1, j in 0..3
            # Block is A[:, 2*u_idx : 2*u_idx+2]
            a_11 = 0 * M + (2 * u_idx)
            a_12 = 0 * M + (2 * u_idx + 1)
            a_21 = 1 * M + (2 * u_idx)
            a_22 = 1 * M + (2 * u_idx + 1)
            
            # Indices in B (M x P)
            # B[j, k] where j in 0..3, k in 0..7
            # Block is B[2*u_idx : 2*u_idx+2, :]
            b_11 = (2 * u_idx) * P + (2 * v_idx)
            b_12 = (2 * u_idx) * P + (2 * v_idx + 1)
            b_21 = (2 * u_idx + 1) * P + (2 * v_idx)
            b_22 = (2 * u_idx + 1) * P + (2 * v_idx + 1)
            
            # Indices in C (N x P) -> W (P x N)
            # C[i, k] where i in 0..1, k in 0..7
            # W index is k * N + i
            # Block is C[:, 2*v_idx : 2*v_idx+2]
            w_11 = (2 * v_idx) * N + 0
            w_12 = (2 * v_idx + 1) * N + 0
            w_21 = (2 * v_idx) * N + 1
            w_22 = (2 * v_idx + 1) * N + 1
            
            # M1
            U[a_11][base_r + 0] += 1.0
            U[a_22][base_r + 0] += 1.0
            V[b_11][base_r + 0] += 1.0
            V[b_22][base_r + 0] += 1.0
            W[w_11][base_r + 0] += 1.0
            W[w_22][base_r + 0] += 1.0
            
            # M2
            U[a_21][base_r + 1] += 1.0
            U[a_22][base_r + 1] += 1.0
            V[b_11][base_r + 1] += 1.0
            W[w_21][base_r + 1] += 1.0
            W[w_22][base_r + 1] -= 1.0
            
            # M3
            U[a_11][base_r + 2] += 1.0
            V[b_12][base_r + 2] += 1.0
            V[b_22][base_r + 2] -= 1.0
            W[w_12][base_r + 2] += 1.0
            W[w_22][base_r + 2] += 1.0
            
            # M4
            U[a_22][base_r + 3] += 1.0
            V[b_21][base_r + 3] += 1.0
            V[b_11][base_r + 3] -= 1.0
            W[w_11][base_r + 3] += 1.0
            W[w_21][base_r + 3] += 1.0
            
            # M5
            U[a_11][base_r + 4] += 1.0
            U[a_12][base_r + 4] += 1.0
            V[b_22][base_r + 4] += 1.0
            W[w_11][base_r + 4] -= 1.0
            W[w_12][base_r + 4] += 1.0
            
            # M6
            U[a_21][base_r + 5] += 1.0
            U[a_11][base_r + 5] -= 1.0
            V[b_11][base_r + 5] += 1.0
            V[b_12][base_r + 5] += 1.0
            W[w_22][base_r + 5] += 1.0
            
            # M7
            U[a_12][base_r + 6] += 1.0
            U[a_22][base_r + 6] -= 1.0
            V[b_21][base_r + 6] += 1.0
            V[b_22][base_r + 6] += 1.0
            W[w_11][base_r + 6] += 1.0

    # Verification
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute tensor T
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Expected tensor
    # T[a, b, c] = 1 if a=i*M+j, b=j*P+k, c=k*N+i
    # a in 0..7, b in 0..31, c in 0..15
    expected = np.zeros((N*M, M*P, P*N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                expected[a, b, c] = 1.0
                
    if not np.allclose(T, expected, atol=1e-9):
        # Fallback to schoolbook if Strassen construction failed (should not happen)
        rank = N * M * P
        U_f = [[0.0] * rank for _ in range(N * M)]
        V_f = [[0.0] * rank for _ in range(M * P)]
        W_f = [[0.0] * rank for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U_f[i * M + j][col] = 1.0
                    V_f[j * P + k][col] = 1.0
                    W_f[k * N + i][col] = 1.0
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}
        
    return {"U": U, "V": V, "W": W}
