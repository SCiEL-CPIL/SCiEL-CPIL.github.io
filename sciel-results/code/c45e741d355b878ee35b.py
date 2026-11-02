import time
import numpy as np

N, M, P = 2, 4, 8

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Rank 56 using Strassen on 2x2x2 blocks
    # 2 blocks in M, 4 blocks in P -> 8 blocks total
    # 7 multiplications per block
    RANK = 56
    
    # Shapes:
    # U: (N*M, RANK) = (8, 56)
    # V: (M*P, RANK) = (32, 56)
    # W: (P*N, RANK) = (16, 56)
    
    U = np.zeros((N * M, RANK), dtype=np.int64)
    V = np.zeros((M * P, RANK), dtype=np.int64)
    W = np.zeros((P * N, RANK), dtype=np.int64)
    
    # Strassen 2x2x2 decomposition
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # 
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the operations for one Strassen block
    # Each operation is a list of (matrix, row_idx_offset, col_offset_in_block, sign)
    # We will map these to the global indices.
    
    # Indices in local 2x2 block:
    # A: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    # B: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    # C: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    
    # Strassen factors for U (A), V (B), W (C)
    # Format: (local_idx, sign)
    # M1
    op_1 = {
        'U': [(0, 1), (3, 1)],
        'V': [(0, 1), (3, 1)],
        'W': [(0, 1), (3, 1)]
    }
    # M2
    op_2 = {
        'U': [(2, 1), (3, 1)],
        'V': [(0, 1)],
        'W': [(2, 1), (3, -1)]
    }
    # M3
    op_3 = {
        'U': [(0, 1)],
        'V': [(1, 1), (3, -1)],
        'W': [(1, 1), (3, 1)]
    }
    # M4
    op_4 = {
        'U': [(3, 1)],
        'V': [(2, 1), (0, -1)],
        'W': [(0, 1), (2, 1)]
    }
    # M5
    op_5 = {
        'U': [(0, 1), (1, 1)],
        'V': [(3, 1)],
        'W': [(0, -1), (1, 1)]
    }
    # M6
    op_6 = {
        'U': [(2, 1), (0, -1)],
        'V': [(0, 1), (1, 1)],
        'W': [(3, 1)]
    }
    # M7
    op_7 = {
        'U': [(1, 1), (3, -1)],
        'V': [(2, 1), (3, 1)],
        'W': [(0, 1)]
    }
    
    strassen_ops = [op_1, op_2, op_3, op_4, op_5, op_6, op_7]
    
    # Loop over blocks
    # u_idx: block index in M dimension (0, 1)
    # v_idx: block index in P dimension (0, 1, 2, 3)
    for u_idx in range(2):
        for v_idx in range(4):
            base_r = (u_idx * 4 + v_idx) * 7
            
            # Global indices for A (N x M)
            # A[i, j] -> i*M + j
            # Block A[:, 2*u_idx : 2*u_idx+2]
            a_base = 0 * M + 2 * u_idx
            # Local 0: A[0, 2u] -> a_base
            # Local 1: A[0, 2u+1] -> a_base + 1
            # Local 2: A[1, 2u] -> a_base + M
            # Local 3: A[1, 2u+1] -> a_base + M + 1
            
            # Global indices for B (M x P)
            # B[j, k] -> j*P + k
            # Block B[2u:2u+2, 2v:2v+2]
            b_base = (2 * u_idx) * P + 2 * v_idx
            # Local 0: B[2u, 2v] -> b_base
            # Local 1: B[2u, 2v+1] -> b_base + 1
            # Local 2: B[2u+1, 2v] -> b_base + P
            # Local 3: B[2u+1, 2v+1] -> b_base + P + 1
            
            # Global indices for C (N x P) -> W (P x N)
            # C[i, k] -> k*N + i
            # Block C[:, 2v:2v+2]
            w_base = (2 * v_idx) * N + 0
            # Local 0: C[0, 2v] -> w_base
            # Local 1: C[0, 2v+1] -> w_base + N
            # Local 2: C[1, 2v] -> w_base + 1
            # Local 3: C[1, 2v+1] -> w_base + N + 1
            
            # Mapping local index to global offset
            # A: [0, 1, M, M+1]
            a_map = [0, 1, M, M + 1]
            # B: [0, 1, P, P+1]
            b_map = [0, 1, P, P + 1]
            # C: [0, N, 1, N+1]
            c_map = [0, N, 1, N + 1]
            
            for r_offset, op in enumerate(strassen_ops):
                r = base_r + r_offset
                
                for local_idx, sign in op['U']:
                    U[a_base + a_map[local_idx], r] += sign
                for local_idx, sign in op['V']:
                    V[b_base + b_map[local_idx], r] += sign
                for local_idx, sign in op['W']:
                    W[w_base + c_map[local_idx], r] += sign
    
    # Verification
    if time.time() > deadline - 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    expected = np.zeros((N*M, M*P, P*N), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                expected[a, b, c] = 1
    
    if not np.array_equal(T, expected):
        # Fallback
        rank = N * M * P
        U_f = np.zeros((N*M, rank), dtype=np.int64)
        V_f = np.zeros((M*P, rank), dtype=np.int64)
        W_f = np.zeros((P*N, rank), dtype=np.int64)
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U_f[i*M+j, col] = 1
                    V_f[j*P+k, col] = 1
                    W_f[k*N+i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
