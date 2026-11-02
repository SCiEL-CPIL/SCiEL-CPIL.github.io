import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # --- Strassen 2x2 (Rank 7) ---
    # U, V, W for 2x2 MM
    # Indices: 0:00, 1:01, 2:10, 3:11
    U2 = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, -1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, -1, -1, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0]
    ])
    # Note: I need to double check these vectors. 
    # I will verify them in code.

    # --- Schoolbook 3x3 (Rank 27) ---
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 3 + i, col] = 1
                col += 1

    # --- Laderman 3x3 (Rank 23) ---
    # I will hardcode Laderman vectors if I can find them reliably.
    # If not, I will fall back to Schoolbook 3x3.
    # For now, I will use Schoolbook 3x3 to ensure correctness, yielding rank 189.
    # Then I will try to implement Laderman.
    
    # Let's stick to 189 first to guarantee a valid improvement.
    # Rank = 27 * 7 = 189.
    
    # Construct 6x6 decomposition from 3x3 (block) and 2x2 (local)
    # Block size 2x2. Grid 3x3.
    # Total Rank = Rank(3x3) * Rank(2x2)
    
    rank_3x3 = 27
    rank_2x2 = 7
    total_rank = rank_3x3 * rank_2x2
    
    U6 = np.zeros((36, total_rank), dtype=int)
    V6 = np.zeros((36, total_rank), dtype=int)
    W6 = np.zeros((36, total_rank), dtype=int)
    
    # Mapping:
    # A index: (i_block, i_local) -> i_block * 2 + i_local ? No.
    # A is 6x6.
    # Block i_block, j_block contains A[i_block*2 + i_local, j_block*2 + j_local].
    # So A index = (i_block * 2 + i_local, j_block * 2 + j_local).
    # Flat index = (i_block * 2 + i_local) * 6 + (j_block * 2 + j_local).
    
    # U3 index: (i_block, j_block) -> i_block * 3 + j_block.
    # U2 index: (i_local, j_local) -> i_local * 2 + j_local.
    
    for r3 in range(rank_3x3):
        for r2 in range(rank_2x2):
            r_total = r3 * rank_2x2 + r2
            
            # U
            for ib in range(3):
                for jb in range(3):
                    u3_val = U3[ib * 3 + jb, r3]
                    if u3_val != 0:
                        for il in range(2):
                            for jl in range(2):
                                u2_val = U2[il * 2 + jl, r2]
                                if u2_val != 0:
                                    a_idx = (ib * 2 + il) * 6 + (jb * 2 + jl)
                                    U6[a_idx, r_total] = u3_val * u2_val
                                    
            # V
            for jb in range(3):
                for kb in range(3):
                    v3_val = V3[jb * 3 + kb, r3]
                    if v3_val != 0:
                        for jl in range(2):
                            for kl in range(2):
                                v2_val = V2[jl * 2 + kl, r2]
                                if v2_val != 0:
                                    b_idx = (jb * 2 + jl) * 6 + (kb * 2 + kl)
                                    V6[b_idx, r_total] = v3_val * v2_val
                                    
            # W
            for kb in range(3):
                for ib in range(3):
                    w3_val = W3[kb * 3 + ib, r3]
                    if w3_val != 0:
                        for kl in range(2):
                            for il in range(2):
                                w2_val = W2[kl * 2 + il, r2]
                                if w2_val != 0:
                                    c_idx = (kb * 2 + kl) * 6 + (ib * 2 + il)
                                    W6[c_idx, r_total] = w3_val * w2_val

    # Verification
    # Check if U6, V6, W6 decompose MM6
    # C = sum_r W_r * (U_r o A) * (V_r o B)
    # Use random integer matrices
    np.random.seed(42)
    A = np.random.randint(-10, 10, (6, 6))
    B = np.random.randint(-10, 10, (6, 6))
    C_expected = A @ B
    
    C_computed = np.zeros((6, 6), dtype=int)
    for r in range(total_rank):
        u = U6[:, r]
        v = V6[:, r]
        w = W6[:, r]
        
        u_mat = u.reshape(6, 6)
        v_mat = v.reshape(6, 6)
        w_mat = w.reshape(6, 6)
        
        # Term is (u_mat o A) * (v_mat o B) * w_mat? No.
        # The tensor decomposition is T = sum U_r o V_r o W_r.
        # C = sum_r W_r * scalar_r.
        # scalar_r = (U_r o A) * (V_r o B) ? No.
        # scalar_r = sum_{i,j} U_r[i,j] A[i,j] * sum_{k,l} V_r[k,l] B[k,l] ? No.
        # It is (sum U_r[i,j] A[i,j]) * (sum V_r[k,l] B[k,l]).
        # Let's check the definition.
        # T[i,j,k,l] = delta_{j,k}.
        # C[i,l] = sum_{j,k} T[i,j,k,l] A[i,j] B[k,l] = sum_j A[i,j] B[j,l].
        # Decomposition: T = sum_r U_r o V_r o W_r.
        # U_r is n x m. V_r is m x p. W_r is n x p.
        # T[i,j,k,l] = sum_r U_r[i,j] V_r[k,l] W_r[i,l]? No.
        # W_r is the output coefficient.
        # C[i,l] = sum_r W_r[i,l] * (sum_{j} U_r[i,j] A[i,j]) * (sum_{k} V_r[k,l] B[k,l])?
        # No, U_r is a matrix.
        # The term is (U_r * A) ? No.
        # The bilinear form is:
        # C = sum_r W_r * ( <U_r, A> * <V_r, B> )
        # Where <,> is inner product (sum of element-wise product).
        
        u_scalar = np.sum(u * A)
        v_scalar = np.sum(v * B)
        C_computed += w * (u_scalar * v_scalar)
        
    if not np.allclose(C_computed, C_expected):
        # Fallback to schoolbook 6x6 (Rank 216)
        rank = 216
        U = [[0] * rank for _ in range(36)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        column = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i * 6 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 6 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists of lists
    U_list = U6.tolist()
    V_list = V6.tolist()
    W_list = W6.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

