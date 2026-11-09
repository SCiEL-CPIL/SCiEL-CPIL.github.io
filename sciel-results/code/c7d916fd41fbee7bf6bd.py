import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen 2x2x2 decomposition (rank 7)
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]

    # Part 1: 4x4x4 decomposition with rank 49 (tensor product of two Strassen)
    rank1 = 49
    U1 = np.zeros((n*m, rank1), dtype=float)
    V1 = np.zeros((m*p, rank1), dtype=float)
    W1 = np.zeros((p*n, rank1), dtype=float)

    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            row_U = i * m + j
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    U1[row_U, col] = S_U[i1*2+j1][c1] * S_U[i2*2+j2][c2]

    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            row_V = j * p + k
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    V1[row_V, col] = S_V[j1*2+k1][c1] * S_V[j2*2+k2][c2]

    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            row_W = k * n + i
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    W1[row_W, col] = S_W[k1*2+i1][c1] * S_W[k2*2+i2][c2]

    # Part 2: 4x4x2 decomposition with rank 28 (tiling)
    rank2 = 28
    U2 = np.zeros((n*m, rank2), dtype=float)
    V2 = np.zeros((m*p, rank2), dtype=float)
    W2 = np.zeros((p*n, rank2), dtype=float)

    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = (bi * 2 + bj) * 7 + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U2[i * m + j, col] = S_U[di * 2 + dj][c]

                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = 4 + dk
                        V2[j * p + k, col] = S_V[dj * 2 + dk][c]

                for dk in range(2):
                    for di in range(2):
                        k = 4 + dk
                        i = bi * 2 + di
                        W2[k * n + i, col] = S_W[dk * 2 + di][c]

    # Combine
    total_rank = rank1 + rank2
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])

    # Verify correctness
    # The tensor decomposition must satisfy:
    # For all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if j is the middle index connecting A[i,j] and B[j,k]
    # i.e., T[i,j,k] = 1 for the standard matrix multiplication tensor
    
    # Build the multiplication tensor
    # A is n x m = 4x4, B is m x p = 4x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) with T[i,j,k] = 1 (it's the identity in the j-dimension pairing)
    
    # Actually the tensor is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{i,j,k} = 1 for all valid i,j,k (the structure tensor)
    # The decomposition requires: T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    
    # Let's verify on a random test
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * V[j*p+k,r] * W[k*n+i,r]
    # Actually: C[i,k] = sum_j sum_r U[i*m+j,r] * A[i,j] * V[j*p+k,r] * W[k*n+i,r]
    # Wait, that's not right either. Let me reconsider.
    
    # The tensor decomposition means:
    # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r (sum_j U_{i*m+j, r} A_{i,j}) * (sum_k' V_{j*p+k', r} ...) 
    # Hmm, let me think more carefully.
    
    # Standard matrix multiplication tensor: T_{ijk} = 1 (for all i,j,k in range)
    # The decomposition: T_{ijk} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    # So: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    
    # Let's verify this identity
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(total_rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    verified = False
                    break
        if not verified:
            break
    if not verified:
        # Fallback: naive decomposition (rank = n*m*p = 96)
        U_naive = np.zeros((n*m, n*m*p))
        V_naive = np.zeros((m*p, n*m*p))
        W_naive = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j, idx] = 1.0
                    V_naive[j*p+k, idx] = 1.0
                    W_naive[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    # Try to eliminate zero columns
    col_sums = np.sum(U**2, axis=0) + np.sum(V**2, axis=0) + np.sum(W**2, axis=0)
    keep_cols = [i for i in range(total_rank) if col_sums[i] > 1e-15]
    
    if len(keep_cols) < total_rank:
        U = U[:, keep_cols]
        V = V[:, keep_cols]
        W = W[:, keep_cols]

    final_rank = U.shape[1]
    
    # Convert to lists with integer/half-integer values
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure values are clean (round small values to 0, etc.)
    def clean_list(lst):
        cleaned = []
        for row in lst:
            clean_row = []
            for v in row:
                v_rounded = round(v, 10)
                if abs(v_rounded) < 1e-10:
                    v_rounded = 0.0
                clean_row.append(v_rounded)
            cleaned.append(clean_row)
        return cleaned
    
    U_final = clean_list(U_list)
    V_final = clean_list(V_list)
    W_final = clean_list(W_list)
    
    return {"U": U_final, "V": V_final, "W": W_final}
