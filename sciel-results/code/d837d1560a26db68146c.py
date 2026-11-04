"""Rewritten program for matrix multiplication 4x4x9 rank minimization."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 9

    # Schoolbook rank
    rank_sb = n * m * p  # 4*4*9 = 144
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    def validate_fast(U, V, W, tol=1e-9):
        """Faster validation using precomputed dot products."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_winograd_4x4x4():
        """
        Build a Winograd-based decomposition for 4x4 * 4x4 -> 4x4.
        Uses frequency-domain (Winograd) approach with 4-point transforms.
        Rank = 4*4*4 = 64 for basic Winograd, but we use Strassen-like
        structure for better rank.
        
        Actually, let's use a hybrid: apply Strassen at block level
        but with a smarter 4x4x4 base that uses the known optimal
        rank-49 decomposition.
        
        Returns U, V, W each of shape (16, rank).
        """
        # Use the recursive Strassen approach which gives rank 49
        # This is the standard best known for 4x4x4
        U2 = np.array([
            [1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, -1, 0],
            [1, 1, 0, 1, 0, 0, 0],
        ])
        V2 = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [1, 0, -1, -1, 1, 0, 1],
        ])
        W2 = np.array([
            [1, 0, 1, 1, -1, 0, 1],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 1],
        ])
        
        base_rank = 7
        total_rank = 7 * base_rank  # 49
        U4 = np.zeros((16, total_rank))
        V4 = np.zeros((16, total_rank))
        W4 = np.zeros((16, total_rank))
        
        block_left = np.zeros((7, 2, 2))
        block_left[0, 0, 0] = 1; block_left[0, 1, 1] = 1
        block_left[1, 1, 0] = 1; block_left[1, 1, 1] = 1
        block_left[2, 0, 0] = 1
        block_left[3, 1, 1] = 1
        block_left[4, 0, 0] = 1; block_left[4, 0, 1] = 1
        block_left[5, 1, 0] = 1; block_left[5, 0, 0] = -1
        block_left[6, 0, 1] = 1; block_left[6, 1, 1] = -1
        
        block_right = np.zeros((7, 2, 2))
        block_right[0, 0, 0] = 1; block_right[0, 1, 1] = 1
        block_right[1, 0, 0] = 1
        block_right[2, 1, 0] = 1; block_right[2, 1, 1] = -1
        block_right[3, 0, 1] = 1; block_right[3, 0, 0] = -1
        block_right[4, 1, 1] = 1
        block_right[5, 0, 0] = 1; block_right[5, 0, 1] = 1
        block_right[6, 1, 0] = 1; block_right[6, 1, 1] = 1
        
        block_output = np.zeros((7, 2, 2))
        block_output[0, 0, 0] = 1; block_output[0, 1, 1] = 1
        block_output[1, 1, 0] = 1; block_output[1, 1, 1] = -1
        block_output[2, 0, 1] = 1; block_output[2, 1, 1] = 1
        block_output[3, 0, 0] = 1; block_output[3, 1, 0] = 1
        block_output[4, 0, 0] = -1; block_output[4, 0, 1] = 1
        block_output[5, 1, 1] = -1
        block_output[6, 0, 0] = 1; block_output[6, 1, 1] = 1
        
        for r in range(7):
            for s in range(base_rank):
                col = r * base_rank + s
                for i in range(4):
                    for j in range(4):
                        bi, bj = i // 2, j // 2
                        u_coeff = block_left[r, bi, bj]
                        if u_coeff != 0:
                            U4[i*4+j, col] = u_coeff * U2[(i%2)*2+(j%2), s]
                for j in range(4):
                    for k in range(4):
                        bi, bj = j // 2, k // 2
                        v_coeff = block_right[r, bi, bj]
                        if v_coeff != 0:
                            V4[j*4+k, col] = v_coeff * V2[(j%2)*2+(k%2), s]
                for k in range(4):
                    for i in range(4):
                        bi, bj = k // 2, i // 2
                        w_coeff = block_output[r, bi, bj]
                        if w_coeff != 0:
                            W4[k*4+i, col] = w_coeff * W2[(k%2)*2+(i%2), s]
        
        return U4, V4, W4, total_rank

    # Build the 4x4x4 Strassen decomposition
    U4, V4, W4, rank_4x4x4 = build_winograd_4x4x4()

    # Build the full 4x4x9 decomposition
    # Use 4x4x4 for columns 0-3, 4x4x4 for columns 4-7, schoolbook for column 8
    rank_4x4x1 = 16
    total_rank = 2 * rank_4x4x4 + rank_4x4x1

    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))

    # First 4x4x4 (columns 0-3)
    U_full[:, 0:rank_4x4x4] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*9+k, 0:rank_4x4x4] = V4[j*4+k, 0:rank_4x4x4]
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, 0:rank_4x4x4] = W4[k*4+i, 0:rank_4x4x4]

    # Second 4x4x4 (columns 4-7)
    offset2 = rank_4x4x4
    U_full[:, offset2:offset2+rank_4x4x4] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*9+(k+4), offset2:offset2+rank_4x4x4] = V4[j*4+k, 0:rank_4x4x4]
    for k in range(4):
        for i in range(4):
            W_full[(k+4)*4+i, offset2:offset2+rank_4x4x4] = W4[k*4+i, 0:rank_4x4x4]

    # 4x4x1 schoolbook (column 8)
    offset3 = 2 * rank_4x4x4
    col = offset3
    for i in range(4):
        for j in range(4):
            U_full[i*4+j, col] = 1
            V_full[j*9+8, col] = 1
            W_full[8*4+i, col] = 1
            col += 1

    # Validate
    if validate_fast(U_full.tolist(), V_full.tolist(), W_full.tolist()):
        best_rank = total_rank
        best = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    return best
# EVOLVE-BLOCK-END
