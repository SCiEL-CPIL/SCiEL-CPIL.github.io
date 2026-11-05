import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Rank
    # 2 blocks of 3x2x7
    # Each 3x2x7: 3x2x4 (22) + 3x2x3 (17) = 39
    # Total 78
    
    rank = 78
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_tensor(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j, col] += u_val
        V[j * p + k, col] += v_val
        W[k * n + i, col] += w_val
        col += 1

    def add_strassen_2x2x2(base_i, base_j, base_k):
        # Adds 7 tensors for 2x2x2 block
        # Local indices 0,1
        # M1
        add_tensor(base_i+0, base_j+0, base_k+0, 1, 1, 1) # W(0,0)
        U[base_i+1, base_j+1, col-1] = 1 # Wait, U is 2D.
        # My add_tensor adds to U[i*m+j, col].
        # I need to support multiple non-zeros in U, V, W for one rank-1 tensor.
        # The current add_tensor only adds ONE non-zero to U, V, W.
        # Strassen tensors have multiple non-zeros.
        # I need a different helper.
        pass

    # Re-define helper to handle sparse vectors
    def add_rank1_tensor(u_indices, v_indices, w_indices):
        """
        u_indices: list of (i, j, val)
        v_indices: list of (j, k, val)
        w_indices: list of (k, i, val)
        """
        nonlocal col
        for i, j, val in u_indices:
            U[i * m + j, col] += val
        for j, k, val in v_indices:
            V[j * p + k, col] += val
        for k, i, val in w_indices:
            W[k * n + i, col] += val
        col += 1

    def add_strassen_2x2x2(base_i, base_j, base_k):
        # M1
        add_rank1_tensor(
            [(base_i+0, base_j+0, 1), (base_i+1, base_j+1, 1)],
            [(base_j+0, base_k+0, 1), (base_j+1, base_k+1, 1)],
            [(base_k+0, base_i+0, 1), (base_k+1, base_i+1, 1)]
        )
        # M2
        add_rank1_tensor(
            [(base_i+1, base_j+0, 1), (base_i+1, base_j+1, 1)],
            [(base_j+0, base_k+0, 1)],
            [(base_k+0, base_i+1, 1)]
        )
        # M3
        add_rank1_tensor(
            [(base_i+0, base_j+0, 1)],
            [(base_j+0, base_k+1, 1), (base_j+1, base_k+1, -1)],
            [(base_k+1, base_i+0, 1)]
        )
        # M4
        add_rank1_tensor(
            [(base_i+1, base_j+1, 1)],
            [(base_j+1, base_k+0, 1), (base_j+0, base_k+0, -1)],
            [(base_k+0, base_i+1, 1), (base_k+0, base_i+0, 1)]
        )
        # M5
        add_rank1_tensor(
            [(base_i+0, base_j+0, 1), (base_i+0, base_j+1, 1)],
            [(base_j+1, base_k+1, 1)],
            [(base_k+1, base_i+1, 1), (base_k+0, base_i+0, -1)]
        )
        # M6
        add_rank1_tensor(
            [(base_i+1, base_j+0, 1), (base_i+0, base_j+0, -1)],
            [(base_j+0, base_k+0, 1), (base_j+0, base_k+1, 1)],
            [(base_k+1, base_i+1, 1)]
        )
        # M7
        add_rank1_tensor(
            [(base_i+0, base_j+1, 1), (base_i+1, base_j+1, -1)],
            [(base_j+1, base_k+0, 1), (base_j+1, base_k+1, 1)],
            [(base_k+0, base_i+0, 1)]
        )

    def add_1x2x2(base_i, base_j, base_k):
        # i is single
        # A[i,0] B[0,0] -> C[i,0]
        add_rank1_tensor(
            [(base_i, base_j+0, 1)],
            [(base_j+0, base_k+0, 1)],
            [(base_k+0, base_i, 1)]
        )
        # A[i,0] B[0,1] -> C[i,1]
        add_rank1_tensor(
            [(base_i, base_j+0, 1)],
            [(base_j+0, base_k+1, 1)],
            [(base_k+1, base_i, 1)]
        )
        # A[i,1] B[1,0] -> C[i,0]
        add_rank1_tensor(
            [(base_i, base_j+1, 1)],
            [(base_j+1, base_k+0, 1)],
            [(base_k+0, base_i, 1)]
        )
        # A[i,1] B[1,1] -> C[i,1]
        add_rank1_tensor(
            [(base_i, base_j+1, 1)],
            [(base_j+1, base_k+1, 1)],
            [(base_k+1, base_i, 1)]
        )

    def add_3x2x2(base_i, base_j, base_k):
        # Split i into 2 (Strassen) and 1 (Schoolbook)
        add_strassen_2x2x2(base_i, base_j, base_k)
        add_1x2x2(base_i+2, base_j, base_k)

    def add_3x2x1(base_i, base_j, base_k):
        # i in 0..2, j in 0..1, k is single
        for i_off in range(3):
            for j_off in range(2):
                add_rank1_tensor(
                    [(base_i+i_off, base_j+j_off, 1)],
                    [(base_j+j_off, base_k, 1)],
                    [(base_k, base_i+i_off, 1)]
                )

    # Block 1: j in 0,1
    # Sub-block 1.1: k in 0,1,2,3
    # 1.1.1: k in 0,1 (3x2x2)
    add_3x2x2(0, 0, 0)
    # 1.1.2: k in 2,3 (3x2x2)
    add_3x2x2(0, 0, 2)
    
    # Sub-block 1.2: k in 4,5,6
    # 1.2.1: k in 4,5 (3x2x2)
    add_3x2x2(0, 0, 4)
    # 1.2.2: k in 6 (3x2x1)
    add_3x2x1(0, 0, 6)
    
    # Block 2: j in 2,3
    # Sub-block 2.1: k in 0,1,2,3
    # 2.1.1: k in 0,1 (3x2x2)
    add_3x2x2(0, 2, 0)
    # 2.1.2: k in 2,3 (3x2x2)
    add_3x2x2(0, 2, 2)
    
    # Sub-block 2.2: k in 4,5,6
    # 2.2.1: k in 4,5 (3x2x2)
    add_3x2x2(0, 2, 4)
    # 2.2.2: k in 6 (3x2x1)
    add_3x2x1(0, 2, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
