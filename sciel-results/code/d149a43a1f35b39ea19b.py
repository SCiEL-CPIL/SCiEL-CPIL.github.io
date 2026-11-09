"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen on 2x2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    Uses 2+2+1 block partition with Strassen on all 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Strassen 2x2x2 decomposition (rank 7)
    STRASSEN_2x2_TERMS = [
        # (U_coeffs, V_coeffs, W_coeffs) where keys are (row, col)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_2x2x2(I_map, J_map, K_map):
        """Add Strassen 2x2x2 decomposition for blocks indexed by I_map, J_map, K_map."""
        for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), val in u_coeffs.items():
                gi = I_map[li]
                gj = J_map[lj]
                u_entries.append((gi * m + gj, val))
            for (lj, lk), val in v_coeffs.items():
                gj = J_map[lj]
                gk = K_map[lk]
                v_entries.append((gj * p + gk, val))
            for (li, lk), val in w_coeffs.items():
                gi = I_map[li]
                gk = K_map[lk]
                w_entries.append((gk * n + gi, val))
            add_column(u_entries, v_entries, w_entries)

    def schoolbook_block(i_list, j_list, k_list):
        """Add schoolbook (rank = len(i)*len(j)*len(k)) for a block."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )

    def add_block(i_list, j_list, k_list):
        """Add decomposition for a block based on its dimensions."""
        ni, nj, nk = len(i_list), len(j_list), len(k_list)
        
        if ni == 2 and nj == 2 and nk == 2:
            # Use Strassen (rank 7)
            add_strassen_2x2x2(
                {0: i_list[0], 1: i_list[1]},
                {0: j_list[0], 1: j_list[1]},
                {0: k_list[0], 1: k_list[1]}
            )
        elif ni == 1 and nj == 1 and nk == 1:
            # Single element
            schoolbook_block(i_list, j_list, k_list)
        elif ni == 2 and nj == 1 and nk == 1:
            # 2x1x1: rank 2
            a0 = i_list[0] * m + j_list[0]
            a1 = i_list[1] * m + j_list[0]
            b0 = j_list[0] * p + k_list[0]
            c0 = k_list[0] * n + i_list[0]
            c1 = k_list[0] * n + i_list[1]
            add_column([(a0, 1.0)], [(b0, 1.0)], [(c0, 1.0)])
            add_column([(a1, 1.0)], [(b0, 1.0)], [(c1, 1.0)])
        elif ni == 1 and nj == 2 and nk == 2:
            # 1x2x2: rank 4 (schoolbook)
            a0 = i_list[0] * m + j_list[0]
            a1 = i_list[0] * m + j_list[1]
            b00 = j_list[0] * p + k_list[0]
            b01 = j_list[0] * p + k_list[1]
            b10 = j_list[1] * p + k_list[0]
            b11 = j_list[1] * p + k_list[1]
            c0 = k_list[0] * n + i_list[0]
            c1 = k_list[1] * n + i_list[0]
            add_column([(a0, 1.0)], [(b00, 1.0)], [(c0, 1.0)])
            add_column([(a1, 1.0)], [(b10, 1.0)], [(c0, 1.0)])
            add_column([(a0, 1.0)], [(b01, 1.0)], [(c1, 1.0)])
            add_column([(a1, 1.0)], [(b11, 1.0)], [(c1, 1.0)])
        elif ni == 2 and nj == 2 and nk == 1:
            # 2x2x1: rank 4 (schoolbook)
            a00 = i_list[0] * m + j_list[0]
            a01 = i_list[0] * m + j_list[1]
            a10 = i_list[1] * m + j_list[0]
            a11 = i_list[1] * m + j_list[1]
            b0 = j_list[0] * p + k_list[0]
            b1 = j_list[1] * p + k_list[0]
            c0 = k_list[0] * n + i_list[0]
            c1 = k_list[0] * n + i_list[1]
            add_column([(a00, 1.0)], [(b0, 1.0)], [(c0, 1.0)])
            add_column([(a01, 1.0)], [(b1, 1.0)], [(c0, 1.0)])
            add_column([(a10, 1.0)], [(b0, 1.0)], [(c1, 1.0)])
            add_column([(a11, 1.0)], [(b1, 1.0)], [(c1, 1.0)])
        elif ni == 2 and nj == 1 and nk == 2:
            # 2x1x2: rank 4 (schoolbook)
            a0 = i_list[0] * m + j_list[0]
            a1 = i_list[1] * m + j_list[0]
            b00 = j_list[0] * p + k_list[0]
            b01 = j_list[0] * p + k_list[1]
            c00 = k_list[0] * n + i_list[0]
            c01 = k_list[1] * n + i_list[0]
            c10 = k_list[0] * n + i_list[1]
            c11 = k_list[1] * n + i_list[1]
            add_column([(a0, 1.0)], [(b00, 1.0)], [(c00, 1.0)])
            add_column([(a0, 1.0)], [(b01, 1.0)], [(c01, 1.0)])
            add_column([(a1, 1.0)], [(b00, 1.0)], [(c10, 1.0)])
            add_column([(a1, 1.0)], [(b01, 1.0)], [(c11, 1.0)])
        elif ni == 1 and nj == 2 and nk == 1:
            # 1x2x1: rank 2
            a0 = i_list[0] * m + j_list[0]
            a1 = i_list[0] * m + j_list[1]
            b0 = j_list[0] * p + k_list[0]
            b1 = j_list[1] * p + k_list[0]
            c0 = k_list[0] * n + i_list[0]
            add_column([(a0, 1.0)], [(b0, 1.0)], [(c0, 1.0)])
            add_column([(a1, 1.0)], [(b1, 1.0)], [(c0, 1.0)])
        elif ni == 1 and nj == 1 and nk == 2:
            # 1x1x2: rank 2
            a0 = i_list[0] * m + j_list[0]
            b00 = j_list[0] * p + k_list[0]
            b01 = j_list[0] * p + k_list[1]
            c0 = k_list[0] * n + i_list[0]
            c1 = k_list[1] * n + i_list[0]
            add_column([(a0, 1.0)], [(b00, 1.0)], [(c0, 1.0)])
            add_column([(a0, 1.0)], [(b01, 1.0)], [(c1, 1.0)])
        else:
            # Fallback to schoolbook
            schoolbook_block(i_list, j_list, k_list)

    # Partition 2+2+1
    parts = [[0, 1], [2, 3], [4]]
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                i_list = parts[bi]
                j_list = parts[bj]
                k_list = parts[bk]
                add_block(i_list, j_list, k_list)

    rank = len(columns)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
            
    # Quick verification
    if not _deadline_exceeded():
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        U_np = np.array(U).T  # (rank, n*m)
        V_np = np.array(V).T  # (rank, m*p)
        W_np = np.array(W).T  # (rank, p*n)
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_np[r].reshape(n, m)
            v_vec = V_np[r].reshape(m, p)
            w_vec = W_np[r].reshape(p, n)
            # The decomposition: C[i,k] = sum_r sum_j U[i,j,r] * V[j,k,r] * W[k,i,r]
            # But W is indexed as [k*n+i, r], so w_vec[k*n+i]
            # This is a 3-tensor contraction
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
                    C_computed[i, k] += s
        
        # Check if decomposition is correct
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook if verification fails
            columns = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        add_column(
                            [(i * m + j, 1.0)],
                            [(j * p + k, 1.0)],
                            [(k * n + i, 1.0)]
                        )
            rank = len(columns)
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            for c, (u_d, v_d, w_d) in enumerate(columns):
                for idx, val in u_d.items():
                    U[idx][c] = val
                for idx, val in v_d.items():
                    V[idx][c] = val
                for idx, val in w_d.items():
                    W[idx][c] = val
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
