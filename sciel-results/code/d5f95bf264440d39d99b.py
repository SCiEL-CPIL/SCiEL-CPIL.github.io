"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Use the four-block decomposition (4 x rank-11 = rank-44)
    U, V, W, rank = _four_block_decomposition(n, m, p)

    # Verify the decomposition
    if not _verify(n, m, p, U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        rank = n * m * p

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank: 44.
    """
    n, m, p = 2, 3, 8
    block_size = 2
    rank_per_block = 11
    total_rank = rank_per_block * 4

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for b in range(4):
        k_offset = b * block_size
        col_offset = b * rank_per_block

        # Rank-11 decomposition for 2x3 * 3x2
        # A is 2x3: indices (i,j) -> flat i*3+j, i in {0,1}, j in {0,1,2}
        # B is 3x2: indices (j,k) -> flat j*8+k_offset+k (in full B), j in {0,1,2}, k in {0,1}
        # C is 2x2: indices (i,k) -> flat (k_offset+k)*2+i, i in {0,1}, k in {0,1}
        
        # Term definitions: (U_dict, V_dict, W_dict)
        # U_dict: {flat_A_index: coeff}
        # V_dict: {flat_B_index: coeff}
        # W_dict: {flat_C_index: coeff}
        
        terms = [
            # Term 0: (a00+a10) * (b00+b01)
            ({0: 1, 3: 1}, {0*8+k_offset+0: 1, 0*8+k_offset+1: 1},
             {(k_offset+0)*2+0: 1, (k_offset+1)*2+0: 1, (k_offset+0)*2+1: 1, (k_offset+1)*2+1: 1}),
            # Term 1: a00 * b01
            ({0: 1}, {0*8+k_offset+1: 1},
             {(k_offset+1)*2+0: 1}),
            # Term 2: a10 * (b10+b11)
            ({3: 1}, {1*8+k_offset+0: 1, 1*8+k_offset+1: 1},
             {(k_offset+0)*2+1: 1, (k_offset+1)*2+1: 1}),
            # Term 3: a11 * b00
            ({4: 1}, {0*8+k_offset+0: 1},
             {(k_offset+0)*2+1: 1}),
            # Term 4: (a00+a01) * (b10-b00)
            ({0: 1, 1: 1}, {1*8+k_offset+0: 1, 0*8+k_offset+0: -1},
             {(k_offset+0)*2+0: 1}),
            # Term 5: (a11-a01) * (b00+b01)
            ({4: 1, 1: -1}, {0*8+k_offset+0: 1, 0*8+k_offset+1: 1},
             {(k_offset+0)*2+1: 1, (k_offset+1)*2+1: 1}),
            # Term 6: (a01+a11) * (b11-b01)
            ({1: 1, 4: 1}, {1*8+k_offset+1: 1, 0*8+k_offset+1: -1},
             {(k_offset+1)*2+0: 1, (k_offset+1)*2+1: 1}),
            # Term 7: a02 * b20
            ({2: 1}, {2*8+k_offset+0: 1},
             {(k_offset+0)*2+0: 1}),
            # Term 8: a02 * b21
            ({2: 1}, {2*8+k_offset+1: 1},
             {(k_offset+1)*2+0: 1}),
            # Term 9: a12 * b20
            ({5: 1}, {2*8+k_offset+0: 1},
             {(k_offset+0)*2+1: 1}),
            # Term 10: a12 * b21
            ({5: 1}, {2*8+k_offset+1: 1},
             {(k_offset+1)*2+1: 1}),
        ]
        
        for c, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, val in u_dict.items():
                U[idx, col_offset + c] += val
            for idx, val in v_dict.items():
                V[idx, col_offset + c] += val
            for idx, val in w_dict.items():
                W[idx, col_offset + c] += val

    return U, V, W, total_rank


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using multiple test cases.
    """
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        u_vals = np.einsum('ijc,ij->ic', U_reshaped, A)
        v_vals = np.einsum('lkp,lk->kp', V_reshaped, B)
        C_comp = np.einsum('kic,ic,kc->ik', W_reshaped, u_vals, v_vals)
        if not np.allclose(C_comp, C_true, atol=1e-9):
            return False
    return True


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
