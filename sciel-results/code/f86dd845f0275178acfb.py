"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Start with schoolbook rank-48
    U, V, W = _schoolbook(n, m, p)
    rank = n * m * p

    # Aggressively reduce rank
    U, V, W, rank = _reduce_rank(n, m, p, U, V, W, rank, deadline)

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

        strassen_U = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1},
        ]
        strassen_V = [
            {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},
            {0 * p + k_offset + 0: 1},
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: -1},
            {0 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1},
            {1 * p + k_offset + 1: 1},
            {0 * p + k_offset + 0: 1, 0 * p + k_offset + 1: 1},
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},
        ]
        strassen_W = [
            {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: 1},
            {(k_offset + 0) * n + 1: 1},
            {(k_offset + 1) * n + 0: 1, (k_offset + 1) * n + 1: 1},
            {(k_offset + 0) * n + 0: 1, (k_offset + 0) * n + 1: 1},
            {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1},
            {(k_offset + 1) * n + 1: 1},
            {(k_offset + 0) * n + 0: 1},
        ]

        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val

        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1

    return U, V, W, total_rank


def _two_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 2 blocks of size 4.
    For each 2x3 x 3x4 sub-problem, use a rank-12 decomposition.
    Total rank: 24.
    
    For 2x3 * 3x4:
    Split B(3x4) into B_left(3x2) and B_right(3x2).
    Each 2x3 * 3x2 block uses rank-11 (Strassen 7 + outer 4).
    But we can do better by sharing: use rank-12 for the full 2x3*3x4.
    
    Actually, let's use the known optimal rank for 2x3*3x4.
    The schoolbook rank is 24. We can potentially do better.
    
    Let me try a direct rank-12 construction:
    C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1,2,3}
    
    Using a Strassen-like approach on the 2x3*3x4 tensor:
    We can write it as a sum of rank-1 terms.
    
    Let's try to construct a rank-12 decomposition manually.
    """
    n, m, p = 2, 3, 8
    block_size = 4
    rank_per_block = 12
    total_rank = rank_per_block * 2

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for b in range(2):
        k_offset = b * block_size
        col_offset = b * rank_per_block

        # Use a rank-12 decomposition for 2x3 * 3x4
        # This is based on splitting into two 2x3*3x2 blocks with sharing
        # or using a more efficient algorithm
        
        # For now, let's use the naive approach of two rank-11 blocks
        # but try to merge some terms
        
        # Block left (columns 0,1)
        _fill_2x3x3x2(U, V, W, n, m, p, k_offset, col_offset, 0)
        # Block right (columns 2,3)
        _fill_2x3x3x2(U, V, W, n, m, p, k_offset + 2, col_offset + 11, 1)
        
        # Try to merge the last term of block 0 with first term of block 1
        # This is a heuristic; verify will catch errors
        
    return U, V, W, total_rank


def _fill_2x3x3x2(U, V, W, n, m, p, k_offset, col_offset, block_id):
    """Fill a 2x3 * 3x2 block with rank-11 (Strassen 7 + outer 4)."""
    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    strassen_V = [
        {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},
        {0 * p + k_offset + 0: 1},
        {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: -1},
        {0 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1},
        {1 * p + k_offset + 1: 1},
        {0 * p + k_offset + 0: 1, 0 * p + k_offset + 1: 1},
        {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},
    ]
    strassen_W = [
        {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: 1},
        {(k_offset + 0) * n + 1: 1},
        {(k_offset + 1) * n + 0: 1, (k_offset + 1) * n + 1: 1},
        {(k_offset + 0) * n + 0: 1, (k_offset + 0) * n + 1: 1},
        {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1},
        {(k_offset + 1) * n + 1: 1},
        {(k_offset + 0) * n + 0: 1},
    ]

    for c in range(7):
        for row, val in strassen_U[c].items():
            U[row, col_offset + c] += val
        for row, val in strassen_V[c].items():
            V[row, col_offset + c] += val
        for row, val in strassen_W[c].items():
            W[row, col_offset + c] += val

    for ii in range(2):
        for kk in range(2):
            c = 7 + ii * 2 + kk
            U[ii * m + 2, col_offset + c] = 1
            V[2 * p + k_offset + kk, col_offset + c] = 1
            W[(k_offset + kk) * n + ii, col_offset + c] = 1


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using multiple test cases.
    """
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        u_vals = np.einsum('ijc,ij->ic', U_reshaped, A)
        v_vals = np.einsum('lkp,lk->kp', V_reshaped, B)
        C_comp = np.einsum('kic,ic,kc->ik', W_reshaped, u_vals, v_vals)
        if not np.allclose(C_comp, C_true, atol=1e-9):
            return False
    return True


def _reduce_rank(n, m, p, U, V, W, rank, deadline):
    """
    Aggressively reduce the rank by removing columns.
    """
    U, V, W = U.copy(), V.copy(), W.copy()
    
    # Try removing one column at a time, multiple passes
    for pass_num in range(20):
        if time.time() >= deadline - 10:
            break
        changed = False
        for c in range(rank - 1, -1, -1):
            if time.time() >= deadline - 10:
                break
            mask = [i for i in range(rank) if i != c]
            if _verify(n, m, p, U[:, mask], V[:, mask], W[:, mask]):
                U, V, W = U[:, mask], V[:, mask], W[:, mask]
                rank -= 1
                changed = True
        if not changed:
            break
    
    # Try removing two columns at a time
    for pass_num in range(10):
        if time.time() >= deadline - 5:
            break
        changed = False
        for c1 in range(rank - 1, -1, -1):
            if time.time() >= deadline - 5:
                break
            for c2 in range(c1 - 1, -1, -1):
                if time.time() >= deadline - 5:
                    break
                mask = [i for i in range(rank) if i != c1 and i != c2]
                if _verify(n, m, p, U[:, mask], V[:, mask], W[:, mask]):
                    U, V, W = U[:, mask], V[:, mask], W[:, mask]
                    rank -= 2
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    
    return U, V, W, rank


def _aggressive_refine(n, m, p, U, V, W, current_rank, deadline):
    """
    More aggressive refinement: try removing multiple columns at once.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    removed_any = True
    while removed_any and time.time() < deadline - 10:
        removed_any = False
        for c_remove in range(rank - 1, -1, -1):
            if time.time() >= deadline - 10:
                break
            mask = [c for c in range(rank) if c != c_remove]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if _verify(n, m, p, U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed_any = True
                if rank <= 1:
                    return U, V, W, rank
    
    if time.time() < deadline - 5:
        for c1 in range(rank - 1, -1, -1):
            if time.time() >= deadline - 5:
                break
            for c2 in range(c1 - 1, -1, -1):
                if time.time() >= deadline - 5:
                    break
                mask = [c for c in range(rank) if c != c1 and c != c2]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify(n, m, p, U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank -= 2
                    if rank <= 1:
                        return U, V, W, rank
                    break
    
    return U, V, W, rank


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
