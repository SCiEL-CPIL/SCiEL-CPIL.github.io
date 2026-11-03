"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using batched matrix products."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # For each (i,j,k), the coefficient should be sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Compute T[i,j,k] = einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    # Check T[i,j,k] == 1 for all i,j,k
    if not np.all(np.abs(T - 1.0) <= tol):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2.
    U: (4,7) indexed by [i*2+j], V: (4,7) indexed by [j*2+k], W: (4,7) indexed by [k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00(b10-b11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b01-b00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b01+b11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p, row_blocks, inner_blocks, col_blocks):
    """
    Build a decomposition by applying Strassen to 2x2x2 blocks where possible.
    row_blocks: partition of rows {0..n-1}
    inner_blocks: partition of inner {0..m-1}
    col_blocks: partition of cols {0..p-1}
    
    For each combination of (row_block, inner_block, col_block):
    - If all sizes are 2, use Strassen rank-7
    - Otherwise, use schoolbook for that sub-block
    """
    U_list = []
    V_list = []
    W_list = []
    
    for rb in row_blocks:
        for ib in inner_blocks:
            for cb in col_blocks:
                r_size = len(rb)
                i_size = len(ib)
                c_size = len(cb)
                
                if r_size == 2 and i_size == 2 and c_size == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2x2()
                    for c in range(7):
                        u_col = [0] * (n * m)
                        v_col = [0] * (m * p)
                        w_col = [0] * (p * n)
                        for ii in range(2):
                            i_val = rb[ii]
                            for jj in range(2):
                                j_val = ib[jj]
                                u_val = sU[ii * 2 + jj][c]
                                if u_val != 0:
                                    u_col[i_val * m + j_val] = u_val
                        for jj in range(2):
                            j_val = ib[jj]
                            for kk in range(2):
                                k_val = cb[kk]
                                v_val = sV[jj * 2 + kk][c]
                                if v_val != 0:
                                    v_col[j_val * p + k_val] = v_val
                        for kk in range(2):
                            k_val = cb[kk]
                            for ii in range(2):
                                i_val = rb[ii]
                                w_val = sW[kk * 2 + ii][c]
                                if w_val != 0:
                                    w_col[k_val * n + i_val] = w_val
                        U_list.append(u_col)
                        V_list.append(v_col)
                        W_list.append(w_col)
                else:
                    # Use schoolbook for this sub-block
                    for ii in range(r_size):
                        i_val = rb[ii]
                        for jj in range(i_size):
                            j_val = ib[jj]
                            for kk in range(c_size):
                                k_val = cb[kk]
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                u_col[i_val * m + j_val] = 1
                                v_col[j_val * p + k_val] = 1
                                w_col[k_val * n + i_val] = 1
                                U_list.append(u_col)
                                V_list.append(v_col)
                                W_list.append(w_col)
    
    rank = len(U_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for idx, val in enumerate(U_list[c]):
            if val != 0:
                U[idx][c] = val
        for idx, val in enumerate(V_list[c]):
            if val != 0:
                V[idx][c] = val
        for idx, val in enumerate(W_list[c]):
            if val != 0:
                W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}

def _all_partitions(lst):
    """Generate all set partitions of a list as a list of tuples."""
    if not lst:
        yield []
        return
    first = lst[0]
    rest = lst[1:]
    for partition in _all_partitions(rest):
        # Insert first into each existing block
        for i in range(len(partition)):
            new_partition = [list(block) for block in partition]
            new_partition[i].append(first)
            yield [tuple(block) for block in new_partition]
        # Create new block with first
        yield [tuple([first])] + [tuple(block) for block in partition]

def _precompute_partition_ranks(n, m, p):
    """
    Precompute the rank contribution for each combination of block sizes.
    For a block of size (r, i, c):
    - if r==2 and i==2 and c==2: rank = 7 (Strassen)
    - else: rank = r*i*c (schoolbook)
    """
    # Map (r_size, i_size, c_size) -> rank
    rank_map = {}
    for r_size in range(1, n+1):
        for i_size in range(1, m+1):
            for c_size in range(1, p+1):
                if r_size == 2 and i_size == 2 and c_size == 2:
                    rank_map[(r_size, i_size, c_size)] = 7
                else:
                    rank_map[(r_size, i_size, c_size)] = r_size * i_size * c_size
    return rank_map

def _partition_total_rank(partition, dim_size, rank_map, dim_idx):
    """
    Compute total rank for a partition of one dimension.
    Actually we need the joint rank, so this is just a helper to get block sizes.
    """
    return sum(len(block) for block in partition)

def _try_4x4x4_subtensor(n, m, p):
    """
    Attempt to find a decomposition by embedding 3x4x4 into 4x4x4.
    This is a heuristic: if we can find a rank-R decomposition for 4x4x4,
    we can restrict it to the first 3 rows of the left operand.
    
    However, we don't have a known rank-19 decomposition for 4x4x4 readily available
    in a simple form that we can hardcode and verify quickly within time limits.
    
    Instead, we'll try a simpler structural approach:
    Treat the 3x4x4 multiplication as:
    - Block 1: rows {0,1} x cols {0,1} x inner {0,1} -> 2x2x2 Strassen (rank 7)
    - Block 2: rows {0,1} x cols {2,3} x inner {0,1} -> 2x2x2 Strassen (rank 7)
    - Block 3: rows {0,1} x cols {0,1} x inner {2,3} -> 2x2x2 Strassen (rank 7)
    - Block 4: rows {0,1} x cols {2,3} x inner {2,3} -> 2x2x2 Strassen (rank 7)
    - Row 2: 1x4x4 schoolbook (rank 16)
    
    Total: 7+7+7+7+16 = 44
    """
    # This specific partition:
    # row_blocks: [(0,1), (2,)]
    # inner_blocks: [(0,1), (2,3)]
    # col_blocks: [(0,1), (2,3)]
    
    row_blocks = [(0, 1), (2,)]
    inner_blocks = [(0, 1), (2, 3)]
    col_blocks = [(0, 1), (2, 3)]
    
    candidate = _build_block_decomposition(n, m, p, row_blocks, inner_blocks, col_blocks)
    return candidate

def _try_alternative_partitions(n, m, p):
    """
    Try several alternative block partitions that might yield lower rank.
    """
    best = None
    best_rank = n * m * p
    
    # Partition 1: The one from _try_4x4x4_subtensor
    # row: [(0,1), (2,)], inner: [(0,1), (2,3)], col: [(0,1), (2,3)]
    # This gives 4*7 + 1*16 = 44
    
    # Partition 2: row: [(0,), (1,), (2,)], inner: [(0,1), (2,3)], col: [(0,1), (2,3)]
    # This gives 6 * 4 = 24 (all schoolbook since row blocks are size 1)
    # Wait, 1x2x2 blocks are schoolbook rank 4 each. 6 blocks * 4 = 24.
    # But schoolbook total is 48, so this is already better!
    
    # Let's systematically try partitions where we pair up indices.
    
    partitions_to_try = [
        # (row_blocks, inner_blocks, col_blocks)
        ([(0,1), (2,)], [(0,1), (2,3)], [(0,1), (2,3)]),  # 44
        ([(0,), (1,), (2,)], [(0,1), (2,3)], [(0,1), (2,3)]),  # 24
        ([(0,1), (2,)], [(0,), (1,), (2,3)], [(0,1), (2,3)]),
        ([(0,1), (2,)], [(0,1), (2,3)], [(0,), (1,), (2,3)]),
        ([(0,), (1,2)], [(0,1), (2,3)], [(0,1), (2,3)]),
        ([(0,1), (2,)], [(0,1,2,3)], [(0,1), (2,3)]),
        ([(0,1), (2,)], [(0,1), (2,3)], [(0,1,2,3)]),
        ([(0,1,2)], [(0,1), (2,3)], [(0,1), (2,3)]),
        ([(0,1), (2,)], [(0,1,2,3)], [(0,1,2,3)]),
        ([(0,1,2)], [(0,1,2,3)], [(0,1,2,3)]),
    ]
    
    for rb, ib, cb in partitions_to_try:
        try:
            candidate = _build_block_decomposition(n, m, p, rb, ib, cb)
            rank = len(candidate["U"][0])
            if rank < best_rank:
                if _verify_fast(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                    best = candidate
                    best_rank = rank
        except Exception:
            pass
    
    return best, best_rank

def _construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Precompute partitions for each dimension
    rows = list(range(n))
    inners = list(range(m))
    cols = list(range(p))
    
    row_parts = list(_all_partitions(rows))
    inner_parts = list(_all_partitions(inners))
    col_parts = list(_all_partitions(cols))
    
    # Precompute rank map for quick filtering
    rank_map = _precompute_partition_ranks(n, m, p)
    
    # Compute total rank for each partition combination quickly
    # Total rank = sum over all (rb, ib, cb) of rank_map[(len(rb), len(ib), len(cb))]
    
    # Sort partition combinations by estimated rank (ascending) to find good ones early
    combos = []
    for rp in row_parts:
        for ip in inner_parts:
            for cp in col_parts:
                total_rank = 0
                for rb in rp:
                    for ib in ip:
                        for cb in cp:
                            total_rank += rank_map[(len(rb), len(ib), len(cb))]
                combos.append((total_rank, rp, ip, cp))
    
    # Sort by rank ascending
    combos.sort(key=lambda x: x[0])
    
    # Only verify candidates with rank < best_rank
    for total_rank, rp, ip, cp in combos:
        if time.time() > deadline:
            break
        if total_rank >= best_rank:
            continue  # No point checking higher rank candidates
        try:
            candidate = _build_block_decomposition(n, m, p, rp, ip, cp)
            if _verify_fast(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                best = candidate
                best_rank = total_rank
        except Exception:
            pass
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    try:
        candidate, rank = _construct()
        if rank < best_rank:
            best = candidate
            best_rank = rank
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
