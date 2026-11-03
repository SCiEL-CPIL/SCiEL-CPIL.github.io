"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    # Check T[i*m+j, j*p+k, k*n+i] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i*m+j, j*p+k, k*n+i)
                if abs(T[idx] - 1.0) > tol:
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

def _all_set_partitions(elements):
    """Generate all set partitions of a list of elements.
    Returns a list of partitions, where each partition is a list of frozensets.
    We return them as lists of sorted lists for consistent ordering.
    """
    n = len(elements)
    if n == 0:
        return [[]]
    
    # Use recursive approach: pick first element, try adding to each existing block or creating new block
    from functools import lru_cache
    
    def _partitions_of_range(k):
        """All partitions of {0,1,...,k-1} as list of tuples (sorted)."""
        if k == 0:
            return [()]
        if k == 1:
            return [((0,),)]
        
        result = []
        # Get all partitions of {0,...,k-2}
        prev = _partitions_of_range(k-1)
        for part in prev:
            # For each partition, try adding element k-1 to each existing block
            for bi in range(len(part)):
                block = part[bi]
                new_block = tuple(sorted(block + (k-1,)))
                new_part = list(part)
                new_part[bi] = new_block
                # Sort blocks by their first element for canonical form
                new_part.sort(key=lambda b: b[0])
                result.append(tuple(new_part))
            # Try creating a new block with just k-1
            new_part = list(part)
            new_part.append((k-1,))
            new_part.sort(key=lambda b: b[0])
            result.append(tuple(new_part))
        
        # Deduplicate
        seen = set()
        unique = []
        for p in result:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique
    
    # Get partitions of range(n)
    range_parts = _partitions_of_range(n)
    
    # Convert to actual element sets
    result = []
    for part in range_parts:
        actual = [sorted([elements[i] for i in block]) for block in part]
        # Sort blocks by their first element for canonical ordering
        actual.sort(key=lambda b: b[0])
        result.append(actual)
    
    return result

def _build_block_decomposition(n, m, p, row_blocks, inner_blocks, col_blocks):
    """
    Build a decomposition by applying Strassen to 2x2x2 blocks where possible.
    row_blocks: partition of rows {0..n-1} (list of lists)
    inner_blocks: partition of inner {0..m-1} (list of lists)
    col_blocks: partition of cols {0..p-1} (list of lists)
    
    For each combination of (row_block, inner_block, col_block):
    - If all sizes are 2, use Strassen rank-7
    - Otherwise, use schoolbook for that sub-block
    """
    # Build the full decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # For each combination of (row_block_id, inner_block_id, col_block_id)
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
                    for i_val in rb:
                        for j_val in ib:
                            for k_val in cb:
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
    if rank == 0:
        return None
    
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

def _try_structured_search(n, m, p, deadline):
    """Try all set partitions to find rank reductions."""
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Generate all set partitions
    row_indices = list(range(n))
    inner_indices = list(range(m))
    col_indices = list(range(p))
    
    row_partitions = _all_set_partitions(row_indices)
    inner_partitions = _all_set_partitions(inner_indices)
    col_partitions = _all_set_partitions(col_indices)
    
    # Sort partitions by potential rank (prefer those with more 2x2x2 blocks)
    def _count_strassen_blocks(rp, ip, cp):
        count = 0
        for rb in rp:
            for ib in ip:
                for cb in cp:
                    if len(rb) == 2 and len(ib) == 2 and len(cb) == 2:
                        count += 1
        return count
    
    # Generate all combinations and sort by potential improvement
    all_combos = []
    for rp in row_partitions:
        for ip in inner_partitions:
            for cp in col_partitions:
                s_count = _count_strassen_blocks(rp, ip, cp)
                # Estimate rank: for each block combo, if 2x2x2 use 7, else use product
                est_rank = 0
                for rb in rp:
                    for ib in ip:
                        for cb in cp:
                            if len(rb) == 2 and len(ib) == 2 and len(cb) == 2:
                                est_rank += 7
                            else:
                                est_rank += len(rb) * len(ib) * len(cb)
                all_combos.append((est_rank, s_count, rp, ip, cp))
    
    # Sort by estimated rank (ascending), then by number of Strassen blocks (descending)
    all_combos.sort(key=lambda x: (x[0], -x[1]))
    
    for est_rank, s_count, rp, ip, cp in all_combos:
        if time.time() > deadline:
            break
        if est_rank >= best_rank:
            # No point trying if estimated rank is already >= best
            continue
        try:
            candidate = _build_block_decomposition(n, m, p, rp, ip, cp)
            if candidate is None:
                continue
            rank = len(candidate["U"][0])
            if rank < best_rank:
                if _verify_fast(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                    best = candidate
                    best_rank = rank
        except Exception:
            pass
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured block decomposition search
    try:
        candidate, rank = _try_structured_search(n, m, p, deadline)
        if rank < best_rank:
            best = candidate
            best_rank = rank
    except Exception:
        pass
    
    if time.time() > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END
