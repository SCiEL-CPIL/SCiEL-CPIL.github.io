"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p).
    The tensor has dimensions (n*m, m*p, p*n) indexed by (i*m+j, j*p+k, k*n+i)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    The decomposition satisfies: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    """
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # c=0: M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # c=2: M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # c=3: M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # c=4: M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # c=5: M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # c=6: M7 = (b-d)(e+g)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Build a decomposition using a mix of Strassen (2x2x2) and schoolbook blocks.
    
    n_blocks, m_blocks, p_blocks: lists of tuples of indices.
    For each triple (nb, mb, pb):
      - if len(nb)==2 and len(mb)==2 and len(pb)==2: use Strassen (rank 7)
      - else: use schoolbook (rank = sn*sm*sp)
    
    Returns (U, V, W, rank) or None if invalid.
    """
    total_rank = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    total_rank += 7
                else:
                    total_rank += sn * sm * sp
    
    if total_rank >= n * m * p:
        return None
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    U_s, V_s, W_s = _strassen_2x2x2()
                    for ci in range(sn):
                        for cj in range(sm):
                            for ck in range(sp):
                                i = nb[ci]
                                j = mb[cj]
                                k = pb[ck]
                                for c in range(7):
                                    U[i * m + j, col + c] = U_s[ci * 2 + cj, c]
                                    V[j * p + k, col + c] = V_s[cj * 2 + ck, c]
                                    W[k * n + i, col + c] = W_s[ck * 2 + ci, c]
                    col += 7
                else:
                    for ci in range(sn):
                        for cj in range(sm):
                            for ck in range(sp):
                                i = nb[ci]
                                j = mb[cj]
                                k = pb[ck]
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
    
    if col != total_rank:
        return None
    
    if not _check_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W, total_rank

def _generate_partitions(n):
    """Generate all ways to partition range(n) into blocks of size 1 or 2 (not necessarily contiguous).
    Returns a list of partitions, where each partition is a list of tuples of indices."""
    results = []
    used = [False] * n
    
    def backtrack(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        # Find the smallest unused index
        first = -1
        for i in range(n):
            if not used[i]:
                first = i
                break
        if first == -1:
            return
        
        # Option 1: first as singleton
        used[first] = True
        current.append((first,))
        backtrack(remaining - 1, current)
        current.pop()
        used[first] = False
        
        # Option 2: pair first with another unused index
        for j in range(first + 1, n):
            if not used[j]:
                used[first] = True
                used[j] = True
                current.append((first, j))
                backtrack(remaining - 2, current)
                current.pop()
                used[first] = False
                used[j] = False
    
    backtrack(n, [])
    return results

def _build_strassen_3x2x2():
    """Build a rank-10 decomposition for 3x2x2 matrix multiplication.
    
    Uses Strassen for the first 2 rows of A, and schoolbook for the third row.
    U: (6, 10), V: (4, 10), W: (6, 10)
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where i = i1*2+i2 (i1 in {0,1,2}, i2 in {0,1}), j = j1*2+j2, k = k1*2+k2
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U = np.zeros((6, 10), dtype=float)
    V = np.zeros((4, 10), dtype=float)
    W = np.zeros((6, 10), dtype=float)
    
    # First 2 rows: use Strassen (columns 0-6)
    U[:4, :7] = U_s
    V[:, :7] = V_s
    W[:4, :7] = W_s
    
    # Third row: schoolbook (columns 7-9)
    # For i=2 (third row of A), we need T[2,j,k] = A[2,j] * B[j,k]
    # This is just the outer product of row 2 of A with each element of B
    for j in range(2):
        for k in range(2):
            idx = 7 + j * 2 + k
            U[4 + j, idx] = 1  # i=2, j=j
            V[j * 2 + k, idx] = 1
            W[k * 2 + 2, idx] = 1  # k=k, i=2
    
    return U, V, W

def _build_strassen_2x3x2():
    """Build a rank-10 decomposition for 2x3x2 matrix multiplication.
    
    Uses Strassen for the first 2 cols of B, and schoolbook for the third col.
    U: (6, 10), V: (6, 10), W: (4, 10)
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where i = i1*2+i2, j = j1*3+j2 (j1 in {0,1}, j2 in {0,1,2}), k = k1*2+k2
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U = np.zeros((6, 10), dtype=float)
    V = np.zeros((6, 10), dtype=float)
    W = np.zeros((4, 10), dtype=float)
    
    # First 2 cols of B: use Strassen (columns 0-6)
    U[:, :7] = U_s
    V[:4, :7] = V_s
    W[:, :7] = W_s
    
    # Third col of B: schoolbook (columns 7-9)
    # For j=2 (third col of B), we need T[i,2,k] = sum_l A[i,l] * B[2,k]
    # This is just the outer product of each element of A with row 2 of B
    for i in range(2):
        for k in range(2):
            idx = 7 + i * 2 + k
            U[i * 2 + 0, idx] = 1  # i=i, j=0 (first col)
            V[0 * 2 + 0, idx] = 1  # j=0, k=0 (placeholder, will be overwritten)
            # Actually, we need to be more careful. Let me redo this.
    
    # Let me rebuild this more carefully
    U = np.zeros((6, 10), dtype=float)
    V = np.zeros((6, 10), dtype=float)
    W = np.zeros((4, 10), dtype=float)
    
    # Strassen part for j in {0,1}
    U[:, :7] = U_s
    V[:4, :7] = V_s
    W[:, :7] = W_s
    
    # Schoolbook part for j=2
    # T[i, 2, k] = A[i, 2] * B[2, k]
    # We need: sum_c U[i,c] * V[2,c] * W[k,c] = A[i,2] * B[2,k]
    # So we can set for each (i,k): U[i, 7+i*2+k] = 1, V[2, 7+i*2+k] = A[i,2] (but A is variable!)
    # Wait, this won't work because the decomposition must be fixed, not depend on A.
    # 
    # Actually, for schoolbook, we have: T[i,j,k] = 1 if (i,j,k) is a valid triple
    # So for j=2: T[i,2,k] = 1 for all i,k
    # We need: sum_c U[i,c] * V[2,c] * W[k,c] = 1
    # One way: for each (i,k), have a separate term where U[i,c]=1, V[2,c]=1, W[k,c]=1
    # But that's 4 terms (2x2), which is what we want.
    
    for i in range(2):
        for k in range(2):
            idx = 7 + i * 2 + k
            U[i * 2 + 0, idx] = 1  # i=i, j=0 (but we need j=2!)
            # Oops, U is indexed by (i,j), so U[i*2+j, c]
            # For j=2: U[i*2+2, c]
            U[i * 2 + 2, idx] = 1
            V[2 * 2 + 0, idx] = 1  # j=2, k=0? No, V is indexed by (j,k)
            # V[j*2+k, c], so for j=2, k=k: V[2*2+k, c] = V[4+k, c]
            V[4 + k, idx] = 1
            W[k * 2 + 0, idx] = 1  # k=k, i=0? No, W is indexed by (k,i)
            # W[k*2+i, c], so for k=k, i=i: W[k*2+i, c]
            W[k * 2 + i, idx] = 1
    
    return U, V, W

def _build_strassen_2x2x3():
    """Build a rank-10 decomposition for 2x2x3 matrix multiplication.
    
    Uses Strassen for the first 2 cols of C, and schoolbook for the third col.
    U: (4, 10), V: (6, 10), W: (6, 10)
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where i = i1*2+i2, j = j1*2+j2, k = k1*3+k2 (k1 in {0,1}, k2 in {0,1,2})
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U = np.zeros((4, 10), dtype=float)
    V = np.zeros((6, 10), dtype=float)
    W = np.zeros((6, 10), dtype=float)
    
    # Strassen part for k in {0,1}
    U[:, :7] = U_s
    V[:, :7] = V_s
    W[:4, :7] = W_s
    
    # Schoolbook part for k=2
    # T[i,j,2] = A[i,j] * B[j,2]
    # We need: sum_c U[i,c] * V[j,c] * W[2,c] = 1
    # For each (i,j): U[i,c]=1, V[j,c]=1, W[2,c]=1
    for i in range(2):
        for j in range(2):
            idx = 7 + i * 2 + j
            U[i * 2 + j, idx] = 1
            V[j * 2 + 0, idx] = 1  # j=j, k=0? No, V is (j,k)
            # V[j*2+k, c], for k=0: V[j*2, c]
            V[j * 2 + 0, idx] = 1
            W[2 * 2 + 0, idx] = 1  # k=2, i=0? No, W is (k,i)
            # W[k*2+i, c], for k=2, i=i: W[2*2+i, c] = W[4+i, c]
            W[4 + i, idx] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Generate all partitions for n, m, p
    n_parts = _generate_partitions(n)
    m_parts = _generate_partitions(m)
    p_parts = _generate_partitions(p)
    
    # Sort partitions by estimated rank (fewer 2-blocks = lower potential rank)
    def _partition_score(parts):
        # Count number of 2-blocks; more 2-blocks = more potential for Strassen
        num_2blocks = sum(1 for block in parts if len(block) == 2)
        return -num_2blocks  # Sort ascending: most 2-blocks first
    
    n_parts.sort(key=_partition_score)
    m_parts.sort(key=_partition_score)
    p_parts.sort(key=_partition_score)
    
    # Try all combinations of partitions, prioritizing those with more 2-blocks
    found_better = False
    for nb in n_parts:
        if found_better:
            break
        for mb in m_parts:
            if found_better:
                break
            for pb in p_parts:
                elapsed = time.time() - start_time
                if elapsed > deadline - 5:
                    break
                # Quick rank estimate before building
                est_rank = 0
                for nb_i in nb:
                    for mb_i in mb:
                        for pb_i in pb:
                            sn, sm, sp = len(nb_i), len(mb_i), len(pb_i)
                            if sn == 2 and sm == 2 and sp == 2:
                                est_rank += 7
                            else:
                                est_rank += sn * sm * sp
                
                if est_rank >= best_rank:
                    continue
                
                result = _build_block_decomposition(n, m, p, nb, mb, pb)
                if result is not None:
                    U, V, W, rank = result
                    if rank < best_rank:
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        found_better = True
                        break
            if found_better:
                break
        if found_better:
            break
    
    # Verify final result
    if _check_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
