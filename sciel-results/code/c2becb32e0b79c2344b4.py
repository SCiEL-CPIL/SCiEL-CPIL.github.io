"""Matrix multiplication 4x5 by 5x6 — Hybrid Strassen/schoolbook decomposition with rank reduction."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    Each decomposition is a tuple: (U_sub, V_sub, W_sub, i_start, j_start, k_start, i_size, j_size, k_size)
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        # Map A indices: i in [0, i_sz), j in [0, j_sz)
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        # Map B indices: j in [0, j_sz), k in [0, k_sz)
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        # Map C indices: k in [0, k_sz), i in [0, i_sz)
        # Convention: W[k*n + i]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_4x4x2_strassen():
    """
    Build a rank-28 decomposition for 4x4 * 4x2 -> 4x2.
    Uses 4 Strassen 2x2x2 blocks.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 28
    n_sub, m_sub, p_sub = 4, 4, 2
    U = [[0]*rank for _ in range(n_sub * m_sub)]
    V = [[0]*rank for _ in range(m_sub * p_sub)]
    W = [[0]*rank for _ in range(p_sub * n_sub)]
    
    for ai in range(2):
        for bj in range(2):
            block_offset = (ai * 2 + bj) * 7
            for t in range(7):
                col = block_offset + t
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j][t]
                        if val != 0:
                            u_row = (2*ai + i) * m_sub + (2*bj + j)
                            U[u_row][col] += val
                
                for j in range(2):
                    for k in range(2):
                        val = V2[j*2+k][t]
                        if val != 0:
                            v_row = (2*bj + j) * p_sub + k
                            V[v_row][col] += val
                
                for k in range(2):
                    for i in range(2):
                        val = W2[k*2+i][t]
                        if val != 0:
                            w_row = k * n_sub + (2*ai + i)
                            W[w_row][col] += val
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build a rank-49 decomposition for 4x4 * 4x4 -> 4x4 using Kronecker product of Strassen.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1][t1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2][t2]
                            if c2 == 0: continue
                            u_row = (i1*2+i2)*4 + (j1*2+j2)
                            U[u_row][col] += c1 * c2
            
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1][t1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2][t2]
                            if c2 == 0: continue
                            v_row = (j1*2+j2)*4 + (k1*2+k2)
                            V[v_row][col] += c1 * c2
            
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1][t1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2][t2]
                            if c2 == 0: continue
                            w_row = (k1*2+k2)*4 + (i1*2+i2)
                            W[w_row][col] += c1 * c2
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-8):
    """Fast vectorized verification using einsum."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = len(U_np[0])
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _remove_linearly_dependent_columns(U, V, W, n, m, p, tol=1e-9):
    """
    Attempt to remove columns that are linear combinations of others.
    Returns new U, V, W and the list of kept column indices.
    """
    rank = len(U[0])
    if rank == 0:
        return U, V, W, []
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the matrix M where each column r is the flattened rank-1 tensor M_r
    # M_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    M_list = []
    for r in range(rank):
        M_r = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M_r[i*m*p + j*p + k] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
        M_list.append(M_r)
    
    M_mat = np.array(M_list).T  # Shape (n*m*p, rank)
    
    # Find the rank of M_mat
    try:
        U_svd, s, Vt = np.linalg.svd(M_mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W, list(range(rank))
    
    num_nonzero = np.sum(s > tol)
    
    if num_nonzero >= rank:
        # No linear dependencies
        return U, V, W, list(range(rank))
    
    # There are linear dependencies. We need to find a minimal subset that still sums to the target.
    # The target tensor T has T[i,j,k] = 1 for all i,j,k.
    target = np.ones(n * m * p)
    
    # We want to find the sparsest set of columns whose sum equals target.
    # This is NP-hard in general, so we use a greedy approach.
    
    # First, check if the current decomposition is valid
    current_sum = M_mat.sum(axis=1)
    if not np.allclose(current_sum, target, atol=tol):
        # Decomposition is not valid, return as-is
        return U, V, W, list(range(rank))
    
    # Greedy removal: try to remove columns one by one if the remaining columns can still sum to target
    kept = list(range(rank))
    
    # Try each column in order of smallest norm (more likely to be removable)
    norms = np.linalg.norm(M_mat, axis=0)
    order = np.argsort(norms)
    
    for r in order:
        if r not in kept:
            continue
        
        # Try removing column r
        kept_test = [x for x in kept if x != r]
        
        # Check if the sum of remaining columns equals target
        if len(kept_test) == 0:
            continue
            
        sum_test = M_mat[:, kept_test].sum(axis=1)
        if np.allclose(sum_test, target, atol=tol):
            # We can remove column r
            kept = kept_test
    
    # Build new U, V, W with only kept columns
    if len(kept) < rank:
        new_U = [[U[i][r] for r in kept] for i in range(len(U))]
        new_V = [[V[i][r] for r in kept] for i in range(len(V))]
        new_W = [[W[i][r] for r in kept] for i in range(len(W))]
        return new_U, new_V, new_W, kept
    
    return U, V, W, list(range(rank))

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140

    decomps = []
    
    # Block 1: 4x4x4 (i=0..3, j=0..3, k=0..3)
    U4, V4, W4 = _build_4x4x4_strassen()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    
    # Block 2: 4x4x2 (i=0..3, j=0..3, k=4..5)
    U5, V5, W5 = _build_4x4x2_strassen()
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    
    # Block 3: 4x1x6 (i=0..3, j=4, k=0..5)
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Verify correctness
    if not _verify_decomp(U, V, W, n, m, p):
        # Fallback to pure schoolbook if Strassen construction failed
        U, V, W = _build_schoolbook(n, m, p)
    
    # Attempt rank reduction
    U, V, W, kept = _remove_linearly_dependent_columns(U, V, W, n, m, p)
    
    # Verify again after reduction
    if not _verify_decomp(U, V, W, n, m, p):
        # If reduction broke validity, use original
        U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
