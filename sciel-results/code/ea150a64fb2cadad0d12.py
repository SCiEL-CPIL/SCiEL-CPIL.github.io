import time
import numpy as np

def _partition_indices(count, max_block=2):
    """Generate all partitions of range(count) into blocks of size 1 or max_block."""
    if count == 0:
        yield []
        return
    # First block is size 1
    for rest in _partition_indices(count - 1, max_block):
        yield [[0]] + rest
    # First block is size 2 (if count >= 2 and max_block >= 2)
    if count >= 2 and max_block >= 2:
        for rest in _partition_indices(count - 2, max_block):
            yield [[0, 1]] + rest

def _strassen_222():
    """Return the 7 rank-1 terms of the 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u,v,w are 4-element lists (flattened 2x2)."""
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]

def _construct_for_partition(part_n, part_m, part_p, n, m, p):
    """Construct U,V,W for given partitions. Returns (rank, U, V, W) or None."""
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    col = 0
    strassen_terms = _strassen_222()
    
    for bn in part_n:
        for bm in part_m:
            for bp in part_p:
                sn, sm, sp = len(bn), len(bm), len(bp)
                if sn == 2 and sm == 2 and sp == 2:
                    i0, j0, k0 = bn[0], bm[0], bp[0]
                    for (u, v, w) in strassen_terms:
                        for di in range(2):
                            for dj in range(2):
                                U[(i0+di)*m + (j0+dj), col] = u[di*2+dj]
                        for dj in range(2):
                            for dk in range(2):
                                V[(j0+dj)*p + (k0+dk), col] = v[dj*2+dk]
                        for di in range(2):
                            for dk in range(2):
                                W[(k0+dk)*n + (i0+di), col] = w[di*2+dk]
                        col += 1
                else:
                    for i in bn:
                        for j in bm:
                            for k in bp:
                                U[i*m + j, col] = 1
                                V[j*p + k, col] = 1
                                W[k*n + i, col] = 1
                                col += 1
    
    if col == 0:
        return None
    return col, U[:, :col], V[:, :col], W[:, :col]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    U, V, W = np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 130
    
    best_rank = n * m * p
    best_result = None
    
    part_n = list(_partition_indices(n))
    part_m = list(_partition_indices(m))
    part_p = list(_partition_indices(p))
    
    for pn in part_n:
        if time.time() > deadline:
            break
        for pm in part_m:
            if time.time() > deadline:
                break
            for pp in part_p:
                if time.time() > deadline:
                    break
                res = _construct_for_partition(pn, pm, pp, n, m, p)
                if res is None:
                    continue
                rank, U, V, W = res
                if _verify(U, V, W, n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_result is None:
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return best_result
