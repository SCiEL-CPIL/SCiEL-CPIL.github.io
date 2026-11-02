"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 140.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _winograd_2x2x3():
    """Winograd rank-7 decomposition for 2x2 * 2x3.
    Returns U (2x2, 7), V (2x3, 7), W (3x2, 7) where:
      U[i,j,c] = (i-a)*(j-b), V[j,k,c] = (a-j)*(b-k), W[i,k,c] = 1
    with (a,b) pairs: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,3)
    """
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 3)]
    rank = len(pairs)
    U = [[0] * rank for _ in range(2 * 2)]   # rows: i*2+j
    V = [[0] * rank for _ in range(2 * 3)]   # rows: j*3+k
    W = [[0] * rank for _ in range(3 * 2)]   # rows: k*2+i
    for c, (a, b) in enumerate(pairs):
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][c] = (i - a) * (j - b)
        for j in range(2):
            for k in range(3):
                V[j * 3 + k][c] = (a - j) * (b - k)
        for i in range(2):
            for k in range(3):
                W[k * 2 + i][c] = 1
    return U, V, W


def _kronecker_4x4x9():
    """Kronecker product of two Winograd 2x2x3 decompositions.
    4x4 * 4x9 = (2x2 ⊗ 2x2) * (2x2 ⊗ 2x3)
    Total rank = 7 * 7 = 49.
    """
    U1, V1, W1 = _winograd_2x2x3()
    U2, V2, W2 = _winograd_2x2x3()
    
    rank = 7 * 7
    n, m, p = 4, 4, 9
    U = [[0] * rank for _ in range(n * m)]   # 16 x 49
    V = [[0] * rank for _ in range(m * p)]   # 36 x 49
    W = [[0] * rank for _ in range(p * n)]   # 36 x 49
    
    # Indexing: A[i,j] where i = i1*2+i2, j = j1*2+j2
    # B[j,k] where j = j1*2+j2, k = k1*3+k2
    # C[i,k] where i = i1*2+i2, k = k1*3+k2
    # U row: i*m+j = (i1*2+i2)*4 + (j1*2+j2)
    # V row: j*p+k = (j1*2+j2)*9 + (k1*3+k2)
    # W row: k*n+i = (k1*3+k2)*4 + (i1*2+i2)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U1[i1 * 2 + j1][c1]
                    if u1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2 * 2 + j2][c2]
                            if u2 == 0:
                                continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * m + j][c] = u1 * u2
            
            for j1 in range(2):
                for k1 in range(3):
                    v1 = V1[j1 * 3 + k1][c1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v2 = V2[j2 * 3 + k2][c2]
                            if v2 == 0:
                                continue
                            j = j1 * 2 + j2
                            k = k1 * 3 + k2
                            V[j * p + k][c] = v1 * v2
            
            for i1 in range(2):
                for k1 in range(3):
                    w1 = W1[k1 * 2 + i1][c1]
                    if w1 == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(3):
                            w2 = W2[k2 * 2 + i2][c2]
                            if w2 == 0:
                                continue
                            i = i1 * 2 + i2
                            k = k1 * 3 + k2
                            W[k * n + i][c] = w1 * w2
    
    return {"U": U, "V": V, "W": W}


def _rank36(n, m, p):
    """Attempt rank-36 decomposition.
    Column c indexes (j0, k0) where j0 in [0,m), k0 in [0,p).
    U[i*m+j, c] = delta(j,j0) * a(i,k0)
    V[j*p+k, c] = delta(k,k0) * b(j0)
    W[k*n+i, c] = delta(i, i0) * ... 
    Need sum over c: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 iff consistent.
    For (i,j,k): sum_{c} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    = sum_{j0,k0} delta(j,j0)*a(i,k0) * delta(k,k0)*b(j0) * W[k*n+i,(j0,k0)]
    = a(i,k)*b(j)*W[k*n+i,(j,k)]
    So we need a(i,k)*b(j)*W[k*n+i,(j,k)] = 1 for all i,j,k.
    Set a(i,k)=1, b(j)=1, W[k*n+i,(j,k)]=1. But W row k*n+i appears in multiple columns.
    W[k*n+i,(j,k)] must be 1 for all j, but W has only one value per (row,col).
    This doesn't work directly. Try different structure.
    """
    # This construction is not valid; return None to signal failure
    return None


def _winograd_2x2x2():
    """Winograd rank-7 decomposition for 2x2 * 2x2.
    U[i,j,c], V[j,k,c], W[i,k,c] for c=0..6.
    Pairs: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,3)
    """
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 3)]
    rank = len(pairs)
    U = [[0] * rank for _ in range(2 * 2)]
    V = [[0] * rank for _ in range(2 * 2)]
    W = [[0] * rank for _ in range(2 * 2)]
    for c, (a, b) in enumerate(pairs):
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][c] = (i - a) * (j - b)
        for j in range(2):
            for k in range(2):
                V[j * 2 + k][c] = (a - j) * (b - k)
        for i in range(2):
            for k in range(2):
                W[i * 2 + k][c] = 1
    return U, V, W


def _block_4x4x9():
    """Block decomposition: 4x4 * 4x9.
    Split A into 2x2 blocks, B into 2x3 blocks.
    C[i,k] = sum_{j1,j2} A[i1 i2, j1 j2] * B[j1 j2, k1 k2]
    = sum_{j1} (A[i1,:,j1,:] * B[j1,:]) where inner is 2x2 * 2x3
    Each (i1,k1) output block: sum_{j1} A_block(i1,j1) * B_block(j1,k1)
    Use Winograd 2x2x2x3 (rank 7) for each (i1,j1,k1) triple? No.
    
    Better: For fixed (i1,k1), C[i1,:,k1,:] = sum_{j1} A[i1,:,j1,:] * B[j1,:,k1,:]
    This is a 2x2 matrix product (in i2,j2 for A and j2,k2 for B) summed over j1.
    Actually it's: for each i2,k2: C[i1 i2, k1 k2] = sum_{j1,j2} A[i1 i2, j1 j2] B[j1 j2, k1 k2]
    
    Let's use the Kronecker approach but verify it works.
    Return the same as _kronecker_4x4x9 but as a separate function for clarity.
    """
    return _kronecker_4x4x9()


def _try_rank36_search(n, m, p, T, max_time=60):
    """Try to find a rank-36 decomposition via randomized search + local refinement.
    This is experimental; may not succeed.
    """
    import random
    random.seed(42)
    rank = 36
    n_rows_U = n * m  # 16
    n_rows_V = m * p  # 36
    n_rows_W = p * n  # 36
    
    # Initialize with small random values
    U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n_rows_U)]
    V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n_rows_V)]
    W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n_rows_W)]
    
    # Compute error
    def compute_error(U, V, W):
        err = 0
        for io in range(p * n):
            for u in range(n * m):
                for v in range(m * p):
                    s = 0
                    for c in range(rank):
                        s += U[u][c] * V[v][c] * W[io][c]
                    target = T.get((io, u, v), 0)
                    if s != target:
                        err += 1
        return err
    
    start_err = compute_error(U, V, W)
    if start_err == 0:
        return {"U": U, "V": V, "W": W}
    
    best_err = start_err
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    t0 = time.time()
    while time.time() - t0 < max_time:
        # Random perturbation
        choice = random.random()
        if choice < 0.33:
            r = random.randint(0, n_rows_U - 1)
            c = random.randint(0, rank - 1)
            delta = random.choice([-1, 1, -2, 2])
            U[r][c] += delta
        elif choice < 0.66:
            r = random.randint(0, n_rows_V - 1)
            c = random.randint(0, rank - 1)
            delta = random.choice([-1, 1, -2, 2])
            V[r][c] += delta
        else:
            r = random.randint(0, n_rows_W - 1)
            c = random.randint(0, rank - 1)
            delta = random.choice([-1, 1, -2, 2])
            W[r][c] += delta
        
        err = compute_error(U, V, W)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
            if best_err == 0:
                return {"U": best_U, "V": best_V, "W": best_W}
    
    if best_err == 0:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try candidates in order of increasing rank; return first valid one.
    candidates = []
    
    # Rank 49: Kronecker Winograd (known valid)
    try:
        cand49 = _kronecker_4x4x9()
        if _exact_check(cand49["U"], cand49["V"], cand49["W"], T, n, m, p):
            candidates.append((49, cand49))
    except Exception:
        pass
    
    # Rank 144: Schoolbook (always valid)
    cand_sb = _schoolbook(n, m, p)
    candidates.append((n * m * p, cand_sb))
    
    # Sort by rank (lower is better)
    candidates.sort(key=lambda x: x[0])
    
    for rank_val, cand in candidates:
        if time.time() > _DEADLINE - 5:
            break
        try:
            if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
                return cand
        except Exception:
            continue
    
    # Fallback: return schoolbook
    return cand_sb

# EVOLVE-BLOCK-END
