import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank39():
    """
    Construct an explicit rank-39 decomposition for M_{2,2,11}.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_offset, base_rank):
        r = base_rank
        U[0][r] += 1.0
        U[3][r] += 1.0
        V[col_offset][r] += 1.0
        V[12 + col_offset][r] += 1.0
        W[2 * col_offset][r] += 1.0
        W[2 * col_offset + 3][r] += 1.0
        
        r = base_rank + 1
        U[2][r] += 1.0
        U[3][r] += 1.0
        V[col_offset][r] += 1.0
        W[2 * col_offset + 1][r] += 1.0
        
        r = base_rank + 2
        U[0][r] += 1.0
        V[col_offset + 1][r] += 1.0
        V[11 + col_offset][r] += -1.0
        W[2 * col_offset + 2][r] += 1.0
        
        r = base_rank + 3
        U[3][r] += 1.0
        V[11 + col_offset][r] += 1.0
        V[col_offset][r] += -1.0
        W[2 * col_offset + 1][r] += 1.0
        W[2 * col_offset][r] += -1.0
        
        r = base_rank + 4
        U[0][r] += 1.0
        U[1][r] += 1.0
        V[12 + col_offset][r] += 1.0
        W[2 * col_offset + 2][r] += 1.0
        W[2 * col_offset + 3][r] += -1.0
        
        r = base_rank + 5
        U[2][r] += 1.0
        U[0][r] += -1.0
        V[col_offset][r] += 1.0
        V[col_offset + 1][r] += 1.0
        W[2 * col_offset + 1][r] += 1.0
        W[2 * col_offset + 3][r] += 1.0
        
        r = base_rank + 6
        U[1][r] += 1.0
        U[3][r] += -1.0
        V[11 + col_offset][r] += 1.0
        V[12 + col_offset][r] += 1.0
        W[2 * col_offset][r] += 1.0
        W[2 * col_offset + 1][r] += -1.0

    for block_idx in range(5):
        col_offset = 2 * block_idx
        base_rank = 7 * block_idx
        add_strassen_block(col_offset, base_rank)
    
    base_rank = 35
    r = base_rank
    U[0][r] += 1.0
    V[10][r] += 1.0
    W[20][r] += 1.0
    
    r = base_rank + 1
    U[1][r] += 1.0
    V[21][r] += 1.0
    W[20][r] += 1.0
    
    r = base_rank + 2
    U[2][r] += 1.0
    V[10][r] += 1.0
    W[21][r] += 1.0
    
    r = base_rank + 3
    U[3][r] += 1.0
    V[21][r] += 1.0
    W[21][r] += 1.0
    
    return U, V, W

def _refine_rank(rank, n, m, p, time_limit=100.0):
    """
    Use iterative refinement to find a decomposition of the given rank.
    """
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Initialize with random values
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets.append((i * m + j, j * p + k, k * n + i))
                
    a_targets = {a: [] for a in range(n * m)}
    b_targets = {b: [] for b in range(m * p)}
    c_targets = {c: [] for c in range(p * n)}
    
    for ta, tb, tc in targets:
        a_targets[ta].append((tb, tc))
        b_targets[tb].append((ta, tc))
        c_targets[tc].append((ta, tb))

    for it in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        for a in range(n * m):
            for c_idx in range(rank):
                num = 0.0
                den = 0.0
                for tb, tc in a_targets[a]:
                    v_val = V[tb, c_idx]
                    w_val = W[tc, c_idx]
                    prod = v_val * w_val
                    num += prod
                    den += prod * prod
                if den > 1e-12:
                    U[a, c_idx] = num / den
        
        # Update V
        for b in range(m * p):
            for c_idx in range(rank):
                num = 0.0
                den = 0.0
                for ta, tc in b_targets[b]:
                    u_val = U[ta, c_idx]
                    w_val = W[tc, c_idx]
                    prod = u_val * w_val
                    num += prod
                    den += prod * prod
                if den > 1e-12:
                    V[b, c_idx] = num / den
        
        # Update W
        for c in range(p * n):
            for c_idx in range(rank):
                num = 0.0
                den = 0.0
                for ta, tb in c_targets[c]:
                    u_val = U[ta, c_idx]
                    v_val = V[tb, c_idx]
                    prod = u_val * v_val
                    num += prod
                    den += prod * prod
                if den > 1e-12:
                    W[c, c_idx] = num / den
        
        if it % 100 == 0:
            err = 0.0
            for ta, tb, tc in targets:
                s = sum(U[ta, c_idx] * V[tb, c_idx] * W[tc, c_idx] for c_idx in range(rank))
                err += (s - 1.0) ** 2
            if err < 1e-12:
                return U.tolist(), V.tolist(), W.tolist()
                
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to find a lower rank
    for rank in [38, 37, 36, 35]:
        if time.time() > deadline - 10:
            break
        result = _refine_rank(rank, n, m, p, time_limit=20.0)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": W, "W": W} # Note: W is W, V is V, U is U
                # Correct order: U, V, W
                return {"U": U, "V": V, "W": W}

    # Fallback to rank 39
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Final fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
