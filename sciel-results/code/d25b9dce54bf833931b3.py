"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    
    Strategy: Split B (3x8) into blocks:
    - B_top: rows 0,1 (2x8) -> A_top (2x2) * B_top (2x8) using Strassen per 2-col block
    - B_bottom: row 2 (1x8) -> A_bottom (2x1) * B_bottom (1x8) as outer products
    
    A_top * B_top: 4 blocks of 2x2*2x2, each rank 7 (Strassen) = 28
    A_bottom * B_bottom: 2*8 = 16 rank-1 terms (outer product)
    Total: 28 + 16 = 44
    """
    n, m, p = 2, 3, 8
    rank_total = 44
    U = np.zeros((n*m, rank_total))  # 6 x 44
    V = np.zeros((m*p, rank_total))  # 24 x 44
    W = np.zeros((p*n, rank_total))  # 16 x 44
    
    col = 0
    
    # Part 1: A_top (2x2, cols 0,1) * B_top (2x8, rows 0,1)
    # A_top indices: i*3+j for i in {0,1}, j in {0,1}
    # B_top indices: j*8+k for j in {0,1}, k in {0..7}
    # C indices: k*2+i for k in {0..7}, i in {0,1}
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for 2x2 * 2x2
        # a00 = A[0,0] -> U[0], a01 = A[0,1] -> U[1]
        # a10 = A[1,0] -> U[3], a11 = A[1,1] -> U[4]
        # b00 = B[0,k0] -> V[k0], b01 = B[0,k1] -> V[k1]
        # b10 = B[1,k0] -> V[8+k0], b11 = B[1,k1] -> V[8+k1]
        # C00 -> W[2*k0], C01 -> W[2*k1]
        # C10 -> W[2*k0+1], C11 -> W[2*k1+1]
        
        # m1 = (a00+a11)(b00+b11) -> contributes to C00 and C11
        t = col
        U[0, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        W[2*k1+1, t] += 1
        col += 1
        
        # m2 = (a10+a11)b00 -> contributes to C10 and -C11
        t = col
        U[3, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1
        W[2*k1+1, t] += -1
        col += 1
        
        # m3 = a00(b01-b11) -> contributes to C01 and C11
        t = col
        U[0, t] += 1
        V[k1, t] += 1
        V[8+k1, t] += -1
        W[2*k1, t] += 1
        W[2*k1+1, t] += 1
        col += 1
        
        # m4 = a11(b10-b00) -> contributes to C00 and C10
        t = col
        U[4, t] += 1
        V[8+k0, t] += 1
        V[k0, t] += -1
        W[2*k0, t] += 1
        W[2*k0+1, t] += 1
        col += 1
        
        # m5 = (a00+a01)b11 -> contributes to -C00 and C01
        t = col
        U[0, t] += 1
        U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1
        W[2*k1, t] += 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01) -> contributes to C11
        t = col
        U[3, t] += 1
        U[0, t] += -1
        V[k0, t] += 1
        V[k1, t] += 1
        W[2*k1+1, t] += 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11) -> contributes to C00
        t = col
        U[1, t] += 1
        U[4, t] += -1
        V[8+k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        col += 1
    
    # Part 2: A_bottom (2x1, col 2) * B_bottom (1x8, row 2)
    # A_bottom indices: i*3+2 for i in {0,1} -> U[2], U[5]
    # B_bottom indices: 2*8+k = 16+k for k in {0..7}
    # C indices: k*2+i for k in {0..7}, i in {0,1}
    
    for i in range(2):
        for k in range(8):
            t = col
            U[i*3+2, t] = 1
            V[16+k, t] = 1
            W[k*2+i, t] = 1
            col += 1
    
    assert col == 44, f"Expected 44 columns, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_schoolbook():
    """Standard schoolbook decomposition of rank n*m*p = 48."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _als_decompose(rank, max_iters=50, deadline=None, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    Uses a structured initialization based on the rank-44 solution.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Initialize with random perturbation of a known structure
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    # Add some structure: initialize from schoolbook-like pattern
    for r in range(rank):
        i = r % n
        j = (r // n) % m
        k = (r // (n*m)) % p
        U[i*m+j, r] += 1.0
        V[j*p+k, r] += 1.0
        W[k*n+i, r] += 1.0
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
                    
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
                    
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
                    
        # Check convergence every 5 iterations
        if it % 5 == 4:
            T = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        err = abs(T[a,b,c] - 1.0)
                        if err > max_err:
                            max_err = err
            target = np.zeros((n*m, m*p, p*n), dtype=bool)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = True
            non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
            
            if max_err < 1e-9 and non_target_err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W = _build_schoolbook()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # First, try the explicit rank-44 decomposition (deterministic, fast)
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            rounded = _round_and_verify(U44, V44, W44)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
            else:
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
    except Exception:
        pass
    
    # Now try to find better with ALS
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something
    # For M(2,3,8), known lower bound is around 30-35
    # Try ranks from 30 to 43
    if best_rank == 44 and time.time() < deadline - 10:
        for rank in range(30, best_rank):
            if time.time() > deadline - 5:
                break
            
            found = False
            for trial in range(5):
                if time.time() > deadline - 5:
                    break
                
                result = _als_decompose(rank, max_iters=40, deadline=deadline - 5, seed=trial*100+rank)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                        else:
                            best_U, best_V, best_W = U, V, W
                            best_rank = rank
                        found = True
                        break
            
            if found:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
