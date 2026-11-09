"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - expected))
    return max_err < tol


def _try_rank(n, m, p, rank, max_iters=300, seed=42, time_budget=20.0):
    """Try to find a rank-r decomposition using random initialization and gradient descent."""
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    start = time.time()
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        lr = 0.05
        for step in range(100):
            if time.time() - start > time_budget:
                break
            
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = T - expected
            loss = np.sum(err**2)
            
            if loss < best_err:
                best_err = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-12:
                break
            
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
    
    if best_U is not None and best_err < 1e-9:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err


def _try_block_strassen(n, m, p, time_budget=30.0):
    """
    Try a block decomposition approach.
    For 4x5x5, we can think of A as 2x2 blocks of 2x5, B as 2x2 blocks of 5x5.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Partition rows of A: top 2 rows, bottom 2 rows
    Partition cols of A / rows of B: left 5 cols, right 5 cols (can't split 5 into 2+2 evenly)
    
    Actually for 4x5x5, a natural partition:
    A is 4x5, B is 5x5, C is 4x5
    Split A into A_top (2x5) and A_bot (2x5)
    Split B into B_left (5x3) and B_right (5x2) -- not clean
    
    Alternative: just try to find a rank-r solution with r < 100 using the general method.
    """
    return None


def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p  # 100
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Known lower bound for matrix multiplication rank: max(n*m, m*p, n*p) = 25
    # But actual rank is likely higher. Let's try from 50 up to 99.
    
    # Try ranks from 50 to 99
    for rank in range(50, 100):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            U, V, W, err = _try_rank(n, m, p, rank, max_iters=50, seed=rank*100+attempt, time_budget=10.0)
            
            if U is not None and err < 1e-9:
                if _verify(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            # Found a solution at this rank, try to go lower
            break
    
    # If we found something at rank 50-99, try to go even lower
    if best_rank < 100:
        for rank in range(30, best_rank):
            if time.time() > deadline - 5:
                break
            
            found = False
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                
                U, V, W, err = _try_rank(n, m, p, rank, max_iters=30, seed=rank*200+attempt, time_budget=8.0)
                
                if U is not None and err < 1e-9:
                    if _verify(U, V, W, n, m, p):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        found = True
                        break
            
            if found:
                break
    
    # Convert to lists for the return interface
    best_U = np.asarray(best_U, dtype=float)
    best_V = np.asarray(best_V, dtype=float)
    best_W = np.asarray(best_W, dtype=float)
    
    U_list = [[float(x) for x in row] for row in best_U.tolist()]
    V_list = [[float(x) for x in row] for row in best_V.tolist()]
    W_list = [[float(x) for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
