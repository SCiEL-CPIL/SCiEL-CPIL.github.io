# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                T[i, j, k] = val
    return np.allclose(T, 1.0, atol=1e-6)

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_block_decomposition(n, m, p, block_size=2):
    """
    Construct a decomposition using Strassen's algorithm on 2x2 blocks.
    For 4x5 * 5x5, we can decompose into blocks.
    4x5 = [2x2 2x3; 2x2 2x3] but let's use 2x2 blocks where possible.
    Actually, 4x5 * 5x5:
    Let A be 4x5, B be 5x5.
    Split A into 2x2 blocks of size 2x2 and 2x3? No, let's do:
    A = [[A11, A12], [A21, A22]] where A11, A21 are 2x2, A12, A22 are 2x3
    B = [[B11, B12], [B21, B22]] where B11, B21 are 2x2, B12, B22 are 3x2? No.
    B is 5x5. Split into 2x2 and 3x3?
    Let's use a 2x2 block structure for the 4x4 part and handle the 5th column/row separately.
    This is getting complicated. Let's try a simpler approach:
    Use the known rank for 4x4x4 which is 12 (Strassen).
    For 4x5x5, we can use a Kronecker-like product or block structure.
    
    Alternative: Use the fact that M_{4,5,5} can be decomposed using M_{2,2,2} (rank 7) blocks.
    4x5 = 2x2 blocks of 2x2 and 2x3?
    Let's try to construct it manually or use a known result.
    The rank of M_{4,5,5} is at most 4*5*5 = 100.
    Using Strassen on 2x2 blocks:
    If we partition A (4x5) into four 2x2 blocks and one 2x1 column? No.
    Let's partition A into [A1 A2] where A1 is 4x2 and A2 is 4x3.
    B into [B1 B2] where B1 is 2x5 and B2 is 3x5? No, B is 5x5.
    B = [B1; B2] where B1 is 2x5 and B2 is 3x5.
    Then C = A1*B1 + A2*B2.
    A1 is 4x2, B1 is 2x5 -> 4x5 product.
    A2 is 4x3, B2 is 3x5 -> 4x5 product.
    This doesn't reduce rank directly.
    
    Let's use the standard Strassen decomposition for 2x2 matrices.
    M_{2,2,2} has rank 7.
    M_{4,4,4} can be done with rank 12 (Strassen on 2x2 blocks of 2x2).
    M_{4,5,5}: 
    Let's try to use a 2x2 block structure on the 4x4 sub-matrix and add the 5th dimension.
    This is complex. Let's stick to the ALS approach but improve it.
    """
    pass

def _als_decomposition(n, m, p, rank, iters=50, seed=0):
    """Attempt to find a decomposition of given rank using ALS."""
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    for it in range(iters):
        # Update U
        for i in range(n):
            for j in range(m):
                V_j = V.reshape(m, p, rank)[j, :, :]
                W_i = W.reshape(p, n, rank)[:, i, :]
                M = (V_j * W_i).T
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                U_j = U.reshape(n, m, rank)[:, j, :]
                W_k = W.reshape(p, n, rank)[k, :, :]
                B = (U_j * W_k).T
                try:
                    V[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                U_i = U.reshape(n, m, rank)[i, :, :]
                V_k = V.reshape(m, p, rank)[:, k, :]
                C = (U_i * V_k).T
                try:
                    W[k * n + i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                except:
                    pass

    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _construct_strassen_4x4x4():
    """
    Construct Strassen decomposition for 4x4x4 matrix multiplication.
    Rank is 12.
    """
    n, m, p = 4, 4, 4
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2 blocks.
    # For 4x4, we have 4 blocks of 2x2.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # This is complex to encode directly. Let's use a known decomposition or skip this.
    # For now, return None to indicate not implemented.
    return None

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 30 to best_rank - 1
    for rank in range(30, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random seeds
        for seed in range(10):
            if time.time() > deadline:
                break
                
            result = _als_decomposition(n, m, p, rank, iters=20, seed=seed)
            if result is not None:
                # Attempt to round to integer/half-integer
                try:
                    U_rounded = [[round(2*x)/2 for x in row] for row in result['U']]
                    V_rounded = [[round(2*x)/2 for x in row] for row in result['V']]
                    W_rounded = [[round(2*x)/2 for x in row] for row in result['W']]
                    
                    # Verify the rounded version
                    if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                        if rank < best_rank:
                            best = {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
                            best_rank = rank
                            if best_rank <= 35:
                                return best
                except:
                    pass
                    
    return best
# EVOLVE-BLOCK-END
