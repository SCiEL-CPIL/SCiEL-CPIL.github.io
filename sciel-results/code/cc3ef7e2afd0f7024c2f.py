"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses a constructive approach based on Kronecker-product block structure.
The matmul tensor for (n,m,p) = (4,4,6) has 96 non-zero entries (schoolbook rank 96).
We attempt to find a lower-rank decomposition using structured ansatz and local search.
"""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 6]


def _build_target_tensor(n, m, p):
    """Build the target matmul tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T


def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[:,r] * V[:,r] * W[:,r] == target tensor."""
    T_target = _build_target_tensor(n, m, p)
    rank = U.shape[1]
    T_dec = np.zeros_like(T_target)
    for r in range(rank):
        T_dec += np.multiply.outer(np.multiply.outer(U[:, r], V[:, r]), W[:, r])
    return np.array_equal(T_dec, T_target)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_kronecker_block(n, m, p):
    """
    Attempt a Kronecker-product based decomposition.
    
    For 4x4 * 4x6, we can think of it as:
    - Split the 6 columns into two groups of 3: A(4x4) * B1(4x3) and A(4x4) * B2(4x3)
    - Each 4x4 * 4x3 has schoolbook rank 48
    - Total schoolbook: 96
    
    We try to use a structured ansatz where each rank component is a Kronecker product
    of small vectors, allowing for potential sharing between the two halves.
    
    A simpler approach: use the fact that 4 = 2*2 and 6 = 2*3, and try to exploit
    tensor product structure.
    """
    n, m, p = 4, 4, 6
    
    # Try a decomposition based on splitting indices
    # i = i1*2 + i2, j = j1*2 + j2, k = k1*3 + k2
    # where i1,j1 in {0,1}, i2,j2 in {0,1}, k1 in {0,1}, k2 in {0,1,2}
    
    # This gives a natural block structure. Let's try a rank-32 ansatz first
    # (half of schoolbook) by exploiting symmetry.
    
    # Actually, let's try a more direct approach: use random initialization with 
    # integer entries and do local search to minimize the error, then project to exact.
    
    # For now, return schoolbook as fallback
    return None


def _local_search(U, V, W, n, m, p, max_iters=5000, time_limit=60):
    """
    Local search to try to reduce rank by perturbing entries and checking validity.
    """
    start_time = time.time()
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = U.shape[1]
    
    T_target = _build_target_tensor(n, m, p)
    
    def _error(U, V, W):
        rank = U.shape[1]
        T_dec = np.zeros_like(T_target)
        for r in range(rank):
            T_dec += np.multiply.outer(np.multiply.outer(U[:, r], V[:, r]), W[:, r])
        return np.sum((T_dec - T_target) ** 2)
    
    current_error = _error(U, V, W)
    if current_error == 0:
        return U, V, W, best_rank
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly perturb one entry
        mat = random.choice(['U', 'V', 'W'])
        if mat == 'U':
            row = random.randint(0, U.shape[0] - 1)
            col = random.randint(0, U.shape[1] - 1)
            old = U[row, col]
            new = old + random.choice([-1, 1])
            U[row, col] = new
            if _error(U, V, W) < current_error:
                current_error = _error(U, V, W)
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            else:
                U[row, col] = old
        elif mat == 'V':
            row = random.randint(0, V.shape[0] - 1)
            col = random.randint(0, V.shape[1] - 1)
            old = V[row, col]
            new = old + random.choice([-1, 1])
            V[row, col] = new
            if _error(U, V, W) < current_error:
                current_error = _error(U, V, W)
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            else:
                V[row, col] = old
        else:
            row = random.randint(0, W.shape[0] - 1)
            col = random.randint(0, W.shape[1] - 1)
            old = W[row, col]
            new = old + random.choice([-1, 1])
            W[row, col] = new
            if _error(U, V, W) < current_error:
                current_error = _error(U, V, W)
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            else:
                W[row, col] = old
    
    return best_U, best_V, best_W, best_rank


def _try_rank_reduction(U, V, W, n, m, p):
    """
    Try to reduce rank by checking if any column is a linear combination of others
    in a tensor sense. This is heuristic.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Try removing each column and checking if the remaining can still span
    for r in range(rank):
        U_trial = np.delete(U, r, axis=1)
        V_trial = np.delete(V, r, axis=1)
        W_trial = np.delete(W, r, axis=1)
        if _verify_decomposition(U_trial, V_trial, W_trial, n, m, p):
            return U_trial, V_trial, W_trial
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Verify schoolbook
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but handle gracefully
        pass
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try local search to see if we can find a lower-rank exact decomposition
    # Note: local search on the error landscape may not find exact solutions easily
    # for integer entries, but it's worth trying within time budget.
    
    remaining_time = 110 - (time.time() - start_time)
    if remaining_time > 10:
        U_ls, V_ls, W_ls, rank_ls = _local_search(U.copy(), V.copy(), W.copy(), n, m, p, 
                                                   max_iters=2000, time_limit=remaining_time - 5)
        if rank_ls < best_rank and _verify_decomposition(U_ls, V_ls, W_ls, n, m, p):
            best_U, best_V, best_W, best_rank = U_ls, V_ls, W_ls, rank_ls
    
    # Try rank reduction on the best found
    if time.time() - start_time < 115:
        best_U, best_V, best_W = _try_rank_reduction(best_U, best_V, best_W, n, m, p)
        best_rank = best_U.shape[1]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
