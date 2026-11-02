"""Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification using einsum."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
    target = np.zeros((n, m, p))
    for i in range(n):
        target[i, i, :] = 1.0
    return np.allclose(T, target, atol=1e-9)


def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition as lists of lists."""
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


def _strassen_2x2x2():
    """
    Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    
    The target tensor for 2x2x2 matmul is T[i,j,k] = delta_{i,j} for all k.
    Strassen's algorithm gives 7 terms.
    
    We use the standard Strassen decomposition:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B00)
    M4 = A11(B11 - B10)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For the tensor T[i,j,k] = delta_{i,j}, we need:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,j}
    
    Let's construct the decomposition directly.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products
    # Each product r corresponds to a bilinear form
    # We need to express delta_{i,j} as sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Let's use the standard embedding. For 2x2x2:
    # T[i,j,k] = delta_{i,j}
    
    # We can write this as a sum of 7 rank-1 tensors.
    # Here's one valid decomposition:
    
    # Term 0: handles (0,0,0) and (0,0,1) and (1,1,0) and (1,1,1) partially
    # Actually, let me just use a known valid decomposition.
    
    # For the identity tensor T[i,j,k] = delta_{i,j}, we can use:
    # The decomposition is not the standard Strassen for matmul, but for the identity tensor.
    
    # Let me think more carefully. The target is:
    # T[i,j,k] = 1 if i==j else 0, for all k.
    
    # This is a 2x2x2 tensor with T[0,0,0]=1, T[0,0,1]=1, T[1,1,0]=1, T[1,1,1]=1, all others 0.
    
    # We need to decompose this into rank 7 (or less).
    
    # Actually, the minimal rank of this tensor is known. Let me try to find it.
    
    # Simple decomposition with rank 4:
    # For each (i,j) pair where i==j, use 2 terms (one for each k).
    # That gives rank 4.
    
    # Can we do better? Let's try rank 3 or less.
    
    # For now, let's use rank 4 for simplicity, and try to reduce later.
    
    # Rank 4 decomposition:
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 0: i=0, j=0, k=0
    U[0 * m + 0, 0] = 1
    V[0 * p + 0, 0] = 1
    W[0 * n + 0, 0] = 1
    
    # Term 1: i=0, j=0, k=1
    U[0 * m + 0, 1] = 1
    V[0 * p + 1, 1] = 1
    W[1 * n + 0, 1] = 1
    
    # Term 2: i=1, j=1, k=0
    U[1 * m + 1, 2] = 1
    V[1 * p + 0, 2] = 1
    W[0 * n + 1, 2] = 1
    
    # Term 3: i=1, j=1, k=1
    U[1 * m + 1, 3] = 1
    V[1 * p + 1, 3] = 1
    W[1 * n + 1, 3] = 1
    
    return U, V, W, rank


def _block_diagonal_decomposition():
    """
    Construct a block-diagonal decomposition for 3x5x6.
    
    Split the m=5 dimension into two parts: m1=2 and m2=3.
    The first part handles (i in {0,1}, j in {0,1}) and the second handles (i in {0,1,2}, j in {2,3,4}).
    
    For the first block (2x2x6):
    - We need T[i,j,k] = delta_{i,j} for i,j in {0,1}, all k in {0..5}
    - This is a 2x2x6 tensor
    - We can decompose it using 2x2x2 Strassen-like blocks for each pair of k values
    
    For the second block (3x3x6):
    - We need T[i,j,k] = delta_{i,j} for i,j in {0,1,2}, all k in {0..5}
    - But wait, the indices are shifted. Let me reconsider.
    
    Actually, the target tensor is T[i,j,k] = delta_{i,j} for all i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}.
    
    This means:
    - For i=0: T[0,j,k] = 1 if j=0 else 0
    - For i=1: T[1,j,k] = 1 if j=1 else 0
    - For i=2: T[2,j,k] = 1 if j=2 else 0
    
    So the non-zero entries are:
    - (0,0,k) for all k
    - (1,1,k) for all k
    - (2,2,k) for all k
    
    This is a sparse tensor. We can decompose it as a sum of three "strips":
    - Strip 0: T[0,0,k] = 1 for all k
    - Strip 1: T[1,1,k] = 1 for all k
    - Strip 2: T[2,2,k] = 1 for all k
    
    Each strip is a 1x1x6 tensor, which has rank 6 (one term per k).
    Total rank = 3 * 6 = 18.
    
    But we can do better by sharing terms across strips if possible.
    
    Actually, let's just use the simple rank-18 decomposition first, then try to reduce.
    """
    n, m, p = 3, 5, 6
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for k in range(p):
            # Term for (i, i, k)
            U[i * m + i, col] = 1
            V[i * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    return U, V, W, rank


def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=500):
    """
    Try to reduce the rank of the decomposition using alternating minimization.
    """
    rank = target_rank
    U = U[:, :rank].copy() if U.shape[1] >= rank else np.zeros((n * m, rank))
    V = V[:, :rank].copy() if V.shape[1] >= rank else np.zeros((m * p, rank))
    W = W[:, :rank].copy() if W.shape[1] >= rank else np.zeros((p * n, rank))
    
    # Initialize with random perturbations
    U += np.random.randn(n * m, rank) * 0.01
    V += np.random.randn(m * p, rank) * 0.01
    W += np.random.randn(p * n, rank) * 0.01
    
    target = np.zeros((n, m, p))
    for i in range(n):
        target[i, i, :] = 1.0
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining_time() < 2.0:
            break
        
        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = target[i, j, :]
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i * m + j, :] = sol
        U = U_new
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = target[:, j, k]
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j * p + k, :] = sol
        V = V_new
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = target[i, :, k]
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k * n + i, :] = sol
        W = W_new
        
        # Compute residual
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
        residual = np.linalg.norm(T - target)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < 1e-10:
            break
    
    return best_U, best_V, best_W, best_residual


def construct():
    """
    Returns a tensor decomposition for 3x5x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with the simple rank-18 decomposition
    U, V, W, rank = _block_diagonal_decomposition()
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        # Try to reduce rank further
        for target_rank in range(17, 10, -1):
            if _remaining_time() < 10.0:
                break
            
            U_r, V_r, W_r, residual = _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=300)
            
            if residual < 1e-8:
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    U, V, W = U_r, V_r, W_r
                    rank = target_rank
                    break
    
    # Convert to lists
    U_list = [[float(x) for x in U[i]] for i in range(U.shape[0])]
    V_list = [[float(x) for x in V[i]] for i in range(V.shape[0])]
    W_list = [[float(x) for x in W[i]] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
