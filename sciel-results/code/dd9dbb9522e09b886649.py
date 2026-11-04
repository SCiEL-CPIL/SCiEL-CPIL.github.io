"""Matrix multiplication 3x5x5 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p, atol=1e-10):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB_direct = A @ B

    U_reshaped = U.T.reshape(rank, n, m)
    V_reshaped = V.T.reshape(rank, m, p)
    W_reshaped_temp = W.T.reshape(rank, p, n)
    W_reshaped = W_reshaped_temp.transpose(0, 2, 1)

    U_A = np.einsum('rij,ij->r', U_reshaped, A)
    V_B = np.einsum('rjk,jk->r', V_reshaped, B)
    AB_decomp = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)

    return np.allclose(AB_direct, AB_decomp, atol=atol)


def _tensor_product_decomposition(n, m, p):
    """
    Use tensor product: M_{n,m,p} = M_{n,1,1} ⊗ M_{1,m,p}
    
    For n=3, m=5, p=5:
    - M_{3,1,1} has rank 3 (each row of output needs one term)
    - M_{1,5,5} has rank 5 (row vector times 5x5 matrix, each output component is a dot product)
    - Total rank = 3 * 5 = 15
    
    Construction:
    M_{n,1,1}: U[i, r] = 1 if r==i, V[0, r] = 1 if r==i, W[0, r] = 1 if r==i
    M_{1,m,p}: U[0, s] = 1 if s==j, V[s, s] = 1, W[s, s] = 1 for s in 0..m-1
    
    Combined for (r, s) in 0..n-1 × 0..m-1:
    U[i*m + s, r*m + s] = 1
    V[s*p + s, r*m + s] = 1  (since p=m=5)
    W[s*n + i, r*m + s] = 1
    """
    rank = n * m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(n):
        for s in range(m):
            col = r * m + s
            U[r * m + s, col] = 1
            V[s * p + s, col] = 1
            W[s * n + r, col] = 1
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 75)
    U, V, W = _schoolbook(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try tensor product decomposition (rank 15)
    if time.time() < deadline:
        U_tp, V_tp, W_tp = _tensor_product_decomposition(n, m, p)
        if _verify(U_tp, V_tp, W_tp, n, m, p):
            rank_tp = U_tp.shape[1]
            if rank_tp < best_rank:
                best_rank = rank_tp
                best_U, best_V, best_W = U_tp, V_tp, W_tp
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
