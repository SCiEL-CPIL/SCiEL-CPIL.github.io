import numpy as np
import time

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS  # 3, 4, 5


def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2. Returns U,V,W each (4,7).
    U order: A00,A01,A10,A11 ; V order: B00,B01,B10,B11 ; W order: C00,C01,C10,C11."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _build_122():
    """Rank-4 schoolbook for 1x2x2: A(1x2)*B(2x2)=C(1x2). U(2,4),V(4,4),W(2,4)."""
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    U[0,2]=1; V[1,2]=1; W[1,2]=1
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    return U, V, W


def _build_121():
    """Rank-2 for 1x2x1: A(1x2)*B(2x1)=C(1x1). U(2,2),V(2,2),W(1,2)."""
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W


def _build_221():
    """Rank-4 schoolbook for 2x2x1: A(2x2)*B(2x1)=C(2x1). U(4,4),V(2,4),W(2,4)."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W


def _verify(U, V, W):
    r = U.shape[1]
    T = np.zeros((n, m, p))
    Umat = np.asarray(U, dtype=float)
    Vmat = np.asarray(V, dtype=float)
    Wmat = np.asarray(W, dtype=float)
    for idx in range(r):
        u = Umat[:, idx].reshape(n, m)
        v = Vmat[:, idx].reshape(m, p)
        w = Wmat[:, idx].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    M = np.ones((n, m, p))
    return np.allclose(T, M, atol=1e-9)


def _construct_rank_56():
    """
    Block decomposition achieving rank 56.
    A rows: [0,1] (2 rows) and [2] (1 row).
    B cols: [0,1] (2), [2,3] (2), [4] (1).
    For A[0:2,:]: 2x2x2 Strassen (rank 7) + 2x2x1 schoolbook (rank 4) per 2-col block,
                  2x2x2 + 2x2x1 for second 2-col block.
                  Total: 7+7+4 + 7+7+4 = 36
    For A[2,:]: 1x2x2 schoolbook (rank 4) + 1x2x1 (rank 2) per 2-col block.
                  Total: 4+4+2 + 4+4+2 = 20
    Grand total: 56
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, Ublk, Vblk, Wblk):
        for r in range(Ublk.shape[1]):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            for i, a in enumerate(u_idx):
                u[a] = Ublk[i, r]
            for i, a in enumerate(v_idx):
                v[a] = Vblk[i, r]
            for i, a in enumerate(w_idx):
                w[a] = Wblk[i, r]
            U_parts.append(u)
            V_parts.append(v)
            W_parts.append(w)
    
    # U indices: A[i,j] -> i*m+j = i*4+j
    # V indices: B[j,k] -> j*p+k = j*5+k
    # W indices: C[i,k] -> k*n+i = k*3+i
    
    # Part A: A[0:2,:] (rows 0,1) * B
    # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    Uu, Vv, Ww = _build_strassen_222()
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], Uu, Vv, Ww)
    
    # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    Uu, Vv, Ww = _build_strassen_222()
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], Uu, Vv, Ww)
    
    # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    Uu, Vv, Ww = _build_221()
    add_block([0,1,4,5], [4,9], [12,13], Uu, Vv, Ww)
    
    # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    Uu, Vv, Ww = _build_strassen_222()
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], Uu, Vv, Ww)
    
    # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    Uu, Vv, Ww = _build_strassen_222()
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], Uu, Vv, Ww)
    
    # A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    Uu, Vv, Ww = _build_221()
    add_block([2,3,6,7], [14,19], [12,13], Uu, Vv, Ww)
    
    # Part B: A[2,:] (row 2) * B
    # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    Uu, Vv, Ww = _build_122()
    add_block([8,9], [0,1,5,6], [2,5], Uu, Vv, Ww)
    
    # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    Uu, Vv, Ww = _build_122()
    add_block([8,9], [2,3,7,8], [8,11], Uu, Vv, Ww)
    
    # A[2,0:2] * B[0:2,4:5] -> C[2,4:5]
    Uu, Vv, Ww = _build_121()
    add_block([8,9], [4,9], [14], Uu, Vv, Ww)
    
    # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    Uu, Vv, Ww = _build_122()
    add_block([10,11], [10,11,15,16], [2,5], Uu, Vv, Ww)
    
    # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    Uu, Vv, Ww = _build_122()
    add_block([10,11], [12,13,17,18], [8,11], Uu, Vv, Ww)
    
    # A[2,2:4] * B[2:4,4:5] -> C[2,4:5]
    Uu, Vv, Ww = _build_121()
    add_block([10,11], [14,19], [14], Uu, Vv, Ww)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def _construct_schoolbook():
    U_parts, V_parts, W_parts = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def _try_compress(U, V, W, t0, deadline=150.0, margin=30.0):
    """
    Try to compress the decomposition by finding a lower-rank equivalent.
    We attempt to find linear dependencies among the rank-1 tensors.
    If the Khatri-Rao product matrix has rank < r, we can compress.
    """
    if time.time() - t0 > deadline - margin:
        return None
    
    r = U.shape[1]
    Umat = np.asarray(U, dtype=float)
    Vmat = np.asarray(V, dtype=float)
    Wmat = np.asarray(W, dtype=float)
    
    # Build the Khatri-Rao product matrix M of size (n*m*p) x r
    # M[:,t] = vec(U[:,t]) ⊙ vec(V[:,t]) ⊙ vec(W[:,t])  (Hadamard product)
    # Actually for tensor decomposition, the condition is:
    # For each t: U_t ⊗ V_t ⊗ W_t, and we need sum_t U_t ⊗ V_t ⊗ W_t = T
    # The compression works if the vectors {U_t ⊙ V_t ⊙ W_t} (flattened) are linearly dependent
    # in a way that preserves the tensor.
    
    # Better approach: try to find a matrix Q (r x r') with r' < r such that
    # U' = U @ Q, V' = V @ Q^{-1} (if Q is invertible), W' = W @ I
    # This preserves the tensor if Q is a valid change of basis.
    
    # Simplest compression: check if columns of the "flattened" matrix are dependent.
    # The tensor T = sum_t u_t ⊗ v_t ⊗ w_t
    # Flatten to matrix: for each (i,j), T[i,j,:] = sum_t u_t[i,j] * v_t[j,:] * w_t[:,i]
    
    # Alternative: use the fact that if we can find a matrix A (r x s), B (r x s), C (r x s)
    # with s < r such that U = U' @ A^{-1}, V = V' @ B^{-1}, W = W' @ C^{-1} ... this is complex.
    
    # Let's try a simpler approach: random linear combinations and check rank reduction.
    # This is heuristic and may not work, but let's try.
    
    # Actually, the most reliable compression for structured decompositions is to check
    # if the decomposition has redundant terms. For our block decomposition, each block
    # is independent, so compression is unlikely unless there's overlap.
    
    # Let's try: compute the effective rank by checking if any subset of terms can be removed.
    # This is expensive, so we'll skip for now and return None.
    
    return None


def construct():
    t0 = time.time()
    deadline = 150.0
    margin = 30.0
    
    try:
        U, V, W = _construct_rank_56()
        if time.time() - t0 > deadline - margin:
            U, V, W = _construct_schoolbook()
        elif not _verify(U, V, W):
            U, V, W = _construct_schoolbook()
        else:
            # Try to compress
            compressed = _try_compress(U, V, W, t0, deadline, margin)
            if compressed is not None:
                U, V, W = compressed
    except Exception:
        U, V, W = _construct_schoolbook()
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
