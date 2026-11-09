import time
import numpy as np

DIMENSIONS = [2, 4, 7]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

T_START = time.time()
DEADLINE = 145.0

def _schoolbook():
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_tensor():
    """Build the matrix multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0."""
    T = np.zeros((NM, MP, PN), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _validate(U, V, W):
    r = len(U[0])
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    T = _build_tensor()
    for t in range(r):
        T -= np.einsum('a,b,c->abc', Ua[:,t], Va[:,t], Wa[:,t])
    return np.all(T == 0)

def _als_decompose(T, rank, max_iters=100, tol=1e-10):
    """Alternating least squares for tensor decomposition."""
    NM, MP, PN = T.shape
    # Initialize with random small integers
    np.random.seed(42)
    U = np.random.randint(-2, 3, size=(NM, rank))
    V = np.random.randint(-2, 3, size=(MP, rank))
    W = np.random.randint(-2, 3, size=(PN, rank))
    
    for it in range(max_iters):
        # Update U
        for a in range(NM):
            # T[a,:,:] = sum_t U[a,t] * V[:,t] * W[:,t]
            # For fixed a: T[a,:,:] = U[a,:] * (V ⊙ W) where ⊙ is element-wise
            VW = V * W  # MP x PN
            # We need: T[a,:,:] = U[a,0]*VW[:,0] + U[a,1]*VW[:,1] + ...
            # This is: T[a,:,:] = VW @ U[a,:]^T ... no
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # For fixed a: T[a,:,:] = sum_t U[a,t] * (V[:,t] * W[:,t])
            # So T[a] = VW @ U[a]^T where VW is MP x PN and U[a] is rank x 1
            # Actually: T[a] (MP x PN) = sum_t U[a,t] * outer(V[:,t], W[:,t])
            # So T[a] = VW @ diag(U[a]) ... no
            # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # Let M = V @ W^T (MP x PN)? No, that's sum_t V[b,t]W[c,t]
            # T[a] = sum_t U[a,t] * (V[:,t] W[:,t]^T)
            # So T[a] = (V * U[a]^T) @ W^T? Let's see:
            # ((V * U[a]^T) @ W^T)[b,c] = sum_t V[b,t] U[a,t] W[c,t] = T[a,b,c] ✓
            # So T[a] = (V * U[a]^T) @ W^T
            # To solve for U[a]: T[a] = (V * U[a]^T) @ W^T
            # This is: T[a] W = (V * U[a]^T) W W^T ... this is getting complicated
            # 
            # Better approach: vectorize
            # T[a] (MP*PN,) = sum_t U[a,t] * vec(V[:,t] * W[:,t])
            # Let A = [vec(V[:,0]*W[:,0]), vec(V[:,1]*W[:,1]), ...] (MP*PN x rank)
            # Then T[a] (MP*PN,) = A @ U[a]^T
            # So U[a] = (A^T A)^{-1} A^T T[a]
            A_mat = np.zeros((MP*PN, rank))
            for t in range(rank):
                A_mat[:,t] = (V[:,t] * W[:,t]).flatten()
            T_a = T[a].flatten()
            # Solve A_mat @ U[a]^T = T_a
            U[a] = np.linalg.lstsq(A_mat, T_a, rcond=None)[0]
        
        # Update V similarly
        for b in range(MP):
            A_mat = np.zeros((NM*PN, rank))
            for t in range(rank):
                A_mat[:,t] = (U[:,t] * W[:,t]).flatten()
            T_b = np.zeros((NM, PN))
            for a in range(NM):
                for c in range(PN):
                    T_b[a,c] = T[a,b,c]
            T_b = T_b.flatten()
            V[b] = np.linalg.lstsq(A_mat, T_b, rcond=None)[0]
        
        # Update W
        for c in range(PN):
            A_mat = np.zeros((NM*MP, rank))
            for t in range(rank):
                A_mat[:,t] = (U[:,t] * V[:,t]).flatten()
            T_c = np.zeros((NM, MP))
            for a in range(NM):
                for b in range(MP):
                    T_c[a,b] = T[a,b,c]
            T_c = T_c.flatten()
            W[c] = np.linalg.lstsq(A_mat, T_c, rcond=None)[0]
        
        # Check convergence
        residual = 0
        for t in range(rank):
            residual += np.abs(np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t]) - T).max()
        if residual < tol:
            break
    
    return U, V, W

def construct():
    best = _schoolbook()
    T = _build_tensor()
    
    # Try ranks from 50 down to see if we can find something
    for r in [50, 48, 46, 44, 42, 40]:
        if time.time() > DEADLINE:
            break
        U, V, W = _als_decompose(T, r, max_iters=50)
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        # Convert to list of lists with integer or half-integer values
        def to_list(mat):
            return [[float(x) if x != int(x) else int(x) for x in row] for row in mat]
        U_list = to_list(U_r)
        V_list = to_list(V_r)
        W_list = to_list(W_r)
        if _validate(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return best
