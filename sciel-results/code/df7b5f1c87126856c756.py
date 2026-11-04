import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to be safe, leaving 10s margin
    
    # Build the target tensor T
    # T[a, b, c] = 1 if exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, r
    
    # Initialize best with schoolbook
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Function to verify decomposition
    def verify(U, V, W):
        # Compute product tensor: P[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        # Using einsum for efficiency
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        err = np.max(np.abs(P - T))
        return err

    # ALS for CP Decomposition
    def als_decompose(T, rank, max_iter=100, tol=1e-10):
        """
        Alternating Least Squares for CP decomposition.
        T is a 3D tensor.
        Returns U, V, W and error.
        """
        I, J, K = T.shape
        U = np.random.randn(I, rank) * 0.1
        V = np.random.randn(J, rank) * 0.1
        W = np.random.randn(K, rank) * 0.1
        
        # Unfold T along mode 1, 2, 3
        T1 = T.reshape(I, J*K)
        T2 = T.reshape(J, I*K)
        T3 = T.reshape(K, I*J)
        
        for it in range(max_iter):
            # Update U
            # T1 ~= U * (V ⊗ W)^T
            # Let M1 = V ⊗ W (J*K x rank)
            M1 = np.einsum('jk,ik->ji', V, W) # This is wrong. V is J x rank, W is K x rank.
            # We need M1 such that M1[c, l] = V[:, c] ⊗ W[:, c] ? No.
            # The term is sum_c U[:, c] ⊗ V[:, c] ⊗ W[:, c].
            # Unfolded along 1: T1[i, j, k] = sum_c U[i, c] V[j, c] W[k, c].
            # T1 is I x (J*K).
            # T1 ~= U * (V ⊗ W)^T is not quite right because the columns of (V ⊗ W) are not independent in the same way.
            # Actually, if we define M1 as a matrix of size (J*K) x rank where column c is V[:, c] ⊗ W[:, c], then T1 ~= U M1^T.
            M1 = np.einsum('jc,kc->jkc', V, W).reshape(J*K, rank)
            U, residuals, rank_u, s = np.linalg.lstsq(M1.T, T1.T, rcond=None)
            
            # Update V
            # T2 is J x (I*K).
            # T2[j, i, k] = sum_c U[i, c] V[j, c] W[k, c].
            # M2 column c is U[:, c] ⊗ W[:, c].
            M2 = np.einsum('ic,kc->ikc', U, W).reshape(I*K, rank)
            V, residuals, rank_v, s = np.linalg.lstsq(M2.T, T2.T, rcond=None)
            
            # Update W
            # T3 is K x (I*J).
            # M3 column c is U[:, c] ⊗ V[:, c].
            M3 = np.einsum('ic,jc->ijc', U, V).reshape(I*J, rank)
            W, residuals, rank_w, s = np.linalg.lstsq(M3.T, T3.T, rcond=None)
            
            # Check convergence
            P = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.max(np.abs(P - T))
            if err < tol:
                break
            if time.time() > deadline - 10:
                break
                
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        err = np.max(np.abs(P - T))
        return U, V, W, err

    # Try to find a better decomposition
    # We try ranks lower than schoolbook (45)
    target_ranks = [30, 35, 40]
    
    for rank in target_ranks:
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue
            
        # Try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            U, V, W, err = als_decompose(T, rank, max_iter=50, tol=1e-12)
            if err < 1e-9:
                # Verify strictly
                if verify(U, V, W) < 1e-9:
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    found = True
                    break
        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
