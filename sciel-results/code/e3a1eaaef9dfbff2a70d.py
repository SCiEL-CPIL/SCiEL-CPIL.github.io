import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _make_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=tol):
        return False
    return True

def _als_decompose(T, rank, n_iter=200, tol=1e-12, seed=123):
    """Alternating Least Squares for tensor decomposition with better initialization"""
    n1, n2, n3 = T.shape
    rng = np.random.RandomState(seed)
    
    # Better initialization using SVD of mode-1 unfolding
    T_mat = T.reshape(n1, n2*n3)
    try:
        U_svd, _, _ = np.linalg.svd(T_mat, full_matrices=False)
        U = U_svd[:, :rank]
    except:
        U = rng.randn(n1, rank)
    
    T_mat2 = T.transpose(1,0,2).reshape(n2, n1*n3)
    try:
        V_svd, _, _ = np.linalg.svd(T_mat2, full_matrices=False)
        V = V_svd[:, :rank]
    except:
        V = rng.randn(n2, rank)
    
    T_mat3 = T.transpose(2,0,1).reshape(n3, n1*n2)
    try:
        W_svd, _, _ = np.linalg.svd(T_mat3, full_matrices=False)
        W = W_svd[:, :rank]
    except:
        W = rng.randn(n3, rank)
    
    for it in range(n_iter):
        if it % 3 == 0:
            # Update U
            K = np.einsum('jc,kc->jkc', V, W).reshape(n2*n3, rank)
            T_mat = T.reshape(n1, n2*n3)
            Kt = K.T  # (rank, n2*n3)
            # Solve least squares: T_mat = U @ Kt
            try:
                U = np.linalg.lstsq(Kt, T_mat.T, rcond=None)[0].T
            except:
                U = T_mat @ np.linalg.pinv(Kt)
        elif it % 3 == 1:
            # Update V
            UW = np.einsum('ic,kc->ikc', U, W).reshape(n1*n3, rank)
            T_mat = T.transpose(1,0,2).reshape(n2, n1*n3)
            Ut = UW.T
            try:
                V = np.linalg.lstsq(Ut, T_mat.T, rcond=None)[0].T
            except:
                V = T_mat @ np.linalg.pinv(Ut)
        else:
            # Update W
            UV = np.einsum('ic,jc->ijc', U, V).reshape(n1*n2, rank)
            T_mat = T.transpose(2,0,1).reshape(n3, n1*n2)
            Ut = UV.T
            try:
                W = np.linalg.lstsq(Ut, T_mat.T, rcond=None)[0].T
            except:
                W = T_mat @ np.linalg.pinv(Ut)
        
        # Check convergence
        if it > 2:
            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < tol:
                break
    
    return U, V, W

def _hals_decompose(T, rank, n_iter=100, tol=1e-12, seed=123):
    """Hierarchical Alternating Least Squares with deflation"""
    n1, n2, n3 = T.shape
    rng = np.random.RandomState(seed)
    
    T_remaining = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for c in range(rank):
        # Random initialization for this component
        U = rng.randn(n1)
        V = rng.randn(n2)
        W = rng.randn(n3)
        
        for it in range(n_iter):
            # Update U
            VW = np.outer(V, W)
            U = np.einsum('ijk,jk->i', T_remaining, VW)
            norm_U = np.linalg.norm(U)
            if norm_U > 1e-15:
                U /= norm_U
            
            # Update V
            UW = np.outer(U, W)
            V = np.einsum('ijk,ik->j', T_remaining, UW)
            norm_V = np.linalg.norm(V)
            if norm_V > 1e-15:
                V /= norm_V
            
            # Update W
            UV = np.outer(U, V)
            W = np.einsum('ijk,ij->k', T_remaining, UV)
            norm_W = np.linalg.norm(W)
            if norm_W > 1e-15:
                W /= norm_W
        
        # Check if this component is meaningful
        U_norm = np.linalg.norm(U)
        V_norm = np.linalg.norm(V)
        W_norm = np.linalg.norm(W)
        if U_norm < 1e-10 or V_norm < 1e-10 or W_norm < 1e-10:
            break
        
        # Deflate
        T_remaining -= np.einsum('i,j,k->ijk', U, V, W)
        
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        
        # Check if residual is small
        if np.linalg.norm(T_remaining) < tol:
            break
    
    if len(U_list) == 0:
        return None, None, None
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return U, V, W

def construct():
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    
    # Try ranks from 40 to 55
    best_result = None
    best_rank = float('inf')
    
    for target_rank in range(40, 56):
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Try ALS with multiple seeds
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            U, V, W = _als_decompose(T, target_rank, n_iter=150, tol=1e-11, seed=seed)
            if _verify_decomposition(U, V, W, T):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
        
        # Try HALS if ALS failed
        if target_rank >= 45:
            for seed in range(5):
                if time.time() > deadline - 20:
                    break
                U, V, W = _hals_decompose(T, target_rank, n_iter=50, tol=1e-11, seed=seed)
                if U is not None and _verify_decomposition(U, V, W, T):
                    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result:
        return best_result
    
    # Fallback to rank 51 (Strassen-based)
    rank = 51
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # j=4 boundary
    U[:, 0] = 1.0
    V[20:, 0] = 1.0
    W[:, 0] = 1.0
    
    # k=4 boundary
    U[0:20, 1] = 1.0
    V[4:25:5, 1] = 1.0
    W[16:20, 1] = 1.0
    
    # 4x4x4 Strassen
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    U_s[0,0]=1; U_s[3,0]=1; V_s[0,0]=1; V_s[3,0]=1; W_s[0,0]=1; W_s[1,0]=1; W_s[2,0]=1; W_s[3,0]=1
    U_s[2,1]=1; U_s[3,1]=1; V_s[0,1]=1; W_s[2,1]=1
    U_s[0,2]=1; V_s[1,2]=1; V_s[3,2]=-1; W_s[1,2]=1
    U_s[3,3]=1; V_s[2,3]=1; V_s[0,3]=-1; W_s[2,3]=1
    U_s[0,4]=1; U_s[1,4]=1; V_s[3,4]=1; W_s[1,4]=1
    U_s[2,5]=1; U_s[0,5]=-1; V_s[0,5]=1; V_s[1,5]=1; W_s[2,5]=1; W_s[3,5]=1
    U_s[1,6]=1; U_s[3,6]=-1; V_s[2,6]=1; V_s[3,6]=1; W_s[1,6]=1; W_s[3,6]=1
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1*7+c2
            col = 2+c
            if col >= rank: break
            for i1 in range(2):
                for i2 in range(2):
                    i = i1*2+i2
                    U[i, col] = U_s[i1,c1]*U_s[i2,c2]
            for j1 in range(2):
                for j2 in range(2):
                    j = j1*2+j2
                    V[j, col] = V_s[j1,c1]*V_s[j2,c2]
            for k1 in range(2):
                for k2 in range(2):
                    k = k1*2+k2
                    W[k, col] = W_s[k1,c1]*W_s[k2,c2]
    
    if _verify_decomposition(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Schoolbook
    rank_sb = n*m*p
    U_fb = np.zeros((n*m, rank_sb))
    V_fb = np.zeros((m*p, rank_sb))
    W_fb = np.zeros((p*n, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, c] = 1
                V_fb[j*p+k, c] = 1
                W_fb[k*n+i, c] = 1
                c += 1
    return {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}
