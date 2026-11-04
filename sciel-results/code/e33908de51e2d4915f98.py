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

def _als_decompose(T, rank, n_iter=100, tol=1e-10, seed=123):
    """Alternating Least Squares for tensor decomposition"""
    n1, n2, n3 = T.shape
    np.random.seed(seed)
    
    U = np.random.randn(n1, rank)
    V = np.random.randn(n2, rank)
    W = np.random.randn(n3, rank)
    
    for it in range(n_iter):
        if it % 3 == 0:
            # Update U
            K = np.einsum('jc,kc->jkc', V, W).reshape(n2*n3, rank)
            T_mat = T.reshape(n1, n2*n3)
            Kt = K.T  # (rank, n2*n3)
            U = T_mat @ np.linalg.pinv(Kt)
        elif it % 3 == 1:
            # Update V
            UW = np.einsum('ic,kc->ikc', U, W).reshape(n1*n3, rank)
            T_mat = T.transpose(1,0,2).reshape(n2, n1*n3)
            Ut = UW.T
            V = T_mat @ np.linalg.pinv(Ut)
        else:
            # Update W
            UV = np.einsum('ic,jc->ijc', U, V).reshape(n1*n2, rank)
            T_mat = T.transpose(2,0,1).reshape(n3, n1*n2)
            Ut = UV.T
            W = T_mat @ np.linalg.pinv(Ut)
        
        # Check convergence
        if it > 0:
            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < tol:
                break
    
    return U, V, W

def construct():
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    
    # Try ranks from 45 to 51
    best_result = None
    
    for target_rank in range(45, 52):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random seeds for each rank
        found = False
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            U, V, W = _als_decompose(T, target_rank, n_iter=150, tol=1e-10, seed=seed)
            if _verify_decomposition(U, V, W, T):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
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
