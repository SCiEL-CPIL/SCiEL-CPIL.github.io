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

def _als_decompose(T, rank, n_iter=200, tol=1e-10, seed=123):
    """Alternating Least Squares for tensor decomposition"""
    n1, n2, n3 = T.shape
    np.random.seed(seed)
    
    U = np.random.randn(n1, rank)
    V = np.random.randn(n2, rank)
    W = np.random.randn(n3, rank)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(n_iter):
        if it % 3 == 0:
            K = np.einsum('jc,kc->jkc', V, W).reshape(n2*n3, rank)
            T_mat = T.reshape(n1, n2*n3)
            Kt = K.T
            U = T_mat @ np.linalg.pinv(Kt)
        elif it % 3 == 1:
            UW = np.einsum('ic,kc->ikc', U, W).reshape(n1*n3, rank)
            T_mat = T.transpose(1,0,2).reshape(n2, n1*n3)
            Ut = UW.T
            V = T_mat @ np.linalg.pinv(Ut)
        else:
            UV = np.einsum('ic,jc->ijc', U, V).reshape(n1*n2, rank)
            T_mat = T.transpose(2,0,1).reshape(n3, n1*n2)
            Ut = UV.T
            W = T_mat @ np.linalg.pinv(Ut)
        
        if it > 2:
            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                break
    
    return best_U, best_V, best_W

def _structured_decomposition(n, m, p, rank_55, U55, V55, W55):
    """
    Build a decomposition for n x m x p where m=p=5 by using:
    - rank_55 for the 5x5x5 block (i=1..4, j=0..4, k=0..4)
    - schoolbook for the i=0 row interactions
    Total rank = rank_55 + n*m*p - (n-1)*m*p = rank_55 + m*p
    For n=4, m=5, p=5: rank = rank_55 + 25
    """
    total_rank = rank_55 + m*p
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: Use the 5x5x5 decomposition for i=1..4 (indices 1..n-1)
    # The 5x5x5 tensor has indices (i', j, k) where i'=0..3 corresponds to i=1..4
    # U55 has shape (m, rank_55) for the i' dimension? No, let's think carefully.
    # 
    # Actually, for the 5x5x5 multiplication tensor, the dimensions are:
    # - i' dimension: 5 (rows of first matrix)
    # - j dimension: 5 (cols of first matrix = rows of second)
    # - k dimension: 5 (cols of second matrix)
    # 
    # The tensor T55[i', j, k] = 1 if the (i',j) entry of A times (j,k) entry of B
    # contributes to (i',k) entry of C.
    # 
    # In our 4x5x5 tensor, for i=1..4 (i'=i-1=0..3), we have the same structure.
    # 
    # U55 should have shape (5, rank_55) for the i' dimension
    # V55 should have shape (5, rank_55) for the j dimension  
    # W55 should have shape (5, rank_55) for the k dimension
    # 
    # But wait, in the standard tensor decomposition for matrix mult, the factor
    # matrices have shapes related to the matrix dimensions. Let me reconsider.
    #
    # For multiplying an (a x b) matrix by a (b x c) matrix:
    # - U has shape (a*b, rank)
    # - V has shape (b*c, rank)
    # - W has shape (c*a, rank)
    #
    # For 5x5x5: a=5, b=5, c=5
    # - U55 shape: (25, rank_55)
    # - V55 shape: (25, rank_55)
    # - W55 shape: (25, rank_55)
    #
    # In our 4x5x5 case: n=4, m=5, p=5
    # - U shape: (4*5, total_rank) = (20, total_rank)
    # - V shape: (5*5, total_rank) = (25, total_rank)
    # - W shape: (5*4, total_rank) = (20, total_rank)
    
    # For the 5x5x5 sub-tensor corresponding to i=1..4:
    # The rows of the "first matrix" are i=1..4 (4 rows), but we're treating it as 5x5x5
    # This doesn't quite work directly. Let me use a different approach.
    
    # Actually, let's just use ALS with a good initial guess and more iterations.
    return None

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    
    best_result = None
    best_rank = 100  # schoolbook
    
    # Try to find a decomposition with rank as low as possible
    # Start from reasonable ranks and go up
    # Theoretical lower bound for 4x5x5 is around 45-50
    # Known results suggest rank around 50-55 might be achievable
    
    for target_rank in range(48, 65):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(8):
            if time.time() > deadline - 20:
                break
            
            U, V, W = _als_decompose(T, target_rank, n_iter=300, tol=1e-10, seed=seed)
            if _verify_decomposition(U, V, W, T):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    if best_result:
        return best_result
    
    # Fallback: schoolbook decomposition with rank = n*m*p = 100
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
