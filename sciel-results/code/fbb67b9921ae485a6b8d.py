"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    Start with the rank-35 Strassen-based decomposition (5 blocks x 7 Strassen products),
    then attempt rank reduction using alternating least squares (ALS) with multiple
    restarts and adaptive learning. The block-diagonal structure of B means each 2x2x2
    block is independent, so we can exploit this structure in the optimization.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Build the rank-35 decomposition first
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(35):
        u = np.array(U35)[:, r]
        v = np.array(V35)[:, r]
        w = np.array(W35)[:, r]
        T += np.einsum('i,j,k->ijk', u, v, w)
    
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # Try ranks from 30 up to 34 (more conservative range with better optimization)
    for target_rank in range(30, 35):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_decomposition_als(T, n, m, p, target_rank, 
                                             time_budget=min(20, deadline - time.time() - 5))
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return best_result


def _build_strassen_decomposition(n, m, p):
    """Build the rank-35 Strassen-based decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    w_contrib[5][3] = 1
    
    w_contrib[6][0] = 1
    
    col = 0
    for b in range(5):
        for r in range(7):
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b+2][col] = w_contrib[r][1]
            W[4*b+1][col] = w_contrib[r][2]
            W[4*b+3][col] = w_contrib[r][3]
            
            col += 1
            
    return U, V, W


def _try_rank_decomposition_als(T, n, m, p, target_rank, time_budget=20):
    """
    Try to find a rank-target_rank decomposition of tensor T using Alternating Least Squares (ALS).
    
    For each factor, we solve a linear system by fixing the other two factors.
    U[i,r] is solved by: for each i, U[i,:] solves min ||T[i,:,:] - U[i,:] V^T W^T||_F
    This can be rewritten using the property that T[i,:,:] = sum_r U[i,r] v_r w_r^T
    So T[i,:,:] = U[i,:] ⊙ (V ⊗ W) where ⊙ is Khatri-Rao product
    
    Actually, more precisely:
    T[i,:,:] = sum_r U[i,r] * (v_r ⊗ w_r)
    So if we define M_i = T[i,:,:].reshape(-1), and K = [v_1⊗w_1, ..., v_R⊗w_R],
    then M_i = K^T @ U[i,:]  (or U[i,:] @ K depending on convention)
    
    Let me use: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    For fixed i: T[i,j,k] = sum_r U[i,r] (V[j,r] W[k,r])
    So T[i,:,:] = sum_r U[i,r] (v_r ⊗ w_r)
    
    Let K be a (dim_v * dim_w) x target_rank matrix where K[j*p+k, r] = V[j,r]*W[k,r]
    Then T[i,:,:].reshape(-1) = K^T @ U[i,:]
    So U[i,:] = (K K^T)^{-1} K T[i,:,:].reshape(-1)
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    np.random.seed(42)
    
    # Try multiple random initializations
    for trial in range(8):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization with proper scaling
        U = np.random.randn(dim_u, target_rank)
        V = np.random.randn(dim_v, target_rank)
        W = np.random.randn(dim_w, target_rank)
        
        # Normalize to prevent numerical issues
        for r in range(target_rank):
            norm = max(1e-10, np.linalg.norm(U[:, r]))
            U[:, r] /= norm
            norm = max(1e-10, np.linalg.norm(V[:, r]))
            V[:, r] /= norm
            norm = max(1e-10, np.linalg.norm(W[:, r]))
            W[:, r] /= norm
        
        # ALS iterations
        for iteration in range(200):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each i, solve U[i,:] from T[i,:,:] = sum_r U[i,r] (v_r ⊗ w_r)
            # K[j*p+k, r] = V[j,r] * W[k,r]
            K = np.einsum('jr,kr->jkr', V, W).reshape(dim_v * dim_w, target_rank)
            KT = K.T  # (target_rank, dim_v*dim_w)
            KTK = KT @ K  # (target_rank, target_rank)
            reg = 1e-10 * np.eye(target_rank)
            
            for i in range(dim_u):
                Ti = T[i, :, :].reshape(-1)  # (dim_v*dim_w,)
                rhs = K.T @ Ti  # (target_rank,)
                U[i, :] = np.linalg.solve(KTK + reg, rhs)
            
            # Update V: for each j, solve V[j,:] from T[:,j,:] = sum_r V[j,r] (u_r ⊗ w_r)
            # L[i*p+k, r] = U[i,r] * W[k,r]
            L = np.einsum('ir,kr->ikr', U, W).reshape(dim_u * dim_w, target_rank)
            LT = L.T
            LTL = LT @ L
            
            for j in range(dim_v):
                Tj = T[:, j, :].reshape(-1)  # (dim_u*dim_w,)
                rhs = L.T @ Tj
                V[j, :] = np.linalg.solve(LTL + reg, rhs)
            
            # Update W: for each k, solve W[k,:] from T[:,:,k] = sum_r W[k,r] (u_r ⊗ v_r)
            # M[i*p+j, r] = U[i,r] * V[j,r]
            M = np.einsum('ir,jr->ijr', U, V).reshape(dim_u * dim_v, target_rank)
            MT = M.T
            MTM = MT @ M
            
            for k in range(dim_w):
                Tk = T[:, :, k].reshape(-1)  # (dim_u*dim_v,)
                rhs = M.T @ Tk
                W[k, :] = np.linalg.solve(MTM + reg, rhs)
            
            # Check convergence
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            loss = np.max(np.abs(T - T_est))
            if loss < 1e-10:
                break
        
        # Final check with tighter tolerance
        T_final = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.max(np.abs(T - T_final)) < 1e-8:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return U_list, V_list, W_list
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute tensor from decomposition
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Compute expected tensor for matrix multiplication
    # A is n x m, B is m x p, C is n x p
    # U index: i*m + j (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k (j in 0..m-1, k in 0..p-1)  
    # W index: k*n + i (k in 0..p-1, i in 0..n-1) -- trace convention
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # T[u,v,w] = 1 if u=i*m+j, v=j'*p+k', w=k''*n+i'', and j==j', i==i'', k==k''
    
    T_expected = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_expected[u, v, w] = 1
    
    return np.max(np.abs(T - T_expected)) < 1e-8
# EVOLVE-BLOCK-END
