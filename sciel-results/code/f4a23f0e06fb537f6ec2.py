import time
import numpy as np
from numpy.linalg import svd

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Dimensions of the tensor modes
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n

    # Construct the multiplication tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # Note: The prompt specifies W[k*n+i, r], so mode C is indexed by k*n+i.
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    # Fallback to schoolbook if search fails
    U_school = np.zeros((dim_A, best_rank))
    V_school = np.zeros((dim_B, best_rank))
    W_school = np.zeros((dim_C, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Search for lower rank using I-SVD
    # Start from a reasonable guess. Strassen 4x4x4 is 49. 
    # 4x4x7 might be around 49 + 4*3 = 61? Or maybe less.
    # Let's try ranks from 100 down to 40.
    
    def run_isvd(tensor, rank, iters=20, tol=1e-8):
        """
        Iterative SVD for tensor decomposition.
        tensor: 3D array
        rank: target rank
        returns: U, V, W matrices
        """
        # Initialize with random or truncated SVD of matricizations
        # Simple initialization: random
        U = np.random.randn(tensor.shape[0], rank)
        V = np.random.randn(tensor.shape[1], rank)
        W = np.random.randn(tensor.shape[2], rank)
        
        # Normalize
        U /= np.linalg.norm(U, axis=0)
        V /= np.linalg.norm(V, axis=0)
        W /= np.linalg.norm(W, axis=0)
        
        for _ in range(iters):
            # Update U: matricize T along mode 1
            # T_(1) is shape (dim_A, dim_B * dim_C)
            T1 = tensor.transpose(0, 1, 2).reshape(tensor.shape[0], -1)
            # Effective matrix M = T1 @ (V ⊗ W)^T
            # (V ⊗ W) is shape (dim_B * dim_C, rank)
            # M = T1 @ (V ⊗ W)
            # T1 is (dim_A, dim_B*dim_C)
            # V ⊗ W is (dim_B*dim_C, rank)
            # M = T1 @ (V ⊗ W)
            # Then SVD M
            VW = np.kron(V, W)
            M = T1 @ VW
            # SVD
            try:
                U_new, s, Vt = svd(M, full_matrices=False)
                U = U_new[:, :rank]
            except:
                break
            
            # Update V
            T2 = tensor.transpose(1, 0, 2).reshape(tensor.shape[1], -1)
            UW = np.kron(U, W)
            M = T2 @ UW
            try:
                V_new, s, Vt = svd(M, full_matrices=False)
                V = V_new[:, :rank]
            except:
                break
                
            # Update W
            T3 = tensor.transpose(2, 0, 1).reshape(tensor.shape[2], -1)
            UV = np.kron(U, V)
            M = T3 @ UV
            try:
                W_new, s, Vt = svd(M, full_matrices=False)
                W = W_new[:, :rank]
            except:
                break
                
            # Check convergence (optional)
            
        return U, V, W

    def check_decomposition(U, V, W, T):
        """
        Check if U, V, W exactly decompose T.
        Reconstruct T_approx = sum U_r V_r W_r
        """
        rank = U.shape[1]
        T_approx = np.zeros_like(T)
        for r in range(rank):
            T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(T - T_approx)
        return err

    def round_and_verify(U, V, W, T):
        """
        Round U, V, W to nearest 0.5 and verify.
        """
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        err = check_decomposition(U_r, V_r, W_r, T)
        if err < 1e-6:
            return U_r, V_r, W_r, True
        return None, None, None, False

    # Try ranks
    # We know schoolbook is 112.
    # Strassen 4x4x4 is 49.
    # Let's try a range.
    # To save time, we'll try a few specific ranks and a small sweep.
    
    ranks_to_try = [90, 85, 80, 75, 70, 65, 60, 55, 50]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        # Run I-SVD with a few restarts to avoid local minima
        found = False
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
            U, V, W = run_isvd(T, R, iters=30)
            U_r, V_r, W_r, success = round_and_verify(U, V, W, T)
            if success:
                best_rank = R
                best_result = (U_r, V_r, W_r)
                found = True
                break
        
        if found:
            break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
