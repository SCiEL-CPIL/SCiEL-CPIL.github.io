import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Tensor size
    dim_u = n * m  # 8
    dim_v = m * p  # 20
    dim_w = p * n  # 10
    
    # Create the target tensor T
    # T[i, j, k] = 1 if i,j,k correspond to a valid multiplication term
    # i = i_A * m + j_A
    # j = j_A * p + k_A
    # k = k_A * n + i_C
    # We need i_A = i_C and j_A = k_A? No.
    # A is n x m, B is m x p.
    # C_{i_C, k_C} = sum_{j} A_{i_C, j} B_{j, k_C}.
    # So the term is A_{i_C, j} B_{j, k_C}.
    # The index in U is (i_C, j).
    # The index in V is (j, k_C).
    # The index in W is (k_C, i_C).
    # So T[(i_C*m + j), (j*p + k_C), (k_C*n + i_C)] = 1.
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i_C in range(n):
        for j in range(m):
            for k_C in range(p):
                u_idx = i_C * m + j
                v_idx = j * p + k_C
                w_idx = k_C * n + i_C
                T[u_idx, v_idx, w_idx] = 1.0

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((dim_u, rank_schoolbook))
    V_sb = np.zeros((dim_v, rank_schoolbook))
    W_sb = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i_C in range(n):
        for j in range(m):
            for k_C in range(p):
                u_idx = i_C * m + j
                v_idx = j * p + k_C
                w_idx = k_C * n + i_C
                U_sb[u_idx, col] = 1
                V_sb[v_idx, col] = 1
                W_sb[w_idx, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try to find lower rank decompositions
    for R in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline:
            break
            
        # Initialize randomly
        np.random.seed(42 + R)
        U = np.random.randn(dim_u, R)
        V = np.random.randn(dim_v, R)
        W = np.random.randn(dim_w, R)
        
        # Alternating Least Squares
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Fix V, W, solve for U
            # T[:, j, :] = U @ (V[j, :] * W.T)
            # We want U such that U @ (V * W.T) approximates T
            # This is a linear system for U.
            # T_flat = (U @ (V * W.T)).T ? No.
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # For fixed j, T[:, j, :] = U @ (V[j, :] * W.T)
            # Let M_j = V[j, :] * W.T  (shape R x dim_w)
            # T[:, j, :] (shape dim_u x dim_w) = U @ M_j
            # We can solve for U by least squares over all j.
            # Stack M_j for all j to get M (shape dim_v x R x dim_w)?
            # Better: T (dim_u x dim_v x dim_w) = U (dim_u x R) @ (V (R x dim_v) * W.T (R x dim_w))?
            # Let's use the property: vec(T) = (W kron V) vec(U)? No.
            # Let's just solve for U by minimizing sum_j || T[:, j, :] - U @ M_j ||_F^2
            # This is equivalent to minimizing || T - U @ (V * W.T) ||_F^2
            # We can flatten T and U.
            # Let's use a simple gradient step or normal equations.
            # Normal equations: U (sum_j M_j M_j.T) = sum_j T[:, j, :].T M_j
            # M_j = V[j, :] * W.T  (R x dim_w)
            # M_j M_j.T = V[j, :]^T (V[j, :] * W.T) W = V[j, :]^T (V[j, :] * (W.T W))
            # This is getting complicated. Let's use a simpler iterative method.
            
            # Update U
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # Fix j, k. T[i, j, k] = sum_r U[i, r] (V[j, r] W[k, r])
            # Let c_{jk} = V[j, :] * W[k, :]  (shape R)
            # T[:, j, k] = U @ c_{jk}
            # We want U such that U @ C = T_flat?
            # C is dim_v * dim_w x R? No.
            # Let's just do a gradient descent step.
            
            # Compute gradient of L = || T - U V W ||^2 with respect to U
            # dL/dU = 2 (T - U V W) V^T W^T ? No.
            # Let E = T - U V W (elementwise product of vectors? No, tensor product sum)
            # E[i, j, k] = T[i, j, k] - sum_r U[i, r] V[j, r] W[k, r]
            # dL/dU[i, r] = -2 sum_{j,k} E[i, j, k] V[j, r] W[k, r]
            # = -2 (E V W)[i, r] where V, W are contracted over j, k.
            # Let's compute this efficiently.
            
            # E = T - np.einsum('ir,jr,kr->ijk', U, V, W)
            # grad_U = -2 * np.einsum('ijk,jr,kr->ir', E, V, W)
            # U -= lr * grad_U
            
            # This is O(dim_u * dim_v * dim_w * R) which is 8*20*10*R = 1600 R.
            # For R=40, this is 64000 operations. Very fast.
            
            lr = 0.01
            E = T - np.einsum('ir,jr,kr->ijk', U, V, W)
            grad_U = -2 * np.einsum('ijk,jr,kr->ir', E, V, W)
            U -= lr * grad_U
            
            E = T - np.einsum('ir,jr,kr->ijk', U, V, W)
            grad_V = -2 * np.einsum('ijk,ir,kr->jr', E, U, W)
            V -= lr * grad_V
            
            E = T - np.einsum('ir,jr,kr->ijk', U, V, W)
            grad_W = -2 * np.einsum('ijk,ir,jr->kr', E, U, V)
            W -= lr * grad_W
            
        # Check error
        E = T - np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(E)
        
        if err < 1e-6:
            # Try to round to integers or half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            E_r = T - np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.linalg.norm(E_r) < 1e-9:
                best_rank = R
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break # Found a better rank, can stop searching for lower? 
                       # Or continue to see if even lower is possible?
                       # Given time, break is safer.
    
    # Convert to list of lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
