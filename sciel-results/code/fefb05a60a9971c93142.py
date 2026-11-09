import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin

    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Build the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W, rank

    def check_half_integer(U, V, W):
        """
        Check if rounding to nearest 0.5 gives exact decomposition.
        """
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Reconstruct and check
        try:
            T_rec = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            if np.allclose(T_rec, T, atol=1e-9):
                return U_r, V_r, W_r
        except Exception:
            pass
        return None

    def als_optimize(rank, n_restarts=10, max_iters=200):
        """
        Alternating Least Squares optimization for tensor decomposition.
        """
        best_err = float('inf')
        best_solution = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            current_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Solve for U: T_{u,v,w} = sum_r U_{u,r} V_{r,v} W_{r,w}
                # Reshape T to (dim_u, dim_v*dim_w)
                T_u = T.reshape(dim_u, dim_v * dim_w)
                # V_r is (dim_v, rank) -> V_r^T is (rank, dim_v)
                # W_r is (dim_w, rank)
                # We need M such that U M = T_u
                # M_{r, (v,w)} = V_{v,r} W_{w,r}
                # This is the outer product of columns of V^T and W?
                # Let's construct the matrix A_u such that U A_u = T_u
                # A_u[:, (v,w)] = sum? No, it's linear in U.
                # T_u[u, (v,w)] = sum_r U[u,r] * (V[v,r] * W[w,r])
                # So A_u[(v,w), r] = V[v,r] * W[w,r]
                
                # Construct A_u
                # V is (dim_v, rank), W is (dim_w, rank)
                # We want a matrix of shape (dim_v*dim_w, rank)
                # Row index is (v,w), col index is r.
                # A_u[(v,w), r] = V[v,r] * W[w,r]
                
                # Efficient construction:
                # V_T = V.T (rank, dim_v)
                # W_T = W.T (rank, dim_w)
                # We can use einsum to create A_u
                # A_u = np.einsum('vr,wr->vwr', V, W) -> shape (dim_v, dim_w, rank)
                # Reshape to (dim_v*dim_w, rank)
                A_u = np.einsum('vr,wr->vwr', V, W).reshape(dim_v * dim_w, rank)
                
                # Solve U A_u = T_u
                # U = T_u A_u^+ (pseudo-inverse)
                try:
                    U_new = np.linalg.lstsq(A_u, T_u, rcond=None)[0].T
                except:
                    break
                
                # Solve for V
                T_v = T.reshape(dim_u * dim_w, dim_v).T # Shape (dim_v, dim_u*dim_w)
                # T_v[v, (u,w)] = sum_r V[v,r] * (U[u,r] * W[w,r])
                # A_v[(u,w), r] = U[u,r] * W[w,r]
                A_v = np.einsum('ur,wr->uwr', U_new, W).reshape(dim_u * dim_w, rank)
                try:
                    V_new = np.linalg.lstsq(A_v, T_v, rcond=None)[0].T
                except:
                    break

                # Solve for W
                T_w = T.reshape(dim_u * dim_v, dim_w).T # Shape (dim_w, dim_u*dim_v)
                # T_w[w, (u,v)] = sum_r W[w,r] * (U[u,r] * V[v,r])
                # A_w[(u,v), r] = U[u,r] * V[v,r]
                A_w = np.einsum('ur,vr->uvr', U_new, V_new).reshape(dim_u * dim_v, rank)
                try:
                    W_new = np.linalg.lstsq(A_w, T_w, rcond=None)[0].T
                except:
                    break

                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = np.linalg.norm(T - T_rec)
                
                if err < 1e-9:
                    break
                if err < current_err:
                    current_err = err
                    if current_err < best_err:
                        best_err = current_err
                        best_solution = (U.copy(), V.copy(), W.copy())
                        # Check if it's a valid half-integer solution
                        sol = check_half_integer(U, V, W)
                        if sol is not None:
                            return sol
            # End ALS loop
        # End restarts
        
        # Final check on best solution found
        if best_solution is not None:
            sol = check_half_integer(best_solution[0], best_solution[1], best_solution[2])
            if sol is not None:
                return sol
        return None

    # Try to find lower rank
    # Start from 59, go up to 55 (if 59 fails, 58 is unlikely to succeed quickly without more time)
    # We prioritize 59 as it's the most likely candidate for improvement.
    
    random.seed(42)
    np.random.seed(42)
    
    for rank in range(59, 54, -1):
        if time.time() > deadline:
            break
        found = als_optimize(rank, n_restarts=20, max_iters=50)
        if found is not None:
            U, V, W = found
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Schoolbook
    U, V, W, rank = get_schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
