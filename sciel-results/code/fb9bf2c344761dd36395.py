import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T = np.zeros((U_size, V_size, W_size))
    indices_u = []
    indices_v = []
    indices_w = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                indices_u.append(u_idx)
                indices_v.append(v_idx)
                indices_w.append(w_idx)
                
    indices_u = np.array(indices_u)
    indices_v = np.array(indices_v)
    indices_w = np.array(indices_w)
    
    start_time = time.time()
    deadline = start_time + 110.0

    def verify(U, V, W):
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def residual(params, R):
        # params is a flat array of length R * (n*m + m*p + p*n)
        U = params[:R*U_size].reshape(U_size, R)
        V = params[R*U_size:R*U_size + R*V_size].reshape(V_size, R)
        W = params[R*U_size + R*V_size:].reshape(W_size, R)
        
        # Compute sum_r U[u,r] V[v,r] W[w,r] for the 18 points
        U_sel = U[indices_u]
        V_sel = V[indices_v]
        W_sel = W[indices_w]
        
        # Product element-wise then sum over R
        prod = U_sel * V_sel * W_sel
        result = prod.sum(axis=1)
        
        # Target is 1 for all 18 points
        return result - 1.0

    def als_decompose(rank, U_init, V_init, W_init, max_iter=100, tol=1e-8):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iter):
            # Update U
            WV = (W * V).T  # (rank, W_size) * (rank, V_size) -> (W_size, V_size)
            # For each u, minimize sum_v sum_w (T[u,v,w] - sum_r U[u,r] WV[r,v,w])^2
            # This is equivalent to minimizing || T_u - U_u @ WV.T ||_F^2
            # Gradient: -2 T_u + 2 U_u @ WV.T @ WV
            # Hessian: 2 WV.T @ WV
            # Solution: U_u = T_u @ (WV.T @ WV)^-1
            
            # More efficiently:
            # Let M = WV.T @ WV (rank x rank)
            # Let B = T_u @ WV (u x rank)
            # U_u = B @ M^-1
            
            # Precompute M
            M = WV @ WV.T  # (V_size, V_size) -- wait, dimensions are tricky.
            
            # Let's use the standard ALS formula.
            # Minimize || T - U V W ||_F^2 wrt U.
            # T is (U_size, V_size, W_size).
            # For fixed V, W, we have a linear system for each row of U.
            # Let's flatten V and W into a matrix M of shape (V_size * W_size, rank)
            # M[v * W_size + w, r] = V[v, r] * W[w, r]
            # Then T_flat[u, :] = T[u, :, :].flatten()
            # We want to minimize || T_flat[u, :] - U[u, :] @ M.T ||^2
            # Solution: U[u, :] = T_flat[u, :] @ (M @ M.T)^-1
            
            M = np.einsum('vr,wr->vw r', V, W) # (V_size, W_size, rank)
            M_mat = M.reshape(V_size * W_size, rank)
            MM = M_mat @ M_mat.T # (V_size*W_size, V_size*W_size)
            
            # Invert MM
            try:
                MM_inv = np.linalg.inv(MM)
            except np.linalg.LinAlgError:
                MM_inv = np.linalg.pinv(MM)
            
            # T_flat is (U_size, V_size*W_size)
            T_flat = T.reshape(U_size, V_size * W_size)
            
            # U_new = T_flat @ MM_inv @ M_mat
            U_new = T_flat @ MM_inv @ M_mat
            U = U_new
            
            # Update V
            # Minimize || T - U V W ||_F^2 wrt V.
            # Flatten U and W.
            M = np.einsum('ur,wr->uw r', U, W) # (U_size, W_size, rank)
            M_mat = M.reshape(U_size * W_size, rank)
            MM = M_mat @ M_mat.T
            try:
                MM_inv = np.linalg.inv(MM)
            except np.linalg.LinAlgError:
                MM_inv = np.linalg.pinv(MM)
                
            # T needs to be reshaped appropriately.
            # T is (U_size, V_size, W_size).
            # We want T_flat[v, :] such that T_flat[v, :] = T[:, v, :].flatten()
            # T_trans = T.transpose(1, 0, 2) -> (V_size, U_size, W_size)
            T_flat_v = T.transpose(1, 0, 2).reshape(V_size, U_size * W_size)
            
            V_new = T_flat_v @ MM_inv @ M_mat
            V = V_new
            
            # Update W
            # Minimize || T - U V W ||_F^2 wrt W.
            M = np.einsum('ur,vr->uv r', U, V) # (U_size, V_size, rank)
            M_mat = M.reshape(U_size * V_size, rank)
            MM = M_mat @ M_mat.T
            try:
                MM_inv = np.linalg.inv(MM)
            except np.linalg.LinAlgError:
                MM_inv = np.linalg.pinv(MM)
                
            # T_flat[w, :] such that T_flat[w, :] = T[:, :, w].flatten()
            # T_trans = T.transpose(2, 0, 1) -> (W_size, U_size, V_size)
            T_flat_w = T.transpose(2, 0, 1).reshape(W_size, U_size * V_size)
            
            W_new = T_flat_w @ MM_inv @ M_mat
            W = W_new
            
            # Check convergence
            # Residual calculation is expensive, so maybe just check norm of updates
            if np.linalg.norm(U_new - U) < tol and np.linalg.norm(V_new - V) < tol and np.linalg.norm(W_new - W) < tol:
                break
                
        return U, V, W

    # Try ranks from 14 upwards
    # Rank 14 is the known minimum for 2x3x3
    for rank in [14, 15, 16]:
        if time.time() > deadline:
            break
        
        # Number of restarts: give rank 14 the most chances
        max_restarts = 100 if rank == 14 else 50
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 1000 + rank)
            
            # Random initialization
            # Scaling is important. Entries in optimal solutions are often small integers or half-integers.
            # Initializing with small random values helps convergence.
            U_init = np.random.randn(U_size, rank) * 0.5
            V_init = np.random.randn(V_size, rank) * 0.5
            W_init = np.random.randn(W_size, rank) * 0.5
            
            try:
                U_opt, V_opt, W_opt = als_decompose(rank, U_init, V_init, W_init, max_iter=50)
                
                result = snap_and_verify(U_opt, V_opt, W_opt)
                if result:
                    U_final, V_final, W_final = result
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            except Exception:
                continue

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
