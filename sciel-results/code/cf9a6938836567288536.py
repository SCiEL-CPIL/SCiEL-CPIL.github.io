import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 12
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def compute_residual(U, V, W):
        """Compute Frobenius norm of tensor difference."""
        C = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        return np.linalg.norm(C - T)
    
    def snap_to_half_int(arr):
        """Snap to nearest 0.5."""
        return np.round(arr * 2) / 2.0
    
    def verify(U, V, W):
        """Check if entries are half-integers and decomposition is exact."""
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-6): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-6): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-6): return False
        
        C = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        
        return np.allclose(C, T, atol=1e-6)
    
    def als_step(U, V, W, max_iter=50, tol=1e-8):
        """Alternating least squares to minimize tensor residual."""
        current_res = compute_residual(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: fix V, W, solve linear system for U
            T_reshaped = T.reshape(n*m, m*p*p*n)
            M = np.zeros((target_rank, m*p*p*n))
            for c in range(target_rank):
                M[c] = (V[:, c] * W[:, c]).ravel()
            
            try:
                U_new = np.linalg.lstsq(M.T, T_reshaped, rcond=None)[0]
            except:
                break
            
            # Update V: fix U, W
            T_reshaped_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            M_v = np.zeros((target_rank, n*m*p*n))
            for c in range(target_rank):
                M_v[c] = (U[:, c] * W[:, c]).ravel()
            try:
                V_new = np.linalg.lstsq(M_v.T, T_reshaped_v, rcond=None)[0]
            except:
                break
            
            # Update W: fix U, V
            T_reshaped_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
            M_w = np.zeros((target_rank, n*m*m*p))
            for c in range(target_rank):
                M_w[c] = (U[:, c] * V[:, c]).ravel()
            try:
                W_new = np.linalg.lstsq(M_w.T, T_reshaped_w, rcond=None)[0]
            except:
                break
            
            new_res = compute_residual(U_new, V_new, W_new)
            if abs(new_res - current_res) < tol:
                U, V, W = U_new, V_new, W_new
                break
            U, V, W = U_new, V_new, W_new
            current_res = new_res
        
        return U, V, W, compute_residual(U, V, W)
    
    def structured_init(rank):
        """Create structured initialization based on schoolbook patterns."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with random signs and small perturbations
        for c in range(rank):
            # Randomly assign non-zero entries
            u_idx = np.random.choice(n*m, size=np.random.randint(1, 4), replace=False)
            v_idx = np.random.choice(m*p, size=np.random.randint(1, 4), replace=False)
            w_idx = np.random.choice(p*n, size=np.random.randint(1, 4), replace=False)
            U[u_idx, c] = np.random.choice([-1, 1], size=len(u_idx))
            V[v_idx, c] = np.random.choice([-1, 1], size=len(v_idx))
            W[w_idx, c] = np.random.choice([-1, 1], size=len(w_idx))
        
        return U, V, W
    
    best_score = float('inf')
    best_solution = None
    
    # Try multiple random initializations with ALS
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Use structured initialization
        U, V, W = structured_init(target_rank)
        
        U, V, W, res = als_step(U, V, W, max_iter=40)
        
        if res < best_score:
            best_score = res
            # Try snapping
            U_snap = snap_to_half_int(U)
            V_snap = snap_to_half_int(V)
            W_snap = snap_to_half_int(W)
            
            if verify(U_snap, V_snap, W_snap):
                # Convert to list format
                U_list = [[float(x) for x in row] for row in U_snap]
                V_list = [[float(x) for x in row] for row in V_snap]
                W_list = [[float(x) for x in row] for row in W_snap]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try without snapping if residual is very small
        if res < 1e-10:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no solution found, return fallback
    if best_solution:
        return best_solution
    else:
        return fallback
