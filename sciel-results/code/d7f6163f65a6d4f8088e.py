"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for row in range(n):
        for col in range(p):
            for j in range(m):
                a_idx = row * m + j
                b_idx = j * p + col
                c_idx = col * n + row
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Generate schoolbook decomposition
    def generate_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, schoolbook_rank = generate_schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Function to check if a decomposition is valid and snap to integers/half-integers
    def try_snap(U, V, W):
        # Snap values to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check error after snapping
        T_approx = np.einsum('is,js,ks->ijk', U_snap, V_snap, W_snap)
        err = np.max(np.abs(T_approx - T))
        if err < 1e-9:
            return U_snap, V_snap, W_snap, True
        return None, None, None, False
    
    # Try to reduce rank from schoolbook_rank down to 50
    for rank in range(schoolbook_rank, 49, -1):
        if time.time() > deadline - 10:
            break
        
        # Initialize with the first 'rank' columns of the schoolbook decomposition
        U_init = U_school[:, :rank]
        V_init = V_school[:, :rank]
        W_init = W_school[:, :rank]
        
        # Flatten parameters
        def params_to_matrices(params):
            U = params[:nm*rank].reshape(nm, rank)
            V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = params[nm*rank+mp*rank:].reshape(pn, rank)
            return U, V, W
        
        def matrices_to_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U, V, W = params_to_matrices(params)
            T_approx = np.einsum('is,js,ks->ijk', U, V, W)
            return np.sum((T_approx - T)**2)
        
        # Use least squares optimization
        x0 = matrices_to_params(U_init, V_init, W_init)
        
        try:
            res = opt.least_squares(objective, x0, method='trf', max_nfev=10000)
        except Exception:
            continue
            
        if res.success or objective(res.x) < 1e-10:
            U_opt, V_opt, W_opt = params_to_matrices(res.x)
            U_snap, V_snap, W_snap, success = try_snap(U_opt, V_opt, W_opt)
            if success:
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                best_rank = rank
                break # Found a valid lower rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
