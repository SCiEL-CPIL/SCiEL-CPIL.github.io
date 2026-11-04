"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """
    Exact verification using Python integers to avoid floating point issues.
    """
    rank = len(U[0])
    
    T = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    for r in range(rank):
        u_col = [U[i * m + j][r] for i in range(n) for j in range(m)]
        v_col = [V[j * p + k][r] for j in range(m) for k in range(p)]
        w_col = [W[k * n + i][r] for k in range(p) for i in range(n)]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    T[i][j][k] += u_val * v_val * w_val
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i][j][k] != 1:
                    return False
    return True

def _sa_search(rank, n, m, p, deadline, max_restarts=100):
    """
    Simulated Annealing search for integer decomposition.
    """
    # Shapes: U[n, m, rank], V[m, p, rank], W[p, n, rank]
    # We use small integers. Let's try range [-3, 3] initially.
    
    best_energy = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random
        # Start with small values to encourage sparsity/simplicity
        U = np.random.randint(-2, 3, size=(n, m, rank))
        V = np.random.randint(-2, 3, size=(m, p, rank))
        W = np.random.randint(-2, 3, size=(p, n, rank))
        
        # Compute initial T and Energy
        T = np.zeros((n, m, p), dtype=np.int64)
        for r in range(rank):
            # T[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Vectorized:
            # u_r: (n, m)
            # v_r: (m, p)
            # w_r: (p, n)
            # Need to align indices.
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # This is a tensor contraction.
            # For fixed r, contribution is outer product-like but with index swaps.
            # Let's just loop r for initialization, it's small.
            u_r = U[:, :, r]
            v_r = V[:, :, r]
            w_r = W[:, :, r]
            # Construct contribution
            # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
            for i in range(n):
                for j in range(m):
                    u_val = u_r[i, j]
                    if u_val == 0: continue
                    # v_r[j, :] * w_r[:, i]
                    contrib = u_val * (v_r[j, :] * w_r[:, i])
                    T[i, j, :] += contrib
                    
        energy = np.sum((T - 1.0)**2)
        
        current_energy = energy
        temp = 10.0
        cooling = 0.95
        
        steps = 2000
        for step in range(steps):
            if time.time() > deadline:
                return None
            
            # Pick a random move
            r = random.randint(0, rank - 1)
            mat = random.randint(0, 2) # 0:U, 1:V, 2:W
            
            if mat == 0:
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                old_val = U[i, j, r]
                delta = random.choice([-1, 1])
                new_val = old_val + delta
                # Limit range
                if abs(new_val) > 5:
                    continue
                # Calculate delta T
                # T[i,j,k] changes by delta * V[j,k,r] * W[k,i,r]
                # Only T[i,j,:] changes
                # v_r[j, :] shape (p,)
                # w_r[:, i] shape (p,)
                change_vec = delta * (V[j, :, r] * W[:, i, r])
                old_T_slice = T[i, j, :]
                new_T_slice = old_T_slice + change_vec
                
                # Calculate delta Energy
                # E = sum (T-1)^2
                # Delta E = sum_new (new-1)^2 - sum_old (old-1)^2
                # Only indices in slice change
                old_e_slice = np.sum((old_T_slice - 1.0)**2)
                new_e_slice = np.sum((new_T_slice - 1.0)**2)
                delta_e = new_e_slice - old_e_slice
                
                # Accept?
                if delta_e < 0 or random.random() < np.exp(-delta_e / temp):
                    U[i, j, r] = new_val
                    T[i, j, :] = new_T_slice
                    current_energy += delta_e
            elif mat == 1:
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                old_val = V[j, k, r]
                delta = random.choice([-1, 1])
                new_val = old_val + delta
                if abs(new_val) > 5:
                    continue
                # T[i,j,k] changes by U[i,j,r] * delta * W[k,i,r]
                # T[:, j, k] changes
                # u_r[:, j] shape (n,)
                # w_r[k, :] shape (n,)
                change_vec = delta * (U[:, j, r] * W[k, :, r])
                old_T_slice = T[:, j, k]
                new_T_slice = old_T_slice + change_vec
                
                old_e_slice = np.sum((old_T_slice - 1.0)**2)
                new_e_slice = np.sum((new_T_slice - 1.0)**2)
                delta_e = new_e_slice - old_e_slice
                
                if delta_e < 0 or random.random() < np.exp(-delta_e / temp):
                    V[j, k, r] = new_val
                    T[:, j, k] = new_T_slice
                    current_energy += delta_e
            else:
                k = random.randint(0, p - 1)
                i = random.randint(0, n - 1)
                old_val = W[k, i, r]
                delta = random.choice([-1, 1])
                new_val = old_val + delta
                if abs(new_val) > 5:
                    continue
                # T[i,j,k] changes by U[i,j,r] * V[j,k,r] * delta
                # T[i, :, k] changes
                # u_r[i, :] shape (m,)
                # v_r[:, k] shape (m,)
                change_vec = delta * (U[i, :, r] * V[:, k, r])
                old_T_slice = T[i, :, k]
                new_T_slice = old_T_slice + change_vec
                
                old_e_slice = np.sum((old_T_slice - 1.0)**2)
                new_e_slice = np.sum((new_T_slice - 1.0)**2)
                delta_e = new_e_slice - old_e_slice
                
                if delta_e < 0 or random.random() < np.exp(-delta_e / temp):
                    W[k, i, r] = new_val
                    T[i, :, k] = new_T_slice
                    current_energy += delta_e
            
            temp *= cooling
            
            if current_energy == 0:
                # Found solution!
                # Convert back to list of lists for output
                U_out = U.transpose(1, 2, 0).reshape(n * m, rank).tolist() # Wait, U is [n, m, r]. Output needs [n*m, r] where index is i*m+j.
                # U[i, j, r] -> U_out[i*m+j][r]
                U_out = [[0]*rank for _ in range(n*m)]
                V_out = [[0]*rank for _ in range(m*p)]
                W_out = [[0]*rank for _ in range(p*n)]
                
                for i in range(n):
                    for j in range(m):
                        for r in range(rank):
                            U_out[i*m+j][r] = int(U[i, j, r])
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            V_out[j*p+k][r] = int(V[j, k, r])
                for k in range(p):
                    for i in range(n):
                        for r in range(rank):
                            W_out[k*n+i][r] = int(W[k, i, r])
                
                # Verify
                if _verify_exact(U_out, V_out, W_out, n, m, p):
                    return U_out, V_out, W_out
                else:
                    # Floating point or logic error?
                    pass
                    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try SA for lower ranks
    for target_rank in [35, 34, 33, 32, 31, 30, 29, 28, 27]:
        if time.time() > deadline - 10:
            break
            
        result = _sa_search(target_rank, n, m, p, deadline - 5)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_sb, "V": V_sb, "W": W_sb}
