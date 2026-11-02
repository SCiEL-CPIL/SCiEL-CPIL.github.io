import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition"""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Vectorized verification"""
        R = U.shape[1]
        # Build all required (i,j,k) triples
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[j * p + k]) * 0  # placeholder
                    # Actually need triple product
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        errors += 1
        return errors == 0
    
    def count_errors(U, V, W):
        """Count how many (i,j,k) triples have wrong value"""
        R = U.shape[1]
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        errors += 1
        return errors
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Random search with various value sets and sparsity
    value_sets = [
        [-1, 0, 1],
        [-1, -0.5, 0, 0.5, 1],
        [-2, -1, -0.5, 0, 0.5, 1, 2],
        [0, 1],
        [-1, 0, 0, 0, 1],  # sparse
    ]
    
    for rank in range(rank_schoolbook - 1, 14, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for vs_idx, values in enumerate(value_sets):
            if time.time() > deadline - 10:
                break
            
            # Number of attempts based on remaining time
            max_attempts = 2000
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                # Generate random matrices with chosen value set
                U = np.random.choice(values, size=(nm, rank))
                V = np.random.choice(values, size=(mp, rank))
                W = np.random.choice(values, size=(pn, rank))
                
                # Sparse initialization: zero out many entries
                if vs_idx >= 3:  # For sparse sets, keep more zeros
                    mask_u = np.random.random((nm, rank)) > 0.5
                    mask_v = np.random.random((mp, rank)) > 0.5
                    mask_w = np.random.random((pn, rank)) > 0.5
                    U = np.where(mask_u, U, 0)
                    V = np.where(mask_v, V, 0)
                    W = np.where(mask_w, W, 0)
                
                if verify(U, V, W):
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
            
            if found:
                break
    
    if best_rank < rank_schoolbook:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    
    # Strategy 2: Local search from schoolbook - try to merge/eliminate terms
    # Start from schoolbook and try to find linear dependencies
    if time.time() < deadline - 20:
        # Try to find a lower rank by solving a linear system
        # The tensor equation is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is a 3-tensor equation, hard to solve directly
        
        # Alternative: try structured constructions
        # For 2x3 x 3x4, try to use the fact that we can share across k dimensions
        
        # Construction idea: For each (i,j), we need sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all k
        # This means for fixed (i,j), the vector [V[j*p+k,r] * W[k*n+i,r] for k] must satisfy
        # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
        
        # Let's try a more systematic random search with local improvement
        for rank in range(23, 18, -1):
            if time.time() > deadline - 15:
                break
            
            improved = False
            for attempt in range(500):
                if time.time() > deadline - 15:
                    break
                
                U = np.random.choice([-1, 0, 1], size=(nm, rank))
                V = np.random.choice([-1, 0, 1], size=(mp, rank))
                W = np.random.choice([-1, 0, 1], size=(pn, rank))
                
                err = count_errors(U, V, W)
                
                # Local search: try flipping single entries to reduce errors
                for step in range(50):
                    if err == 0:
                        best_rank = rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        improved = True
                        break
                    
                    # Try flipping a random entry in U, V, or W
                    mat_choice = random.choice(['U', 'V', 'W'])
                    if mat_choice == 'U':
                        row, col = random.randint(0, nm - 1), random.randint(0, rank - 1)
                        old_val = U[row, col]
                        new_val = -old_val if old_val != 0 else (1 if random.random() < 0.5 else -1)
                        if new_val == old_val:
                            new_val = 1 - old_val
                        U[row, col] = new_val
                    elif mat_choice == 'V':
                        row, col = random.randint(0, mp - 1), random.randint(0, rank - 1)
                        old_val = V[row, col]
                        new_val = -old_val if old_val != 0 else (1 if random.random() < 0.5 else -1)
                        if new_val == old_val:
                            new_val = 1 - old_val
                        V[row, col] = new_val
                    else:
                        row, col = random.randint(0, pn - 1), random.randint(0, rank - 1)
                        old_val = W[row, col]
                        new_val = -old_val if old_val != 0 else (1 if random.random() < 0.5 else -1)
                        if new_val == old_val:
                            new_val = 1 - old_val
                        W[row, col] = new_val
                    
                    new_err = count_errors(U, V, W)
                    if new_err > err:
                        # Revert
                        if mat_choice == 'U':
                            U[row, col] = old_val
                        elif mat_choice == 'V':
                            V[row, col] = old_val
                        else:
                            W[row, col] = old_val
                    else:
                        err = new_err
                
                if improved:
                    break
            
            if improved:
                break
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
