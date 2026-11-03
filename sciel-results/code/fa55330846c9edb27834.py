import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def verify_halfint(U, V, W):
        """Verify that U, V, W (as half-integers) exactly decompose T."""
        # Convert to integers (×2)
        U_int = np.round(np.array(U, dtype=np.float64) * 2)
        V_int = np.round(np.array(V, dtype=np.float64) * 2)
        W_int = np.round(np.array(W, dtype=np.float64) * 2)
        
        # Check they're actually integers
        if not np.allclose(U_int, np.array(U) * 2):
            return False
        if not np.allclose(V_int, np.array(V) * 2):
            return False
        if not np.allclose(W_int, np.array(W) * 2):
            return False
        
        U_int = U_int.astype(np.int64)
        V_int = V_int.astype(np.int64)
        W_int = W_int.astype(np.int64)
        
        # Compute the tensor product
        # T_calc[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
        # We need T_calc == 8 where T == 1, and 0 elsewhere
        # Use einsum for efficiency
        T_calc = np.einsum('ac,bc,cc->abc', U_int, V_int, W_int, optimize=True)
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if T_calc[a, b, c] != 8:
                        return False
        
        # Check zero entries
        # T_calc should be 0 where T is 0
        # Create mask of where T is non-zero
        T_mask = np.zeros((n*m, m*p, p*n), dtype=np.bool_)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_mask[a, b, c] = True
        
        # Check that T_calc is 0 where T_mask is False
        if not np.all(T_calc[~T_mask] == 0):
            return False
        
        return True
    
    def local_search(U, V, W, max_iter=100):
        """Try to refine a solution using local search on half-integer values."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        # Round to half-integers
        U = np.round(U * 2) / 2.0
        V = np.round(V * 2) / 2.0
        W = np.round(W * 2) / 2.0
        
        best_error = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute current error
            U_int = np.round(U * 2).astype(np.int64)
            V_int = np.round(V * 2).astype(np.int64)
            W_int = np.round(W * 2).astype(np.int64)
            
            T_calc = np.einsum('ac,bc,cc->abc', U_int, V_int, W_int, optimize=True)
            
            # Target: 8 where T==1, 0 elsewhere
            T_target = np.zeros_like(T_calc)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T_target[a, b, c] = 8
            
            error = np.sum((T_calc - T_target) ** 2)
            
            if error == 0:
                return U, V, W, True
            
            if error < best_error:
                best_error = error
            
            # Try small perturbations
            improved = False
            for _ in range(50):
                # Pick a random matrix and a random entry
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, U.shape[1]-1)
                    old_val = U[row, col]
                    # Try nearby half-integer values
                    delta = random.choice([-1.0, -0.5, 0.5, 1.0])
                    new_val = old_val + delta
                    U[row, col] = new_val
                    
                    if verify_halfint(U, V, W):
                        return U, V, W, True
                    U[row, col] = old_val
                    
                elif mat_choice == 'V':
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, V.shape[1]-1)
                    old_val = V[row, col]
                    delta = random.choice([-1.0, -0.5, 0.5, 1.0])
                    new_val = old_val + delta
                    V[row, col] = new_val
                    
                    if verify_halfint(U, V, W):
                        return U, V, W, True
                    V[row, col] = old_val
                    
                else:  # W
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, W.shape[1]-1)
                    old_val = W[row, col]
                    delta = random.choice([-1.0, -0.5, 0.5, 1.0])
                    new_val = old_val + delta
                    W[row, col] = new_val
                    
                    if verify_halfint(U, V, W):
                        return U, V, W, True
                    W[row, col] = old_val
        
        return U, V, W, False
    
    def structured_init(rank, seed):
        """Create a structured initial guess for rank-r decomposition."""
        rng = random.Random(seed)
        
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        # Strategy: assign each rank-1 component to cover a subset of (i,j,k)
        # For each component c, choose a pattern that partially satisfies the constraints
        
        for c in range(rank):
            # Randomly choose which (i,j,k) tuples this component will help with
            # But we need to be systematic
            
            # Try a pattern where U, V, W have specific structures
            # For example, U[i*m+j, c] depends on i,j
            # V[j*p+k, c] depends on j,k
            # W[k*n+i, c] depends on k,i
            
            # Random half-integer values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            for i in range(n):
                for j in range(m):
                    U[i*m+j, c] = rng.choice(vals)
            
            for j in range(m):
                for k in range(p):
                    V[j*p+k, c] = rng.choice(vals)
            
            for k in range(p):
                for i in range(n):
                    W[k*n+i, c] = rng.choice(vals)
        
        return U, V, W
    
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n*m, m*p, p*n)  # 21
    schoolbook_rank = n * m * p  # 63
    
    best_result = None
    
    # First, try structured search for ranks near the lower bound
    for rank in range(lower_bound, min(lower_bound + 20, schoolbook_rank)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple structured initializations
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            U, V, W = structured_init(rank, seed + rank * 1000)
            
            # Run local search
            U, V, W, success = local_search(U, V, W, max_iter=50)
            
            if success:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_result is not None:
            break
    
    # If not found, try ALS-based approach with rounding
    if best_result is None:
        for rank in range(lower_bound, min(lower_bound + 15, schoolbook_rank)):
            if time.time() > deadline - 15:
                break
            
            for seed in range(10):
                if time.time() > deadline - 15:
                    break
                
                rng = np.random.RandomState(seed + rank * 10000)
                U = rng.randn(n*m, rank)
                V = rng.randn(m*p, rank)
                W = rng.randn(p*n, rank)
                
                # ALS iterations with rounding
                for it in range(30):
                    if time.time() > deadline - 15:
                        break
                    
                    # Update U
                    for a in range(n*m):
                        # Build M_a: M_a[b*p*n+c, :] = V[b,:] * W[c,:]
                        V_rep = np.repeat(V, p*n, axis=0)  # (m*p*p*n, rank)
                        W_rep = np.tile(W, (m*p, 1))       # (m*p*p*n, rank)
                        M_a = V_rep * W_rep
                        t_a = T[a, :, :].flatten()
                        U[a, :], _, _, _ = np.linalg.lstsq(M_a, t_a, rcond=None)
                    
                    # Update V
                    for b in range(m*p):
                        U_rep = np.repeat(U, p*n, axis=0)  # (n*m*p*n, rank)
                        W_rep = np.tile(W, (n*m, 1))       # (n*m*p*n, rank)
                        M_b = U_rep * W_rep
                        t_b = T[:, b, :].flatten()
                        V[b, :], _, _, _ = np.linalg.lstsq(M_b, t_b, rcond=None)
                    
                    # Update W
                    for c in range(p*n):
                        U_rep = np.repeat(U, m*p, axis=0)  # (n*m*m*p, rank)
                        V_rep = np.tile(V, (n*m, 1))       # (n*m*m*p, rank)
                        M_c = U_rep * V_rep
                        t_c = T[:, :, c].flatten()
                        W[c, :], _, _, _ = np.linalg.lstsq(M_c, t_c, rcond=None)
                    
                    # Round to half-integers and check
                    if it % 3 == 0:
                        U_rounded = np.round(U * 2) / 2.0
                        V_rounded = np.round(V * 2) / 2.0
                        W_rounded = np.round(W * 2) / 2.0
                        
                        if verify_halfint(U_rounded, V_rounded, W_rounded):
                            best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                            break
                if best_result is not None:
                    break
            
            if best_result is not None:
                break
    
    # Fallback to schoolbook
    if best_result is None:
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
