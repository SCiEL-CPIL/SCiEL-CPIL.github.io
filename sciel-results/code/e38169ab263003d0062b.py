"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
    def schoolbook():
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

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def verify_numpy(U, V, W):
        """Vectorized verification using numpy."""
        try:
            import numpy as np
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            rank = U_np.shape[1]
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            if not np.allclose(T, 1.0, atol=1e-9):
                return False
            return True
        except Exception:
            return False

    best_result = schoolbook()
    best_rank = n * m * p

    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Upper bound from schoolbook: 3*4*7 = 84
    # Try to find lower-rank decompositions using structured random search
    
    # Value pool for half-integer entries (must be integers or half-integers)
    value_pool = [0, 1, -1, 0.5, -0.5, 2, -2]
    
    # Strategy: Use Kronecker product structure based on 2x2x2 Strassen
    # Strassen decomposition for 2x2x2 has rank 7
    # We can try to exploit this structure for the 3x4x7 problem
    
    # First, try a structured approach using the Kronecker product idea
    # 3x4x7 can be decomposed using block structures
    
    # Let's try random search with the value pool, focusing on ranks that might work
    # The theoretical minimum is at least 28, but practical lower bounds are higher
    
    # Strategy: Basis change sparsification
    # Start with schoolbook and apply random invertible transformations to the rank space
    # to induce zeros in the factor matrices, allowing rank reduction.
    
    import numpy as np
    
    U_curr = np.array(best_result['U'], dtype=float)
    V_curr = np.array(best_result['V'], dtype=float)
    W_curr = np.array(best_result['W'], dtype=float)
    
    current_rank = U_curr.shape[1]
    
    # We will try to reduce the rank by finding columns that become zero
    # after a random basis transformation.
    
    max_iterations = 500 # Safety cap
    
    for iter_num in range(max_iterations):
        if time.time() > deadline:
            break
        
        if current_rank <= 28: # Theoretical lower bound
            break
            
        # Generate a random invertible matrix T of size current_rank x current_rank
        # We use a random Gaussian matrix, which is almost surely invertible.
        T = np.random.randn(current_rank, current_rank)
        
        try:
            T_inv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            continue
            
        # Apply transformation:
        # U' = U * T
        # V' = V * T_inv
        # W' = W (unchanged)
        # Note: The tensor sum is sum_r U_ijr V_jkr W_kir
        # New sum is sum_r (U T)_ijr (V T_inv)_jkr W_kir
        # = sum_r (sum_s U_ijs T_sr) (sum_t V_jkt (T_inv)_tr) W_kir
        # = sum_st U_ijs V_jkt W_kir (sum_r T_sr (T_inv)_tr)
        # = sum_st U_ijs V_jkt W_kir delta_st
        # = sum_s U_ijs V_jks W_kis
        # This is correct.
        
        U_new = U_curr @ T
        V_new = V_curr @ T_inv
        W_new = W_curr
        
        # Find columns that are effectively zero in any of the matrices
        # Tolerance for "zero"
        tol = 1e-9
        
        zero_cols = []
        for r in range(current_rank):
            if np.all(np.abs(U_new[:, r]) < tol) or \
               np.all(np.abs(V_new[:, r]) < tol) or \
               np.all(np.abs(W_new[:, r]) < tol):
                zero_cols.append(r)
                
        if not zero_cols:
            # If no columns are zero, we might still try to keep the transformed matrices
            # if they are "sparser" or just randomize again.
            # For now, just continue.
            continue
            
        # Remove the zero columns
        keep_cols = [r for r in range(current_rank) if r not in zero_cols]
        
        U_curr = U_new[:, keep_cols]
        V_curr = V_new[:, keep_cols]
        W_curr = W_new[:, keep_cols]
        
        current_rank = len(keep_cols)
        
        # Update best if improved
        if current_rank < best_rank:
            best_rank = current_rank
            best_result = {
                "U": U_curr.tolist(),
                "V": V_curr.tolist(),
                "W": W_curr.tolist()
            }
            
    return best_result
# EVOLVE-BLOCK-END
