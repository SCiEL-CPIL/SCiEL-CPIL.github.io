"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: decompose into 3x3x3 core + 3x3x4 extension
    # Core: multiply 3x3 by 3x3 -> use Bini-Capovani rank 23
    # Extension: multiply 3x3 by 3x4 -> search for low rank
    # Total rank = rank_core + rank_ext
    
    # First, get a valid schoolbook decomposition as fallback
    def schoolbook_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find a low-rank decomposition for 3x3x4
    def find_low_rank_3x3x4(time_budget):
        """Search for low-rank decomposition of 3x3x4 matrix mult."""
        n, m, p = 3, 3, 4
        best_rank = n * m * p  # schoolbook = 36
        best_U, best_V, best_W = None, None, None
        
        # Start with schoolbook
        U, V, W, rank = schoolbook_decomp(n, m, p)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
        
        # Try to reduce rank by merging pairs of terms
        # Two terms c1 and c2 can be merged if there exist scalars a,b,c,d such that
        # the combined term works. This is complex, so use random perturbation approach.
        
        # Alternative: use the Bini-Capovani construction for 3x3x3 and extend
        # For now, try random search with rank reduction
        
        for attempt in range(200):
            if time.time() - start_time > time_budget:
                break
            
            # Try rank = best_rank - 1
            target_rank = best_rank - 1
            if target_rank < 12:  # theoretical lower bound for 3x3x4 is at least 12
                break
            
            # Initialize with schoolbook and randomly zero out one column
            U_try = best_U[:, :target_rank].copy()
            V_try = best_V[:, :target_rank].copy()
            W_try = best_W[:, :target_rank].copy()
            
            # Verify this is still a valid decomposition
            if _verify_decomp(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
            else:
                # Try a different approach: random initialization and optimization
                # This is expensive, so only do a few attempts
                if attempt % 10 == 0:
                    U_rand = np.random.randn(n*m, target_rank) * 0.5
                    V_rand = np.random.randn(m*p, target_rank) * 0.5
                    W_rand = np.random.randn(p*n, target_rank) * 0.5
                    
                    # Check if close to valid
                    if _is_approx_valid(U_rand, V_rand, W_rand, n, m, p, tol=1e-3):
                        # Try to polish
                        U_pol, V_pol, W_pol, valid = _polish_decomp(U_rand, V_rand, W_rand, n, m, p)
                        if valid:
                            best_U, best_V, best_W = U_pol, V_pol, W_pol
                            best_rank = target_rank
                            break
        
        return best_U, best_V, best_W, best_rank
    
    def _verify_decomp(U, V, W, n, m, p):
        """Verify that U, V, W form an exact decomposition."""
        # For each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
        # But this is O(n*m*p*rank) which is fine for small sizes
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def _is_approx_valid(U, V, W, n, m, p, tol=1e-3):
        """Check if decomposition is approximately valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def _polish_decomp(U, V, W, n, m, p, max_iter=1000):
        """Try to polish an approximate decomposition to exact."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for it in range(max_iter):
            # Compute error for each (i,j,k)
            max_err = 0
            worst_ijk = None
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                        err = abs(val - 1)
                        if err > max_err:
                            max_err = err
                            worst_ijk = (i, j, k)
            
            if max_err < 1e-12:
                return U, V, W, True
            
            if max_err > 0.1:
                return U, V, W, False
            
            # Gradient step to reduce error
            i, j, k = worst_ijk
            current_val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
            error = 1 - current_val
            
            # Update each factor
            lr = 0.01
            for c in range(rank):
                v_val = V[j*p + k, c]
                w_val = W[k*n + i, c]
                u_grad = v_val * w_val
                v_grad = U[i*m + j, c] * w_val
                w_grad = U[i*m + j, c] * v_val
                
                U[i*m + j, c] += lr * error * u_grad
                V[j*p + k, c] += lr * error * v_grad
                W[k*n + i, c] += lr * error * w_grad
        
        # Final check
        if _verify_decomp(U, V, W, n, m, p):
            return U, V, W, True
        return U, V, W, False
    
    # Get core 3x3x3 decomposition (use schoolbook for simplicity, rank 27)
    # In a more sophisticated approach, use Bini-Capovani rank 23
    U_core, V_core, W_core, rank_core = schoolbook_decomp(3, 3, 3)
    
    # Get extension 3x3x4 decomposition
    U_ext, V_ext, W_ext, rank_ext = find_low_rank_3x3x4(60.0)
    
    total_rank = rank_core + rank_ext
    
    # Combine into full 3x3x7 decomposition
    # Core handles columns 0-2 of B, extension handles columns 3-6
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # Core part: first rank_core columns
    U_full[:, :rank_core] = U_core
    # For core, B has only first 3 columns, so V_core has shape (3*3, rank_core)
    # V_full[j*p + k, c] for k in 0..2
    for j in range(m):
        for k in range(3):
            V_full[j*p + k, :rank_core] = V_core[j*3 + k, :]
    # W_full[k*n + i, c] for k in 0..2
    for k in range(3):
        for i in range(n):
            W_full[k*n + i, :rank_core] = W_core[k*3 + i, :]
    
    # Extension part: next rank_ext columns
    U_full[:, rank_core:rank_core+rank_ext] = U_ext
    # For extension, B has columns 3-6, so V_ext has shape (3*4, rank_ext)
    # V_ext[j*4 + k_ext, c] where k_ext in 0..3 corresponds to k in 3..6
    for j in range(m):
        for k_ext in range(4):
            V_full[j*p + (3 + k_ext), rank_core:rank_core+rank_ext] = V_ext[j*4 + k_ext, :]
    # W_full[k*n + i, c] for k in 3..6
    for k_ext in range(4):
        for i in range(n):
            W_full[(3 + k_ext)*n + i, rank_core:rank_core+rank_core+rank_ext] = W_ext[k_ext*3 + i, :]
    
    # Verify the full decomposition
    if not _verify_decomp(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
        U_full, V_full, W_full, total_rank = schoolbook_decomp(n, m, p)
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
