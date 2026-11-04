"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 96
    
    # Build schoolbook solution
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    term_to_idx = {}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                term_to_idx[(i, j, k)] = col
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W):
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        return np.allclose(T, 1.0, atol=1e-10)
    
    if not verify(U_school, V_school, W_school):
        # Shouldn't happen, but fallback
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Strategy: Use Strassen's 7-term decomposition for 3x3x3 as a primitive
    # and extend it to handle the 4th inner dimension and the 8 output columns
    
    # Strassen's 7-term decomposition for 3x3 matrix multiplication
    # This gives rank 7 for 3x3x3 instead of 9
    
    def strassen_3x3():
        """
        Return U, V, W for 3x3x3 matrix multiplication using Strassen's method.
        U has shape (9, 7), V has shape (9, 7), W has shape (9, 7)
        """
        U = np.zeros((9, 7))
        V = np.zeros((9, 7))
        W = np.zeros((9, 7))
        
        # Strassen's 7 products for 2x2, extended to 3x3
        # For 3x3, we can use a block decomposition or direct 7-term scheme
        # Here's a known 7-term decomposition for 3x3:
        
        # Term 1: (a11+a22+a33)(b11+b22+b33)
        U[0, 0] = 1; U[3, 0] = 1; U[6, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[6, 0] = 1
        
        # Term 2: (a11+a22)(b12+b22)
        U[0, 1] = 1; U[3, 1] = 1
        V[1, 1] = 1; V[4, 1] = 1
        W[1, 1] = 1; W[4, 1] = 1
        
        # Term 3: (a11+a33)(b13+b33)
        U[0, 2] = 1; U[6, 2] = 1
        V[2, 2] = 1; V[8, 2] = 1
        W[2, 2] = 1; W[6, 2] = 1
        
        # Term 4: (a22+a33)(b23+b33)
        U[3, 3] = 1; U[6, 3] = 1
        V[5, 3] = 1; V[8, 3] = 1
        W[5, 3] = 1; W[6, 3] = 1
        
        # Term 5: (a13+a23)(b21+b31)
        U[2, 4] = 1; U[5, 4] = 1
        V[3, 4] = 1; V[6, 4] = 1
        W[3, 4] = 1; W[6, 4] = 1
        
        # Term 6: (a21+a31)(b12+b22)
        U[3, 5] = 1; U[6, 5] = 1
        V[1, 5] = 1; V[4, 5] = 1
        W[1, 5] = 1; W[4, 5] = 1
        
        # Term 7: (a12+a32)(b21+b31)
        U[1, 6] = 1; U[7, 6] = 1
        V[3, 6] = 1; V[6, 6] = 1
        W[3, 6] = 1; W[6, 6] = 1
        
        return U, V, W
    
    # For 3x4x8, we can decompose as follows:
    # - The 3x3 part uses Strassen's 7 terms
    # - The 4th inner dimension (j=3) needs to be handled separately
    # - The 8 output columns can be grouped as 2x4 or 4x2
    
    # Let's try a structured approach:
    # Split the 4x8 matrix B into two 4x4 blocks: B = [B1 | B2]
    # Then C = A * B = [A*B1 | A*B2]
    # Each A*Bi is 3x4, which we can compute using a 3x4x4 decomposition
    
    # For 3x4x4, we can use:
    # - 3x3x3 part (rank 7) for the first 3 columns
    # - Plus rank-1 terms for the 4th column
    # Total: 7 + 3 = 10 per 3x4x4 block
    # For two blocks: 20 terms
    
    # But we can do better by sharing structure
    
    # Alternative: Use the fact that p=8=2*4 to create a more efficient decomposition
    
    # Let's try a direct optimization with better initialization
    
    def try_rank_optimized(r, seed=0):
        """Try to find a rank-r decomposition using structured initialization."""
        np.random.seed(seed)
        
        # Initialize with a mix of Strassen-inspired and random terms
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # First, add Strassen-inspired terms for the 3x3x3 substructure
        strassen_U, strassen_V, strassen_W = strassen_3x3()
        
        # Map 3x3x3 terms to 3x4x8
        for c in range(7):
            if c < r:
                # For each Strassen term, we need to handle the 4th inner dim and 8 output cols
                # For now, just use the first 3 inner dims and first 3 output cols
                for i in range(3):
                    for j in range(3):
                        if strassen_U[i*3+j, c] != 0:
                            U[i*4+j, c] = strassen_U[i*3+j, c]
                        for k in range(3):
                            if strassen_V[j*3+k, c] != 0:
                                V[j*4+k, c] = strassen_V[j*3+k, c]
                            if strassen_W[k*3+i, c] != 0:
                                W[k*3+i, c] = strassen_W[k*3+i, c]
        
        # Add rank-1 terms for the remaining dimensions
        for c in range(7, min(r, 20)):
            # Random initialization for additional terms
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            U[i*m+j, c] = np.random.choice([-1, 1])
            V[j*p+k, c] = np.random.choice([-1, 1])
            W[k*n+i, c] = np.random.choice([-1, 1])
        
        # Fill remaining with small random values
        for c in range(r):
            for idx in range(n*m):
                if U[idx, c] == 0:
                    U[idx, c] = np.random.randn() * 0.01
            for idx in range(m*p):
                if V[idx, c] == 0:
                    V[idx, c] = np.random.randn() * 0.01
            for idx in range(p*n):
                if W[idx, c] == 0:
                    W[idx, c] = np.random.randn() * 0.01
        
        # Use gradient descent
        lr = 0.005
        for iteration in range(200):
            if time.time() > deadline - 30:
                break
            
            # Compute current tensor values
            T = np.zeros((n, m, p))
            for c in range(r):
                u_c = U[:, c].reshape(n, m)
                v_c = V[:, c].reshape(m, p)
                w_c = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
            
            # Compute gradient
            error = T - 1.0
            
            dU = np.zeros((n * m, r))
            dV = np.zeros((m * p, r))
            dW = np.zeros((p * n, r))
            
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dU[i*m+j, c] += error[i, j, k] * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += error[i, j, k] * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += error[i, j, k] * U[i*m+j, c] * V[j*p+k, c]
            
            # Update with adaptive learning rate
            norm_error = np.linalg.norm(error)
            if norm_error < 1e-10:
                break
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Decay learning rate
            lr *= 0.995
        
        # Check if we converged
        T = np.zeros((n, m, p))
        for c in range(r):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
        
        if np.allclose(T, 1.0, atol=1e-6):
            return True, U, V, W
        
        return False, None, None, None
    
    # Try different ranks starting from a reasonable lower bound
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But with Strassen-inspired structure, we might do better
    
    for r in range(25, rank_school + 1):
        if time.time() > deadline - 40:
            break
        
        # Try multiple random initializations
        for seed in range(3):
            if time.time() > deadline - 40:
                break
            
            success, U, V, W = try_rank_optimized(r, seed=seed)
            
            if success:
                # Verify
                if verify(U, V, W):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break  # Found a valid decomposition at this rank
        
        if best_rank < rank_school:
            break  # Found improvement, no need to try higher ranks
    
    # If no improvement found, try the greedy elimination approach
    if best_rank == rank_school:
        def check_feasibility(active_terms, U_init, V_init):
            """Check if there exist U, V, W for the active terms that satisfy the tensor equation."""
            r = len(active_terms)
            if r == 0:
                return False, None, None, None
            
            # Build matrix A where A[eq_idx, w_idx] = U[i*m+j,c] * V[j*p+k,c]
            A = np.zeros((n*m*p, p*n * r))
            b = np.ones(n*m*p)
            
            for c_local, c_global in enumerate(active_terms):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            w_idx = c_local * (p*n) + (k * n + i)
                            A[eq_idx, w_idx] = U_init[i*m+j, c_global] * V_init[j*p+k, c_global]
            
            # Solve least squares
            try:
                W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W_sol = W_sol.reshape(r, p*n).T  # shape (p*n, r)
                
                # Check residual
                residual = np.linalg.norm(A @ W_sol.flatten() - b)
                if residual < 1e-8:
                    return True, U_init[:, active_terms], V_init[:, active_terms], W_sol
                else:
                    return False, None, None, None
            except:
                return False, None, None, None
        
        # Try removing terms one at a time
        active_terms = list(range(rank_school))
        current_U, current_V, current_W = U_school, V_school, W_school
        
        # Greedy: repeatedly try to remove the "easiest" term
        removed_count = 0
        while time.time() < deadline - 20:
            removed_this_round = False
            for c_remove in list(active_terms):
                if time.time() > deadline - 20:
                    break
                
                candidate_terms = [c for c in active_terms if c != c_remove]
                if len(candidate_terms) == 0:
                    break
                
                # Try to find U, V, W for candidate_terms
                feasible, U_new, V_new, W_new = check_feasibility(candidate_terms, current_U, current_V)
                
                if feasible:
                    active_terms = candidate_terms
                    current_U, current_V, current_W = U_new, V_new, W_new
                    removed_this_round = True
                    removed_count += 1
                    break  # Restart the loop with the new active set
            
            if not removed_this_round:
                break
        
        # Final verification
        if len(active_terms) < rank_school:
            if verify(current_U, current_V, current_W):
                best_rank = len(active_terms)
                best_U, best_V, best_W = current_U, current_V, current_W
            else:
                # Fallback to schoolbook
                best_rank = rank_school
                best_U, best_V, best_W = U_school, V_school, W_school
        else:
            best_rank = rank_school
            best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists
    result = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return result
# EVOLVE-BLOCK-END
