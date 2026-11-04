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
    
    # Group-theoretic approach: exploit symmetry to find linear dependencies
    # The matrix multiplication tensor has symmetries from S_3 (on n-dim) and S_4 (on m-dim)
    # We can use these to identify and eliminate redundant terms
    
    # Strategy: Use the structure to find a lower-rank decomposition
    # For 3x4x8, we can exploit that p=8=2*m, suggesting some structure
    
    # First, try a smarter initialization using block structure
    # Since p=2m, we can think of the output as two m-dimensional blocks
    
    # Try to find rank using a structured approach
    # The key insight: for each output position (i,k), we need sum_j A[i,j]*B[j,k]
    # This can be decomposed using the fact that we have 3 input rows and 8 output columns
    
    # Let's try a direct optimization approach with symmetry constraints
    # We'll use the fact that the tensor is linear in each factor
    
    def try_rank(r, seed=None):
        """Try to find a rank-r decomposition using optimization."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with random factors
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Add schoolbook terms as basis if r >= some threshold
        if r >= 10:
            # Initialize with a subset of schoolbook terms
            for c in range(min(r, rank_school)):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                if i < n and j < m and k < p:
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
        
        # Use gradient descent to minimize ||T - 1||^2 where T = sum_c U_c ⊗ V_c ⊗ W_c
        # This is non-convex, so we need good initialization
        
        # Build the tensor equation: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # We'll use a simple iterative method
        
        lr = 0.01
        for iteration in range(500):
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
            
            # Gradient with respect to U
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
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if np.linalg.norm(error) < 1e-8:
                return True, U, V, W
        
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
    # But we can do better with structure
    
    for r in range(32, rank_school + 1):
        if time.time() > deadline - 40:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline - 40:
                break
            
            success, U, V, W = try_rank(r, seed=seed)
            
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
            """Check if there exist U, V, W for the active terms that satisfy the tensor equation.
            We fix U and V to U_init and V_init, then solve for W."""
            r = len(active_terms)
            if r == 0:
                return False, None, None, None
            
            # Build matrix A where A[eq_idx, w_idx] = U[i*m+j,c] * V[j*p+k,c]
            # eq_idx = i*m*p + j*p + k
            # w_idx = c * (p*n) + (k*n + i)
            
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
                # Use current U and V as initialization
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
