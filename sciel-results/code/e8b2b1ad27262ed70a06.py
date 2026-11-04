"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 96
    
    # Build schoolbook solution
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
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
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Greedy elimination with feasibility checking
    def check_feasibility(active_terms, U_init, V_init):
        """Check if there exist U, V, W for the active terms that satisfy the tensor equation.
        We fix U and V to U_init and V_init, then solve for W."""
        r = len(active_terms)
        if r == 0:
            return False, None, None, None
        
        A = np.zeros((n*m*p, p*n * r))
        b = np.ones(n*m*p)
        
        for c_local, c_global in enumerate(active_terms):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        w_idx = c_local * (p*n) + (k * n + i)
                        A[eq_idx, w_idx] = U_init[i*m+j, c_global] * V_init[j*p+k, c_global]
        
        try:
            W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W_sol = W_sol.reshape(r, p*n).T
            
            residual = np.linalg.norm(A @ W_sol.flatten() - b)
            if residual < 1e-8:
                return True, U_init[:, active_terms], V_init[:, active_terms], W_sol
            else:
                return False, None, None, None
        except:
            return False, None, None, None
    
    # Try removing terms one at a time using greedy elimination
    active_terms = list(range(rank_school))
    current_U, current_V, current_W = U_school, V_school, W_school
    
    removed_count = 0
    while time.time() < deadline - 20:
        removed_this_round = False
        for c_remove in list(active_terms):
            if time.time() > deadline - 20:
                break
            
            candidate_terms = [c for c in active_terms if c != c_remove]
            if len(candidate_terms) == 0:
                break
            
            feasible, U_new, V_new, W_new = check_feasibility(candidate_terms, current_U, current_V)
            
            if feasible:
                active_terms = candidate_terms
                current_U, current_V, current_W = U_new, V_new, W_new
                removed_this_round = True
                removed_count += 1
                break
        
        if not removed_this_round:
            break
    
    # Final verification
    if len(active_terms) < rank_school:
        if verify(current_U, current_V, current_W):
            best_rank = len(active_terms)
            best_U, best_V, best_W = current_U, current_V, current_W
        else:
            best_rank = rank_school
            best_U, best_V, best_W = U_school, V_school, W_school
    else:
        best_rank = rank_school
        best_U, best_V, best_W = U_school, V_school, W_school
    
    result = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return result
# EVOLVE-BLOCK-END
