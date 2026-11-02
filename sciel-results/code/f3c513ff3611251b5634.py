"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    def verify_np(U_np, V_np, W_np, tol=1e-9):
        """Verify tensor equation: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook solution
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # The tensor equation is: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    # This is a system of n*m*p = 112 polynomial equations (degree 3) in 72*r unknowns
    
    # Strategy: try ranks from a lower bound upward, using random initialization + local search
    # Lower bound: at least max(n, m, p) = 7, but realistically much higher
    
    # Known: 4x4x4 has rank 23, so 4x4x7 should be at most 23 + 3*4 = 35 (naive extension)
    # Let's try to find good decompositions
    
    random.seed(42)
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in range(20, min(112, 60)):
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            break
        
        # Multiple random restarts for this rank
        found = False
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_r = np.random.randn(n * m, target_rank) * 0.5
            V_r = np.random.randn(m * p, target_rank) * 0.5
            W_r = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple local search: gradient-free, perturb and accept if error decreases
            def compute_error(Ur, Vr, Wr):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(Ur[i*m+j] * Vr[j*p+k] * Wr[k*n+i])
                            err += (s - 1.0) ** 2
                return err
            
            current_err = compute_error(U_r, V_r, W_r)
            
            # Local search iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                if current_err < 1e-12:
                    break
                
                # Perturb a random element
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    row = random.randint(0, n*m-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, 0.1)
                    new_val = U_r[row, colr] + delta
                    U_r[row, colr] = new_val
                    new_err = compute_error(U_r, V_r, W_r)
                    if new_err > current_err:
                        U_r[row, colr] -= delta
                    else:
                        current_err = new_err
                elif mat_choice == 1:
                    row = random.randint(0, m*p-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, 0.1)
                    new_val = V_r[row, colr] + delta
                    V_r[row, colr] = new_val
                    new_err = compute_error(U_r, V_r, W_r)
                    if new_err > current_err:
                        V_r[row, colr] -= delta
                    else:
                        current_err = new_err
                else:
                    row = random.randint(0, p*n-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, 0.1)
                    new_val = W_r[row, colr] + delta
                    W_r[row, colr] = new_val
                    new_err = compute_error(U_r, V_r, W_r)
                    if new_err > current_err:
                        W_r[row, colr] -= delta
                    else:
                        current_err = new_err
            
            if current_err < 1e-10:
                if verify_np(U_r, V_r, W_r, tol=1e-9):
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    # Convert to list format
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
