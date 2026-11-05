"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify_full(U, V, W, n, m, p, tol=1e-6):
    """Full verification of all tensor entries."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s limit
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank decompositions using ALS
    for target_rank in range(85, rank_school):
        if time.time() > deadline:
            break
            
        found = False
        for trial in range(8):
            if time.time() > deadline or found:
                break
            
            np.random.seed(trial * 7919 + target_rank * 31)
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            for iteration in range(300):
                if time.time() > deadline:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        AT = A.T
                        ATA = AT @ A + 1e-12 * np.eye(target_rank)
                        rhs = AT @ np.ones(p)
                        U[i*m+j, :] = np.linalg.solve(ATA, rhs)
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        BT = B.T
                        BTA = BT @ B + 1e-12 * np.eye(target_rank)
                        rhs = BT @ np.ones(n)
                        V[j*p+k, :] = np.linalg.solve(BTA, rhs)
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        CT = C.T
                        CTA = CT @ C + 1e-12 * np.eye(target_rank)
                        rhs = CT @ np.ones(m)
                        W[k*n+i, :] = np.linalg.solve(CTA, rhs)
                
                # Check convergence every 30 iterations
                if iteration % 30 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                    if max_err < 1e-8:
                        if _verify_full(U, V, W, n, m, p, tol=1e-6):
                            best_rank = target_rank
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            found = True
                            break
                    if max_err < 1e-10:
                        break  # Converged
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
