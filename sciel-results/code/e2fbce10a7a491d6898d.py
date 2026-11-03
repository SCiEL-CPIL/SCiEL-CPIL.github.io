"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

def _is_half_integer(x, tol=1e-9):
    """Check if x is a half-integer (k/2 for integer k)."""
    return abs(x * 2 - round(x * 2)) < tol

def _snap_to_half_integer(x):
    """Snap x to nearest half-integer."""
    return round(x * 2) / 2.0

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each rank index c: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    if (i,j,k) corresponds to a valid multiplication term, else 0.
    Actually the tensor T[i,j,k] = 1 for all i,j,k in the 2x3x4 case? No.
    
    The matrix multiplication tensor for (n x m) * (m x p) -> (n x p):
    T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    So T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p).
    
    The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
    for all i,j,k.
    """
    rank = len(U[0])
    # Build the tensor
    T = np.ones((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_target = n * m * p  # 24
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # First, return the schoolbook decomposition as fallback
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: For 2x3x4, the minimum rank is known to be 12 (or maybe lower?)
    # Let's try ranks from 12 up to 23
    
    # Use random initialization and iterative refinement
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        # Multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 3.0:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize
            for c in range(target_rank):
                norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                if norm > 1e-12:
                    U[:, c] /= norm**(1/3)
                    V[:, c] /= norm**(1/3)
                    W[:, c] /= norm**(1/3)
            
            # Iterative refinement: minimize sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            # Use simple gradient-free approach: alternating minimization
            converged = False
            for it in range(200):
                if time.time() > deadline - 2.0:
                    break
                
                # Update U: for each (i,j), minimize over U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        # We want sum_c U[idx, c] * (sum_k V[j*p+k, c] * W[k*n+i, c]) = 1
                        # Let A[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                        A = np.zeros(target_rank)
                        for k in range(p):
                            A += V[j*p+k, :] * W[k*n+i, :]
                        # Minimize (sum_c U[idx,c]*A[c] - 1)^2
                        # Optimal: U[idx, :] = A / (A·A) if A·A > 0
                        dot = np.dot(A, A)
                        if dot > 1e-12:
                            U[idx, :] = A / dot
                        else:
                            U[idx, :] = 0
                
                # Update V: for each (j,k), minimize over V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        # We want sum_c V[idx, c] * (sum_i U[i*m+j, c] * W[k*n+i, c]) = 1
                        A = np.zeros(target_rank)
                        for i in range(n):
                            A += U[i*m+j, :] * W[k*n+i, :]
                        dot = np.dot(A, A)
                        if dot > 1e-12:
                            V[idx, :] = A / dot
                        else:
                            V[idx, :] = 0
                
                # Update W: for each (k,i), minimize over W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        # We want sum_c W[idx, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
                        A = np.zeros(target_rank)
                        for j in range(m):
                            A += U[i*m+j, :] * V[j*p+k, :]
                        dot = np.dot(A, A)
                        if dot > 1e-12:
                            W[idx, :] = A / dot
                        else:
                            W[idx, :] = 0
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            error += (s - 1.0) ** 2
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Check if entries are close to half-integers
                is_valid = True
                for i in range(n * m):
                    for c in range(target_rank):
                        if not _is_half_integer(U[i, c], 0.1):
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    for i in range(m * p):
                        for c in range(target_rank):
                            if not _is_half_integer(V[i, c], 0.1):
                                is_valid = False
                                break
                        if not is_valid:
                            break
                if is_valid:
                    for i in range(p * n):
                        for c in range(target_rank):
                            if not _is_half_integer(W[i, c], 0.1):
                                is_valid = False
                                break
                        if not is_valid:
                            break
                
                if is_valid:
                    # Snap to half-integers and validate
                    U_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in U])
                    V_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in V])
                    W_snap = np.array([[_snap_to_half_integer(x) for x in row] for row in W])
                    
                    if _validate(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                        best_rank = target_rank
                        best_result = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        # We can stop since we found a valid decomposition at this rank
                        return best_result
    
    return best_result

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
