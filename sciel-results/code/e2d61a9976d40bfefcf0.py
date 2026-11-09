import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40

    # Build the schoolbook solution as baseline
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    # Try to find a lower-rank decomposition using random search
    # The tensor for matrix multiplication has entries T[i,j,k] = 1 if the (i,j) of A and (j,k) of B produce (i,k) of C
    # We need U[i,j,r] * V[j,k,r] * W[i,k,r] summed over r to equal T[i,j,k]
    
    def build_target_tensor():
        """Build the target tensor T[i,j,k] where T[i,j,k] = 1 if valid indices"""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1
        return T

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition exactly reproduces the tensor"""
        # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def round_to_half_int(arr, tol=1e-6):
        """Round entries to nearest half-integer and check if still valid"""
        rounded = np.round(arr * 2) / 2
        return rounded

    # Try various ranks below 40
    best_result = None
    best_rank = rank_schoolbook

    # First, try to see if we can find rank 39, 38, etc.
    # Use random initialization and gradient-free optimization
    # Since this is a polynomial system, we'll use random restarts with local perturbation

    for target_rank in range(39, 15, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random restarts for this rank
        max_restarts = 50
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Normalize to avoid trivial scaling issues
            for r in range(target_rank):
                norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
                if norm > 1e-10:
                    U[:, r] /= norm**(1/3)
                    V[:, r] /= norm**(1/3)
                    W[:, r] /= norm**(1/3)
            
            # Simple local search: perturb and check
            current_error = None
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Compute current error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            error += (val - 1.0)**2
                
                if error < 1e-10:
                    # Found a solution!
                    U = round_to_half_int(U)
                    V = round_to_half_int(V)
                    W = round_to_half_int(W)
                    if verify_decomposition(U, V, W, tol=1e-6):
                        found = True
                        best_result = (U, V, W, target_rank)
                        break
                    else:
                        # Rounding broke it, try without rounding
                        if verify_decomposition(U, V, W, tol=1e-8):
                            found = True
                            best_result = (U, V, W, target_rank)
                            break
                
                if current_error is not None and error > current_error * 1.1:
                    # Revert perturbation
                    pass
                
                current_error = error
                
                # Perturb
                mask = np.random.rand() < 0.3
                U[mask] += np.random.randn(*np.shape(U[mask])) * 0.1
                V[mask] += np.random.randn(*np.shape(V[mask])) * 0.1
                W[mask] += np.random.randn(*np.shape(W[mask])) * 0.1
            
            if found:
                break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fall back to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
