import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_decomp = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try ranks from 19 upward (known lower bound for 2x3x4 is around 19-20)
    for target_rank in [19, 20, 21, 22]:
        if time.time() > deadline - 30:
            break
            
        max_attempts = 30 if target_rank <= 20 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(12345 + attempt * 7919 + target_rank * 104729)
            
            # Initialize with small random values, allowing rationals
            scale = 1.0
            U = np.random.randint(-2, 3, size=(n * m, target_rank)) * scale
            V = np.random.randint(-2, 3, size=(m * p, target_rank)) * scale
            
            # Iteratively solve for W using Gauss-Seidel style updates
            # For each (kp, ip), we need:
            # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = delta(i==ip, j==jp, k==kp)
            
            W = np.zeros((p * n, target_rank))
            
            # Precompute for efficiency
            # For each (kp, ip), build the linear system
            # A[(i,j,jp,k), r] = U[i*m+j, r] * V[jp*p+k, r]
            # b[(i,j,jp,k)] = 1 if i==ip, j==jp, k==kp else 0
            
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 15:
                    break
                
                W_old = W.copy()
                
                for kp in range(p):
                    for ip in range(n):
                        # Build A and b for this (kp, ip)
                        A = np.zeros((n * m * m * p, target_rank))
                        b = np.zeros(n * m * m * p)
                        
                        eq_idx = 0
                        for i in range(n):
                            for j in range(m):
                                for jp in range(m):
                                    for k in range(p):
                                        coeff = U[i * m + j] * V[jp * p + k]
                                        A[eq_idx, :] = coeff
                                        if i == ip and j == jp and k == kp:
                                            b[eq_idx] = 1.0
                                        eq_idx += 1
                        
                        # Solve least squares
                        try:
                            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        except:
                            x = np.zeros(target_rank)
                        
                        W[kp * n + ip, :] = x
                
                # Check convergence
                diff = np.max(np.abs(W - W_old))
                if diff < 1e-10:
                    converged = True
                    break
            
            if not converged:
                continue
            
            # Verify the solution
            # Check if all tensor equations are satisfied
            valid = True
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    lhs = 0.0
                                    for r in range(target_rank):
                                        lhs += U[i * m + j, r] * V[jp * p + k, r] * W[kp * n + ip, r]
                                    rhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                    if abs(lhs - rhs) > 1e-8:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Valid decomposition found!
            # Convert to rational representation if possible
            # For now, use the floating point values
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if target_rank < best_rank:
                best_rank = target_rank
                best_decomp = {"U": U_list, "V": V_list, "W": W_list}
            
            # Found a valid one at this rank, we can stop
            break
        
        # If we found a valid decomposition at this rank, we can stop
        if best_rank <= target_rank:
            break
    
    return best_decomp
