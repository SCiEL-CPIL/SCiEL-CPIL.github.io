"""Tensor decomposition for 3x4x6 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    rank_school = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
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
    
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin before 150s kill
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matmul tensor."""
        r = U.shape[1]
        # Target: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # All other entries should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        # Check non-target entries are 0
        # T[a, b, c] where a=i*m+j, b=j2*p+k2, c=k3*n+i3
        # Should be 0 unless j==j2 and k==k3 (and then i,i3 can be anything? No...)
        # Actually T[i*m+j, j'*p+k', k''*n+i''] = 1 if j==j' and k==k'', else 0
        # Wait, the matmul tensor is T[i,j,k] = delta_{j,k} in the sense that
        # (AB)[i,k] = sum_j A[i,j]B[j,k]
        # So the tensor has T[i*m+j, j*p+k, k*n+i] = 1
        # And T[i*m+j, j'*p+k', k''*n+i''] = 0 if j != j' or k != k'
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (j == j2 and k == k2) else 0.0
                    val = 0.0
                    for cc in range(r):
                        val += U[a, cc] * V[b, cc] * W[c, cc]
                    if abs(val - expected) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy operations."""
        r = U.shape[1]
        # Compute all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check a subset of non-target entries for speed
        # Check entries where j != j2
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                val = np.sum(U[i * m + j, :] * V[j2 * p + k, :] * W[k2 * n + i2, :])
                                if abs(val) > tol:
                                    return False
        return True
    
    # Try structured Strassen-like approaches for sub-blocks
    # For 3x4x6, we can try to use Strassen on 2x2 blocks
    # Then handle remaining rows/columns with schoolbook
    
    # Strategy: Try ranks from 71 down to some minimum
    # Use random initialization + local search
    
    # First, try a structured approach:
    # Split the 3x4 matrix into a 2x4 part and a 1x4 part
    # Split the 4x6 matrix into a 4x2 part and a 4x4 part (or similar)
    # Use Strassen for the 2x2 core, schoolbook for the rest
    
    # Actually, let's try a more systematic approach:
    # For each target rank, try random initialization and gradient descent
    
    # Build the full target tensor for comparison
    # T[a, b, c] where a = i*m+j, b = j'*p+k', c = k''*n+i''
    # T = 1 if j == j' and k == k'', else 0
    
    # We'll use a different strategy: try to find low-rank decompositions
    # using the fact that the tensor has a specific structure
    
    # For now, try random search with verification
    for target_rank in range(rank_school - 1, max(20, rank_school // 2), -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        max_attempts = 3
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 131)
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Check if it works (very unlikely)
            if verify_fast(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    # If no better found, return schoolbook
    if best_rank == rank_school:
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
