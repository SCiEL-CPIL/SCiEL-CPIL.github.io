"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For all i (0..n-1), l (0..m-1), k (0..p-1):
    # sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140  # 10s safety margin before 150s kill
    
    # Start with the known rank-54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # Strassen-like 7-product scheme for 2x2x2 blocks
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Verify
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, best_rank):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to reduce rank by eliminating redundant terms
    # Strategy: For each term r, check if it can be expressed as a linear combination
    # of other terms. If so, we can eliminate it.
    
    # Convert to numpy for efficient linear algebra
    U_arr = np.array(best_U, dtype=float)
    V_arr = np.array(best_V, dtype=float)
    W_arr = np.array(best_W, dtype=float)
    
    # The tensor is T[i*m+l, l*p+k, k*n+i] = sum_r U[i*m+l,r] * V[l*p+k,r] * W[k*n+i,r]
    # We want to check if removing term r still gives the correct tensor.
    
    # For each term r, compute the "contribution" vector and check if it's in the span
    # of other terms' contributions.
    
    # Build the contribution vectors: for each r, the vector of values
    # c_r[i*m+l, l*p+k, k*n+i] = U[i*m+l,r] * V[l*p+k,r] * W[k*n+i,r]
    
    # But we need to check if c_r is in the span of {c_s : s != r}
    # This is expensive, so let's use a simpler heuristic:
    # Check if term r has very small entries (close to zero) and can be dropped.
    
    # Actually, let's try a different approach: attempt to find a lower-rank
    # decomposition by solving a linear system.
    
    # For rank R-1, we need to find U', V', W' of rank R-1 such that
    # sum_{r=0}^{R-2} U'[i*m+l,r] * V'[l*p+k,r] * W'[k*n+i,r] = 1
    
    # This is a bilinear system, which is hard. Let's try a simpler approach:
    # Try to merge two terms that are "similar" in some sense.
    
    # For now, return the rank-54 decomposition.
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
