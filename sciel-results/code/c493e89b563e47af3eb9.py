"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses randomized linear algebra to find a low-rank decomposition, then verifies exactness.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Known lower bound for matrix multiplication rank: max(n*m, m*p, p*n)
    # For 2x4 * 4x5: max(8, 20, 10) = 20
    # Upper bound: n*m*p = 40 (schoolbook)
    # We search from the lower bound upward
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if the (i,k)-th output
    # depends on A[i,j]*B[j,k]. In our indexing:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] summed over c
    # Wait, let me re-read the convention.
    
    # The convention is: W[k*n+i, r] is the coefficient for output element (i,k) at rank-r.
    # The multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: C[i,k] = sum_{j} sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Actually, for exact decomposition of the multiplication tensor:
    # T_{i,j,k} (the coefficient of A[i,j]*B[j,k] in C[i,k]) = 1 for all valid i,j,k
    # The tensor decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify with schoolbook: for each (i,j,k), one column r has U=V=W=1.
    # Then sum_r U*V*W = 1 for that (i,j,k). Correct.
    
    # Strategy: Try to find a decomposition of rank r starting from lower bounds.
    # Use random initialization and solve the linear system.
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def try_random_decomposition(rank, num_trials=100):
        """Try to find a valid decomposition of given rank using random initialization."""
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
            
            # We need: for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is a system of n*m*p equations in n*m*rank + m*p*rank + p*n*rank unknowns.
            # But it's nonlinear (trilinear). Let's try a different approach.
            
            # Alternative: Fix U and V randomly, then solve for W.
            # For each (i,k): sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Wait, that's not quite right either. Let me re-examine.
            
            # The constraint is: for each triple (i,j,k):
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # If we fix U and V, then for each (i,k), the equation involves W[k*n+i, r] for all r,
            # but it's coupled across j. Actually no - for fixed (i,k), we have m equations (one per j):
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for j=0..m-1
            #
            # This is a linear system in W[k*n+i, :] (a vector of length rank).
            # The matrix of this system is m x rank, with entry (j,r) = U[i*m+j,r] * V[j*p+k,r].
            #
            # So for each (i,k), we solve an m x rank linear system. If rank >= m, we might have
            # a solution. Let's try this.
            
            if rank < m:
                continue  # Need at least m columns to solve m equations
            
            # Fix U and V randomly
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            
            # For each (i,k), solve for W[k*n+i, :]
            W = [[0.0] * rank for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build m x rank matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = [[U[i * m + j][r] * V[j * p + k][r] for r in range(rank)] for j in range(m)]
                    b = [1.0] * m
                    
                    # Solve A * w = b where w is W[k*n+i, :]
                    # Use simple Gaussian elimination or numpy if available
                    try:
                        import numpy as np
                        A_np = np.array(A)
                        b_np = np.array(b)
                        if rank == m:
                            # Square system
                            w = np.linalg.solve(A_np, b_np)
                        else:
                            # Overdetermined: use least squares, but we need exact solution
                            # Check if b is in the column space of A
                            w, residuals, rank_A, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                            # Check residual
                            if np.linalg.norm(A_np @ w - b_np) > 1e-8:
                                valid = False
                                break
                        W[k * n + i] = [float(x) for x in w]
                    except Exception:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify
                if verify(U, V, W, rank):
                    # Convert to integer/half-integer if possible
                    U_out = [[_to_half_int(x) for x in row] for row in U]
                    V_out = [[_to_half_int(x) for x in row] for row in V]
                    W_out = [[_to_half_int(x) for x in row] for row in W]
                    # Re-verify with converted values
                    if verify(U_out, V_out, W_out, rank):
                        return {"U": U_out, "V": V_out, "W": W_out}
        
        return None
    
    def _to_half_int(x):
        """Convert to nearest half-integer if close, otherwise round to integer."""
        # Check if close to a half-integer
        half = round(x * 2) / 2.0
        if abs(x - half) < 1e-9:
            return half
        # Otherwise round to integer
        return round(x)
    
    # Search from lower bound to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But actually the true lower bound might be higher. Let's start from 20.
    
    for rank in range(max(n * m, m * p, p * n), n * m * p + 1):
        if time.time() > deadline:
            break
        
        result = try_random_decomposition(rank, num_trials=50)
        if result is not None:
            return result
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
