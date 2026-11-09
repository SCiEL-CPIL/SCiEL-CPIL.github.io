"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor from U, V, W
    # T[(i,j),(j',k),(k',i')] = sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i,c]
    # Should equal delta(i,i') * delta(j,j') * delta(k,k')
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(actual - expected) > 1e-9:
                    return False
                # Off-diagonal should be 0
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 != i or j2 != j or k2 != k:
                                val = 0.0
                                for c in range(rank):
                                    val += U[i2*m+j2][c] * V[j2*p+k2][c] * W[k2*n+i2][c]
                                if abs(val) > 1e-9:
                                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank using block/Strassen ideas
    # For 2x3 x 3x4, try rank 22, 21, ...
    for target_rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 30:
            break
        # Try a constructive approach: perturb schoolbook and project
        # Use random initialization + ALS
        np.random.seed(42 + target_rank)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 30:
                break
            # Solve for U
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, :] should satisfy: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta(i,i')*delta(j,j')*delta(k,k')
                    # For fixed (i,j), the equation is: sum_{k} V[j*p+k,c] * W[k*n+i,c] * U[i*m+j,c] = delta(j,j')*delta(k,k') for all (j',k)
                    # This is a linear system in U[i*m+j,:]
                    # Let A[c, (j',k)] = V[j*p+k,c] * W[k*n+i,c]
                    # Then A @ U[i*m+j,:] should equal b where b[j',k] = delta(j,j')*delta(k,k')
                    A = np.zeros((target_rank, m * p))
                    for j2 in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                A[c, j2 * p + k] = V[j2 * p + k, c] * W[k * n + i, c]
                    b = np.zeros(m * p)
                    for k in range(p):
                        b[j * p + k] = 1.0
                    # Solve A^T A U = A^T b (least squares)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((target_rank, n * m))
                    for i in range(n):
                        for j2 in range(m):
                            for c in range(target_rank):
                                A[c, i * m + j2] = U[i * m + j2, c] * W[k * n + i, c]
                    b = np.zeros(n * m)
                    for i in range(n):
                        b[i * m + j] = 1.0
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Solve for W
            for i in range(n):
                for k in range(p):
                    A = np.zeros((target_rank, m * p))
                    for j in range(m):
                        for j2 in range(m):
                            for c in range(target_rank):
                                A[c, j2 * p + k] = U[i * m + j2, c] * V[j2 * p + k, c]
                    b = np.zeros(m * p)
                    for j in range(m):
                        b[j * p + k] = 1.0
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
        
        # Check if we found an exact solution
        # Convert to fractions for exact check
        U_frac = [[Fraction(x).limit_denominator(1000000) for x in row] for row in U]
        V_frac = [[Fraction(x).limit_denominator(1000000) for x in row] for row in V]
        W_frac = [[Fraction(x).limit_denominator(1000000) for x in row] for row in W]
        
        if _verify(U_frac, V_frac, W_frac, n, m, p):
            best = {"U": U_frac, "V": V_frac, "W": W_frac}
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
