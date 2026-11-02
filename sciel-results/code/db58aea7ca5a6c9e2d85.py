"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    # Target: find rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Actually, the matrix multiplication tensor is T[i,j,k] = 1 (it's the standard bilinear map)
    # Wait, let me reconsider. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k, and the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k]
    # In the factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
    
    # So we need: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try rank = 8 first (which is less than 80 = 4*4*5)
    # Actually, known lower bounds and constructions:
    # For 4x4 * 4x5, the rank is at most... let's think about known results.
    # The rank of 2x2x2 is 7. For larger matrices, we can use Strassen-like methods.
    # But let's try a simple approach: use the border rank heuristic with random init and gradient descent.
    
    # Start with a small rank and see if we can find a solution
    # Known: rank(4x4x4) <= 49 (Landsberg lower bound is 48)
    # For 4x4x5, it should be around similar or slightly larger.
    
    # Let's try rank = 8 first? No, that's too small. The minimum rank for 4x4x4 is at least 48.
    # For 4x4x5, let's try rank = 50, 55, 60, etc.
    
    # Actually, let's be smarter. The schoolbook rank is 4*4*5 = 80.
    # Strassen's algorithm for 2x2x2 gives rank 7 instead of 8.
    # We can embed Strassen's method.
    
    # For now, let's try to use a random initialization with gradient descent for a moderate rank.
    # But gradient descent on a non-convex problem is tricky. Let's try a simpler approach first.
    
    # Let's try rank = 80 (schoolbook) but see if we can do better.
    # Actually, let's just return the schoolbook for now and see if the evaluator accepts it.
    # No wait, the task says "Find a rank-minimal exact tensor decomposition". The score is the rank.
    # So we need to minimize the rank.
    
    # Let me think about known results for 4x4x5 matrix multiplication rank.
    # The rank of n x n x n matrix multiplication:
    # n=2: 7 (Strassen)
    # n=3: at most 23 (Strassen), lower bound 19
    # n=4: at most 49 (Landsberg), lower bound 48
    # For 4x4x5: I'm not sure of the exact value, but it should be at most 4*4*5=80 and likely around 50-60.
    
    # Let's try to construct a solution using Strassen's method embedded.
    # Actually, let's try a random approach with rational reconstruction.
    
    # Start with rank = 50
    for rank in [50, 55, 60, 65, 70, 75, 80]:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Gradient descent on the residual
        # Residual R[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1
        # We want to minimize sum_{i,j,k} R[i,j,k]^2
        
        lr = 0.01
        for iteration in range(1000):
            # Compute residual
            R = np.zeros((n, m, p))
            for c in range(rank):
                # u_c is U[:, c] reshaped to (n, m)
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                # Outer product contribution
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            R[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            R -= 1.0
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for c in range(rank):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dU[i*m+j, c] += R[i, j, k] * v[j, k] * w[k, i]
                            dV[j*p+k, c] += R[i, j, k] * u[i, j] * w[k, i]
                            dW[k*n+i, c] += R[i, j, k] * u[i, j] * v[j, k]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if the residual is small
        R = np.zeros((n, m, p))
        for c in range(rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        R -= 1.0
        if np.max(np.abs(R)) < 1e-6:
            # Found a solution, now do rational reconstruction
            U_r = _rationalize(U)
            V_r = _rationalize(V)
            W_r = _rationalize(W)
            return {"U": U_r, "V": V_r, "W": W_r}
    
    # Fallback: return schoolbook
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

def _rationalize(matrix, max_denom=10):
    """Convert a matrix to half-integers by rounding to nearest 0.5."""
    result = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            val = matrix[i][j]
            # Snap to nearest 0.5
            snapped = round(val * 2) / 2.0
            result[i][j] = snapped
    return result
# EVOLVE-BLOCK-END
