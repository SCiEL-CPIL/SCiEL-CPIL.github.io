"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if we need the product
    # Specifically, (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i]
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # The tensor we're decomposing has T[i,j,k] = 1 for all i,j,k (since each product term
    # A[i,j]*B[j,k] contributes to (AB)[i,k])
    
    # Wait, let me re-read the convention. The output factor follows W[k*n+i, r].
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is the standard matrix multiplication tensor of size n x m x p
    # The known minimal rank for 2x3 x 3x4... let me think about what's known.
    
    # For small matrices, the minimal rank is often equal to the "group algorithm" rank.
    # For 2x2 x 2x2, it's 7 (Strassen).
    # For 2x3 x 3x4, let's try to find a good decomposition.
    
    # Strategy: Use a greedy set-cover-like approach with precomputed rank-1 tensors
    # that have small integer/half-integer entries.
    
    # First, let's define the target tensor
    target = np.ones((n, m, p), dtype=float)
    
    # Precompute a pool of candidate rank-1 tensors with entries in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    # But this is a huge space. Let's be smarter.
    
    # Actually, let's use a numerical optimization approach:
    # 1. Start with a random initialization
    # 2. Use gradient descent or alternating minimization to find a low-rank decomposition
    # 3. Round to nearest half-integers and verify
    
    # The tensor equation is:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try ranks from 12 upward (since n*m*p = 24 is the schoolbook rank)
    # Known lower bound: at least max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using alternating minimization."""
        # Random initialization
        np.random.seed(42)
        
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Alternating minimization
        for iteration in range(5000):
            if time.time() > deadline:
                return None
            
            # Fix V and W, solve for U
            # For each row u = i*m+j: sum_c U[u,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # This is a linear system for each (i,j)
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # Build the system: for each k, sum_c U[u,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # A linear system: X @ U[u,:]^T = b where X[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    X = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        X[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve X @ u_col = b
                    try:
                        u_sol, residuals, rank_sol, sv = np.linalg.lstsq(X, b, rcond=None)
                        U[u_idx, :] = u_sol
                    except:
                        pass
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    # For each i, sum_c U[i*m+j,c] * V[v_idx,c] * W[k*n+i,c] = 1
                    X = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        X[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v_sol, residuals, rank_sol, sv = np.linalg.lstsq(X, b, rcond=None)
                        V[v_idx, :] = v_sol
                    except:
                        pass
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # For each j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[w_idx,c] = 1
                    X = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        X[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w_sol, residuals, rank_sol, sv = np.linalg.lstsq(X, b, rcond=None)
                        W[w_idx, :] = w_sol
                    except:
                        pass
            
            # Check convergence
            if iteration % 100 == 0:
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        error += (val - 1) ** 2
                if error < 1e-12:
                    break
        
        # Check final error
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    error += (val - 1) ** 2
        
        if error > 1e-10:
            return None
        
        # Try to round to half-integers
        # Multiply by 2 to get integers
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        
        # Verify the rounded solution
        error2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U2[i * m + j, :] * V2[j * p + k, :] * W2[k * n + i, :])
                    error2 += (val - 1) ** 2
        
        if error2 < 1e-10:
            return U2, V2, W2
        
        # Try different rounding strategies
        # Also try integer-only rounding
        Ui = np.round(U)
        Vi = np.round(V)
        Wi = np.round(W)
        
        errori = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(Ui[i * m + j, :] * Vi[j * p + k, :] * Wi[k * n + i, :])
                    errori += (val - 1) ** 2
        
        if errori < 1e-10:
            return Ui, Vi, Wi
        
        return None
    
    # Try ranks from the lower bound upward
    # Lower bound is at least 12 (max of n*m=6, m*p=12, p*n=8)
    # Let's try 12, 13, 14, ...
    
    for rank in range(12, 25):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
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
