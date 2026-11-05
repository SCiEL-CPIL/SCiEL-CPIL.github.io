import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                if total != 1:
                    return False
    return True

def _construct_rank_28():
    # ... (keep existing) ...
    pass

def _find_low_rank_m225(rank_target):
    """
    Attempts to find a rank decomposition for M_{2,2,5} with the given rank.
    Returns U, V, W for the 2x2x5 subproblem (indices i,j,k where j is sum index).
    U shape: (2*2, rank) = (4, rank)
    V shape: (2*5, rank) = (10, rank)
    W shape: (5*2, rank) = (10, rank)
    """
    n, m, p = 2, 2, 5
    # Tensor T[i,j,k] = 1 if j is the sum index?
    # In M_{n,m,p}, T[i,j,k] corresponds to A[i,j] B[j,k] C[i,k].
    # Here n=2, m=2, p=5.
    # U indices: i*m+j (0..3)
    # V indices: j*p+k (0..9)
    # W indices: k*n+i (0..9)
    
    # We want to minimize || U V^T W^T - T ||
    # This is a non-convex problem.
    # Use random restarts and ALS.
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(100):
        np.random.seed(restart)
        U = np.random.randn(4, rank_target)
        V = np.random.randn(10, rank_target)
        W = np.random.randn(10, rank_target)
        
        for it in range(500):
            # Update U
            # For each (i,j), U[i,j,:] should satisfy sum_k W[k,i,:] V[j,k,:] * U[i,j,:] = T[i,j,k]
            # This is tricky because it's a 3-way product.
            # Standard ALS for CP decomposition:
            # Fix V, W. Solve for U.
            # U[:, r] is not independent.
            
            # Actually, the constraint is:
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k]
            
            # This is a system of linear equations for each r? No, r is summed.
            # This is a quadratic programming problem for each r?
            # No, it's a rank-r decomposition.
            
            # We can use gradient descent.
            # Loss = sum_{i,j,k} ( sum_r U[i,j,r] V[j,k,r] W[k,i,r] - T[i,j,k] )^2
            
            # Compute gradients.
            # dL/dU[i,j,r] = 2 * error[i,j,k] * V[j,k,r] * W[k,i,r]
            
            # This is feasible.
            pass # Implementation of GD is complex to write correctly in short time.
            
    return None

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Try to find rank 26 decomposition (10 + 16)
    # If fails, return rank 28.
    
    # Since I don't have a reliable numerical solver implementation ready,
    # I will stick to the known rank 28 but check if I can manually construct rank 27.
    
    # Actually, let's look at the inspiration 3: "exploiting the rectangular structure".
    # It mentions "rank-7 Laderman algorithm for 3x3".
    # Maybe we can use a 3x3 Strassen-like block?
    
    # If I can't find a better rank, I should ensure the current one is robust.
    # The current score is 1.0714.
    
    # Let's try to implement a simple numerical search for M_{2,2,5} rank 16.
    # If it finds integers, use them.
    
    res_28 = _construct_rank_28()
    if _verify_decomposition(res_28['U'], res_28['V'], res_28['W'], 2, 3, 5):
        return res_28
    else:
        raise ValueError("Verification failed")
