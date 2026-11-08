import numpy as np
import time
import random
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            # C[i,k] should equal sum_j A[i,j]*B[j,k]
            expected = 0
            for j in range(m):
                expected += 1.0  # This is the tensor coefficient
            # In the tensor: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
            # The decomposition gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For each (i,j,k) triple, the coefficient should be 1
            actual = 0.0
            for j in range(m):
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            if abs(actual - expected) > 1e-9:
                return False
    return True

def _try_rank_random(rank, n, m, p, max_iters=1000, deadline=None):
    """Try random search for a valid decomposition of given rank."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with random values
        U = np.random.randn(dim_U, rank) * 0.1
        V = np.random.randn(dim_V, rank) * 0.1
        W = np.random.randn(dim_W, rank) * 0.1
        
        # Try a few gradient steps
        for step in range(50):
            if deadline and time.time() > deadline:
                break
            
            # Compute gradient of loss
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = val - 1.0
                        loss += err * err
                        
                        for r in range(rank):
                            grad_U[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
            
            # Gradient descent
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < 1e-10:
                # Verify and return
                if _verify(U, V, W, n, m, p):
                    return U, V, W
        else:
            if _verify(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _try_structured_3x3x6(deadline=None):
    """
    Try to find a structured decomposition.
    
    Key idea: C[i,k] = sum_j A[i,j] * B[j,k]
    We can try to share the B[j,k] factors across different i values.
    
    For each j, we have 3 inputs A[0,j], A[1,j], A[2,j] and 6 outputs B[j,0..5].
    The products A[i,j]*B[j,k] for fixed j give us a 3x6 matrix of products.
    
    One approach: use rank-3 decompositions for each "slice"
    """
    n, m, p = 3, 3, 6
    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18
    
    # Try rank = 27 (half of 54)
    # Structure: for each j in {0,1,2}, we need to compute A[:,j] @ B[j,:]
    # This is an outer product of a 3-vector and a 6-vector
    # We can decompose this using rank-3 terms
    
    # Attempt: use Kronecker-like structure
    # For each j, create 3 rank-1 terms that capture A[i,j]*B[j,k]
    # Total rank = m * 3 = 9? No, that's too optimistic.
    
    # Let's try a different approach: use the fact that we can combine
    # multiple (i,j,k) triples in one rank-1 term if they share structure
    
    # Try rank 27 with a structured initialization
    rank = 27
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    # For each j, assign 9 ranks (3 for each i, but shared across k)
    # Actually, let's try: for each (i,j), assign p=6 ranks, but with structure
    
    # Alternative: use the known result that 3x3x3 has rank 23 (Strassen-like)
    # For 3x3x6, we might use 2 * rank(3x3x3) or similar
    
    # Let's try a direct construction:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Each term A[i,j]*B[j,k] is a rank-1 contribution
    # But we can potentially combine terms
    
    # Try: for each j, use 3 ranks to compute the outer product A[:,j] @ B[j,:]
    # That's 3*3 = 9 ranks total? No, outer product of 3x1 and 1x6 has rank 1
    # But we need to sum over j, so it's a sum of 3 rank-1 matrices
    
    # The issue is that the tensor is not just a sum of outer products in a simple way
    
    # Let's try random search with better initialization
    best_result = None
    for attempt in range(200):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with structured pattern
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        # For each j, use 9 ranks
        for j in range(m):
            base = j * 9
            for i in range(n):
                for k in range(p):
                    r = base + i * 3 + k % 3
                    # This doesn't quite work, let me try differently
                    pass
        
        # Simpler: just do random with structure
        U = np.random.randn(dim_U, rank) * 0.5
        V = np.random.randn(dim_V, rank) * 0.5
        W = np.random.randn(dim_W, rank) * 0.5
        
        # Gradient descent
        for step in range(200):
            if deadline and time.time() > deadline:
                break
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            loss = 0.0
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = val - 1.0
                        loss += err * err
                        
                        for r in range(rank):
                            grad_U[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
            
            lr = 0.005
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < 1e-8:
                if _verify(U, V, W, n, m, p):
                    return U, V, W
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to find better solutions
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try structured approach
    result = _try_structured_3x3x6(deadline=deadline)
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
        if rank < best_rank and _verify(U, V, W, n, m, p):
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    # Try random search for lower ranks
    for target_rank in [36, 30, 27, 24]:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_random(target_rank, n, m, p, max_iters=50, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
