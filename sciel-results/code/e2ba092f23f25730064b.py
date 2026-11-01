"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try to find a decomposition with rank < 54
    # Start from schoolbook and try to reduce rank
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 54
    
    # Try ranks from 53 down to a reasonable lower bound
    # Lower bound: at least n*p = 18, but realistically much higher
    # Try a few ranks below 54
    for rank in range(best_rank - 1, max(best_rank - 10, 40), -1):
        if time.time() > deadline - 30:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found minimal, stop
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition using numerical optimization.
    
    Returns None if no valid decomposition found within time budget.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T[i,j,k] for matrix mult: T[(i,j),(j,k),(k,i)] = delta
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) corresponds to valid mult
    
    # Actually, the matrix multiplication tensor is:
    # T[a,b,c] where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # T[a,b,c] = 1 if the j-index in a matches the j-index in b AND the k-index in b matches the k-index in c AND the i-index in c matches the i-index in a
    # More precisely: a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 iff (a mod m) == (b div p) and (b mod p) == (c div n) and (c mod n) == (a div m)
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We want to find U (nm x rank), V (mp x rank), W (pn x rank) such that:
    # sum_r U[:,r] outer V[:,r] outer W[:,r] = T
    
    # This is a hard non-convex problem. Let's use a randomized approach:
    # Initialize with random values and do gradient descent on the loss.
    
    np.random.seed(42)
    
    # Multiple restarts
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            return None
        
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Simple gradient descent
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient
            # Loss = 0.5 * ||sum_r U_r outer V_r outer W_r - T||^2
            # dL/dU = sum over (b,c) where T[a,b,c] != (U*V*W)[a,b,c]
            
            # Compute current reconstruction
            # R[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            R = np.einsum('ar,br,cr->abc', U, V, W)
            diff = R - T
            
            # Gradient for U: dL/dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if step % 500 == 0:
                loss = np.sum(diff**2) / 2
                if loss < 1e-6:
                    # Found a good solution, try to round to integers/half-integers
                    return _round_and_verify(U, V, W, T, n, m, p)
        
        # After optimization, check if we got close enough
        R = np.einsum('ar,br,cr->abc', U, V, W)
        loss = np.sum((R - T)**2) / 2
        if loss < 1e-6:
            return _round_and_verify(U, V, W, T, n, m, p)
    
    return None

def _round_and_verify(U, V, W, T, n, m, p):
    """Try to round U, V, W to integers or half-integers and verify exactness."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Round to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify
    R = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
    if np.allclose(R, T, atol=1e-10):
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U_r]
        V_list = [[float(x) for x in row] for row in V_r]
        W_list = [[float(x) for x in row] for row in W_r]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
