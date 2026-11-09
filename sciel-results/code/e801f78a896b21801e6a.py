"""Matrix multiplication 3x5x7 rank-minimal decomposition.

The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,7).
Rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.

Strategy:
  1. Schoolbook (rank 105) as fallback.
  2. Randomized search with iterative refinement for lower ranks.
  3. Try structured decompositions exploiting dimension structure.
"""

import time
import random
import sys
import numpy as np

DIMENSIONS = [3, 5, 7]


def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_random_rank(r, n, m, p, max_iters=2000, seed=None, deadline=None):
    """Try to find rank-r decomposition via random init + gradient descent on
    the constraint sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = np_rng.randn(n*m, r)
    V = np_rng.randn(m*p, r)
    W = np_rng.randn(p*n, r)
    
    # Scale to make initial products closer to 1
    # Average product per term is ~1/r, so scale each by r^(1/3)
    scale = r ** (1.0/3.0)
    U *= scale
    V *= scale
    W *= scale
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute loss: sum over all (i,j,k) of (sum_c prod - 1)^2
        # and compute gradients
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    prod = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err = prod - 1.0
                    loss += err * err
                    
                    # d/dU[i*m+j,c] = 2*err * V[j*p+k,c] * W[k*n+i,c]
                    dU[i*m+j] += 2 * err * V[j*p+k] * W[k*n+i]
                    dV[j*p+k] += 2 * err * U[i*m+j] * W[k*n+i]
                    dW[k*n+i] += 2 * err * U[i*m+j] * V[j*p+k]
        
        # Gradient step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
        
        # Check convergence
        if best_loss < 1e-8:
            break
    
    if best_loss < 1e-6:
        # Convert to list format with rounding to half-integers if close
        U_list = [[round(float(U[i][c]), 6) for c in range(r)] for i in range(n*m)]
        V_list = [[round(float(V[i][c]), 6) for c in range(r)] for i in range(m*p)]
        W_list = [[round(float(W[i][c]), 6) for c in range(r)] for i in range(p*n)]
        
        # Try to snap to half-integers
        for mat in [U_list, V_list, W_list]:
            for row in mat:
                for idx in range(len(row)):
                    row[idx] = round(row[idx] * 2) / 2.0
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-4):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_structured_rank(r, n, m, p, deadline=None):
    """Try structured decomposition where U, V, W have block structure."""
    # Try: U[i*m+j, c] = a[i,c]*b[j,c], V[j*p+k,c] = b[j,c]*d[k,c], W[k*n+i,c] = d[k,c]*a[i,c]
    # Then product = a[i,c]*b[j,c]*b[j,c]*d[k,c]*d[k,c]*a[i,c] = a[i,c]^2 * b[j,c]^2 * d[k,c]^2
    # This won't give 1 in general. Skip this.
    
    # Try simpler: see if we can factor using the specific dimensions
    # For 3x5x7, try rank = 3*5+5*7+3*7 - something (Strassen-like)
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 105
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try ranks from just below schoolbook down to reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But realistically for matrix mult, rank >= n*m*p / (something)
    # Try a range of ranks
    
    # First, try to improve from schoolbook by small amounts
    # Try ranks: 104, 103, 102, ... down to maybe 80
    # But this is very hard. Let's try a smarter approach.
    
    # Approach: try random rank-r for various r
    # Start with r close to schoolbook and work down
    
    # For efficiency, try a few ranks
    ranks_to_try = []
    # Try some key ranks
    for r in range(schoolbook_rank - 1, max(35, schoolbook_rank - 30), -1):
        ranks_to_try.append(r)
    
    # Also try some lower ranks with more iterations
    for r in [80, 75, 70, 65, 60, 55, 50]:
        if r not in ranks_to_try and r > 35:
            ranks_to_try.append(r)
    
    # Sort by rank (try higher ranks first as they're easier)
    ranks_to_try = sorted(set(ranks_to_try), reverse=True)
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds for this rank
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = _try_random_rank(r, n, m, p, max_iters=500, seed=seed, deadline=deadline - 10)
            if result is not None:
                best = result
                best_rank = r
                found = True
                break
        
        if found:
            # Found a valid decomposition at rank r, continue trying lower
            continue
    
    # If we found something better, try to polish
    if best_rank < schoolbook_rank and time.time() < deadline - 20:
        # Try to reduce rank further with the found solution as starting point
        # (This is complex, skip for now)
        pass
    
    return best


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
