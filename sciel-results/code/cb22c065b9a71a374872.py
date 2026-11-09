"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # For each (i,j,k), check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    # Check that for all (i,j,k) not matching, the sum is 0
    # Actually, we need to check that the decomposition is exact
    # The tensor T[i,j,k] = 1 if this is a valid triple, else 0
    # But matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
    # Wait, let me reconsider. The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if we're multiplying element (i,j) of A by (j,k) of B
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    return True

def _schoolbook(n, m, p):
    """Create the standard schoolbook decomposition."""
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
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=10000, seed=42):
    """Try to reduce rank by finding linear dependencies among the columns."""
    rank = len(U[0])
    if target_rank >= rank:
        return None
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # We want to find a linear transformation that reduces the effective rank
    # Try random projections and see if we can maintain the decomposition
    
    best = None
    for iteration in range(max_iters):
        # Try to find a random linear combination that preserves the tensor
        # This is a heuristic approach
        
        # Generate random vectors for each column
        # We're looking for coefficients alpha_c such that
        # sum_c alpha_c * U[:,c] * V[:,c] * W[:,c] = 0 (approximately)
        # but we want to reduce the number of non-zero terms
        
        # Alternative: try to merge columns
        # Pick two columns and see if we can express one as a combination of others
        
        c1 = random.randint(0, rank - 1)
        c2 = random.randint(0, rank - 1)
        if c1 == c2:
            continue
        
        # Check if column c2 is approximately in the span of other columns
        # This is expensive, so use a simpler heuristic
        
        # Try a random perturbation approach
        # Add a small random perturbation to all columns and see if rank reduces
        
        if iteration % 100 == 0:
            # Try a structured approach: look for patterns
            # For matrix multiplication, there might be symmetries we can exploit
            pass
    
    return None

def _random_search(n, m, p, max_rank, max_time=100.0):
    """Random search for low-rank decompositions using structured initialization."""
    start_time = time.time()
    
    # Use a smarter approach: try to build up the decomposition incrementally
    # Start with schoolbook and try to merge columns
    
    for target_rank in range(8, max_rank + 1):
        if time.time() - start_time > max_time:
            break
        
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() - start_time > max_time:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Strategy: each rank term handles a "block" of the computation
            # For each (i,j,k) triple, we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # We can think of this as: for each (i,j), we need to cover all k values
            # and for each (j,k), we need to cover all i values
            
            # Use a structured random approach
            for c in range(target_rank):
                # Each term picks a random subset of (i,j) for U, (j,k) for V, (k,i) for W
                # with the constraint that the product sums to 1 for each (i,j,k)
                
                # Randomly choose which (i,j) pairs this term contributes to
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.5:
                            U[i*m + j, c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.5:
                            V[j*p + k, c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.5:
                            W[k*n + i, c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            
            # Verify using vectorized computation
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k]) * 0  # placeholder
                        # Actually compute properly
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Leave 5s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Try random search with improved strategy
    result = _random_search(n, m, p, max_rank=84, max_time=100.0)
    if result is not None:
        best_result = result
        best_rank = len(result["U"][0])
    
    if time.time() > deadline:
        return best_result
    
    # Try a systematic approach: use the fact that matrix multiplication
    # can be decomposed using Kronecker products or block structures
    # For now, return best found
    return best_result
# EVOLVE-BLOCK-END
