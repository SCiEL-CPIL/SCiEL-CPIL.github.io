"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the trivial rank-112 decomposition."""
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

def _verify(U, V, W):
    """Check that the tensor decomposition is exact."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build the multiplication tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) of B
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # And for all other combinations it should be 0
    # Actually the tensor for matrix multiplication is:
    # T[i,j,k,l] = delta_{j,k}  (A is n x m, B is m x p, C is n x p)
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # So the tensor is T[i,j,l] with index j being the contraction index
    # In the factorization: sum_r U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r] = delta_{j,?}
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So W is indexed by (k, i) where k is the column of B / row of C? 
    # Let me think: A is 4x4 (n x m), B is 4x7 (m x p), C is 4x7 (n x p)
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # U has shape (n*m, rank) -> indexed by (i, j)
    # V has shape (m*p, rank) -> indexed by (j, k)  
    # W has shape (p*n, rank) -> indexed by (k, i)
    # The tensor equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{j,?}
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # The standard tensor: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's just the identity in the contraction)
    # No wait: the multiplication tensor has T[i,j,k,l] = delta_{j,k}... 
    # Let me just check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    # And for the "wrong" indices it should be 0, but since we only have these indices
    # that's the full check.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(r):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _random_search(max_rank=112, num_trials=50000):
    """Randomly search for a low-rank exact decomposition."""
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    best = None
    best_rank = max_rank + 1
    
    # Try ranks from low to high
    for rank in range(50, max_rank + 1):
        if time.time() > _DEADLINE:
            break
        if best_rank <= rank:
            break
            
        # For each rank, try random decompositions
        # Strategy: random vectors, check if they form a valid decomposition
        # A valid decomposition requires:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Better approach: use the structure. 
        # For a rank-r decomposition, we need r terms.
        # Each term r contributes U_r[i,j] * V_r[j,k] * W_r[k,i]
        # and the sum over r must equal 1 for all (i,j,k).
        
        # Random approach: generate random U, V, W with small integer/half-integer values
        # and check.
        
        for trial in range(num_trials):
            if time.time() > _DEADLINE:
                break
            if best_rank <= rank:
                break
                
            # Generate random entries from {-1, 0, 1} or {-0.5, 0, 0.5, 1}
            U = np.zeros((nm, rank), dtype=float)
            V = np.zeros((mp, rank), dtype=float)
            W = np.zeros((pn, rank), dtype=float)
            
            # Fill with random values
            for r in range(rank):
                # U column
                U[:, r] = np.random.choice([-1, 0, 1], size=nm)
                V[:, r] = np.random.choice([-1, 0, 1], size=mp)
                W[:, r] = np.random.choice([-1, 0, 1], size=pn)
            
            # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            # This is equivalent to: for each (i,j,k), the inner product of
            # U[i*m+j,:], V[j*p+k,:], W[k*n+i,:] (elementwise product summed) == 1
            # i.e., U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:] summed over r == 1
            
            # Efficient check using matrix operations:
            # For fixed (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # = (U[i*m+j,:] * W[k*n+i,:]) dot V[j*p+k,:]
            # Let's precompute: for each pair (i,k), compute elementwise product of U row and W row
            # Then for each j, check if dot with V[j*p+k,:] equals 1
            
            valid = True
            for i in range(n):
                for k in range(p):
                    # Elementwise product of U[i*m+j, :] and W[k*n+i, :] for each j
                    # Actually U[i*m+j, r] depends on j too, so we need all three
                    # Let's just do it directly but vectorized over r
                    for j in range(m):
                        val = np.dot(U[i*m+j, :] * W[k*n+i, :], V[j*p+k, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid decomposition!
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Convert to integer/half-integer representation
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break  # This rank works, move to next (lower) rank
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    
    # Try random search for better rank
    try:
        result, best_rank = _random_search(max_rank=111, num_trials=30000)
        if result is not None:
            # Verify the result
            if _verify(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return schoolbook

# EVOLVE-BLOCK-END
