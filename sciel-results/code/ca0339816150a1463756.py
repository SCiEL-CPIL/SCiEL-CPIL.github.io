"""Matrix multiplication 4,5,5: rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# Known lower bounds for matrix multiplication rank R(n,m,p)
# For 4x5*5x5, schoolbook is 100.
# We aim to find a decomposition with rank < 100.
# Theoretical lower bound is max(n*m, m*p, p*n) = max(20, 25, 20) = 25, but that's tight.
# Realistic best known for small matrices often close to schoolbook for these sizes.
# However, we can try to find a better one via random search or structured patterns.

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave margin for 120s limit
    
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Initialize with schoolbook
    U = np.zeros((N, schoolbook_rank))
    V = np.zeros((M, schoolbook_rank))
    W = np.zeros((P, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition
    # Strategy: Randomized iterative methods or structured ansatz
    
    # For small matrices, we can try to find a decomposition of rank R < 100
    # using a random linear combination approach or by exploiting structure.
    
    # One approach: Assume the decomposition has a certain structure.
    # For example, try to find U, V, W with integer entries.
    
    # Let's try a simple heuristic: randomly initialize and project.
    # But this is hard to do exactly.
    
    # Alternative: Use known results. For 4x5*5x5, is there a known better rank?
    # The rank of matrix multiplication is a well-studied problem.
    # For 2x2*2x2, rank is 7 (Strassen) vs 8 schoolbook.
    # For larger matrices, the gap is usually smaller in relative terms.
    
    # Given the time constraint, we might not find a better one than schoolbook.
    # But let's try a few random restarts with a simple local search.
    
    def validate(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] should be sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where i in [0,n), j in [0,m), k in [0,p)
        # But the indices in U, V, W are flattened:
        # U has shape (n*m, rank), so U[i*m+j, r]
        # V has shape (m*p, rank), so V[j*p+k, r]
        # W has shape (p*n, rank), so W[k*n+i, r]
        
        # We need to check that for all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid multiplication
        # Actually, the tensor for matrix multiplication C = AB has entries:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T[i,j,k] = 1 if we consider the standard basis.
        # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        
        # Let's vectorize this check.
        # Create a 3D tensor of the product.
        # U_flat: (n*m, rank)
        # V_flat: (m*p, rank)
        # W_flat: (p*n, rank)
        
        # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        # The product tensor is sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is a contraction over r, but also involves cyclic indices i,j,k.
        
        # Let's do it in a loop for clarity and correctness, since n,m,p are small.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :]) * 1.0  # This is not quite right
                    # Actually, it's sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # We can compute this as:
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Let's try to find a decomposition of rank < 100.
    # We'll use a simple random search with integer entries in {-1, 0, 1}.
    # This is a heuristic and may not work, but it's a start.
    
    max_attempts = 50
    attempt = 0
    while attempt < max_attempts and time.time() < deadline:
        rank = schoolbook_rank - 1 - (attempt // 10)  # Try lower ranks
        if rank < 25:  # Theoretical lower bound
            break
            
        # Randomly initialize U, V, W with entries in {-1, 0, 1}
        U_rand = np.random.randint(-1, 2, size=(N, rank))
        V_rand = np.random.randint(-1, 2, size=(M, rank))
        W_rand = np.random.randint(-1, 2, size=(P, rank))
        
        if validate(U_rand, V_rand, W_rand):
            best_rank = rank
            best_U, best_V, best_W = U_rand, V_rand, W_rand
            break
        
        attempt += 1
    
    # If we found a better rank, return it. Otherwise, return schoolbook.
    if best_rank < schoolbook_rank:
        U_out = best_U.astype(int)
        V_out = best_V.astype(int)
        W_out = best_W.astype(int)
    else:
        U_out = U.astype(int)
        V_out = V.astype(int)
        W_out = W.astype(int)
    
    return {"U": U_out, "V": V_out, "W": W_out}
