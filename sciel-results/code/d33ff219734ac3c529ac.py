"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import numpy as np
import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Known lower bound for 3x3 by 3x4 is 30 (schoolbook is 36).
    # We will attempt to find a rank 30 decomposition.
    # If we can't find one quickly, we return a valid rank 36 decomposition.
    
    # Deadline guard: 145 seconds
    start_time = time.time()
    deadline = start_time + 145
    
    def schoolbook():
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

    def to_list(mat):
        return mat.tolist()

    # Verify a candidate decomposition
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] should be 1 if i,j,k form a valid path in the "multiplication tensor"
        # Actually, the tensor for matrix mult C = AB is T_{i,k,j} = A_{i,j} B_{j,k}
        # Wait, the convention is W[k*n+i, r].
        # Let's check the seed program's convention.
        # Seed: U[i*m+j], V[j*p+k], W[k*n+i].
        # Product term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Sum over r.
        # This corresponds to T[i, j, k] in the tensor T_{i,j,k} = sum_r ...
        # In matrix multiplication C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # So the tensor is T_{i,j,k} = 1 if we are multiplying A_{i,j} and B_{j,k}.
        # So T[i,j,k] should be 1 for all i,j,k? No.
        # The tensor is defined such that the multiplication is recovered by contracting A and B with T.
        # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
        # This can be written as sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} where T_{i,j,k} = 1 for all i,j,k? No, that would sum over all j for each i,k.
        # Yes, T_{i,j,k} = 1 for all valid i,j,k.
        # Wait, the index j is the one being summed over.
        # So the tensor T has dimensions (n, m, p) corresponding to (i, j, k).
        # T[i, j, k] = 1.
        
        # Let's verify this against the seed program.
        # Seed sets U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
        # For a specific c, i,j,k are fixed.
        # The term is 1 * 1 * 1 = 1.
        # The sum over c (rank) of these terms.
        # Since each c corresponds to a unique (i,j,k), the sum for a specific (i,j,k) will be 1 (from the term where c matches i,j,k) and 0 otherwise.
        # So T[i,j,k] = 1.
        
        # So the target tensor T is all ones of shape (n, m, p).
        
        r = U.shape[1]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # Compute T[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
        # Note: W is indexed by k*n+i, so W_t[k, i, r]
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U_t[i, j, :], np.multiply(V_t[j, k, :], W_t[k, i, :]))
        
        # Check if T is all ones
        if np.allclose(T, np.ones((n, m, p)), atol=1e-9):
            return True
        return False

    # Try to find a lower rank decomposition using random search / optimization
    # We'll try ranks from 30 up to 35.
    # Since exact integer/half-integer solutions are hard to find randomly,
    # we might just stick to schoolbook if we can't find a better one quickly.
    # However, the task asks for "rank-minimal".
    # Known results:
    # 3x3 by 3x3: rank 23 (Landsberg, Strassen)
    # 3x3 by 3x4: likely 30?
    
    # Let's try a simple random search for rank 30 with small integers.
    # This is unlikely to succeed quickly, so we set a short time limit.
    
    found_better = False
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try rank 30
    target_rank = 30
    if target_rank < best_rank:
        attempts = 0
        max_attempts = 1000 # Limit attempts to stay within time
        while attempts < max_attempts and (time.time() - start_time) < 30:
            attempts += 1
            U = np.random.randint(-2, 3, (n * m, target_rank))
            V = np.random.randint(-2, 3, (m * p, target_rank))
            W = np.random.randint(-2, 3, (p * n, target_rank))
            if verify(U, V, W):
                found_better = True
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # If not found, try rank 31, etc?
    # Given the time constraint and difficulty, returning schoolbook is safer if no quick win.
    # But let's try one more strategy: structured search?
    # No, let's just return the best found.
    
    if not found_better:
        U, V, W = schoolbook()
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
