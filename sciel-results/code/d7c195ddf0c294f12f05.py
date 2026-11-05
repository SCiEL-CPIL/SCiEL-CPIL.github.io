import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p into 3 and 2
    p1, p2 = 3, 2
    
    # We will decompose M_{3,3,3} and M_{3,3,2} separately
    # and concatenate the results.
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
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

    # Helper to verify decomposition
    def verify(U, V, W, n, m, p):
        # Tensor T_{i,j,k} should be 1
        # Sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} == 1
        # Note: W index is k*n + i
        # U index is i*m + j
        # V index is j*p + k
        # We check for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if s != 1:
                        return False
        return True

    # Local search to reduce rank
    def reduce_rank(U, V, W, n, m, p, max_iter=1000, time_limit=5.0):
        start_time = time.time()
        rank = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Try to remove columns (reduce rank)
        # If a column can be zeroed out without breaking the decomposition, remove it.
        # Or if two columns can be merged?
        
        # Simple heuristic: Try to set entries to 0 to create zero columns
        # Then remove zero columns.
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Try to zero out a random entry
            if rank == 0:
                break
                
            # Randomly pick a non-zero entry and try to zero it
            # This is very slow. 
            # Better: Check if any column is "redundant" or can be scaled to allow another to be zeroed?
            
            # For now, just try to find a column that is all zeros? No, schoolbook has no zero columns.
            
            # Let's try a different approach: 
            # Use a known lower rank decomposition if available.
            # For 3x3x3, rank 23 is known.
            # For 3x3x2, rank 15?
            
            # Since we don't have hardcoded optimal decompositions, we will just return schoolbook 
            # unless we find a quick improvement.
            
            # Quick improvement: 
            # For 3x3x3, we can use the fact that it's 3 copies of 3x3x1.
            # No obvious reduction.
            
            pass
            
        # Remove any zero columns
        cols_to_keep = []
        for c in range(rank):
            if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
                cols_to_keep.append(c)
                
        U = U[:, cols_to_keep]
        V = V[:, cols_to_keep]
        W = W[:, cols_to_keep]
        
        return U, V, W

    # Get schoolbook decompositions
    U1, V1, W1 = schoolbook(3, 3, 3)
    U2, V2, W2 = schoolbook(3, 3, 2)
    
    # Attempt reduction (currently a no-op, but structure is there)
    # U1, V1, W1 = reduce_rank(U1, V1, W1, 3, 3, 3)
    # U2, V2, W2 = reduce_rank(U2, V2, W2, 3, 3, 2)
    
    # Concatenate
    # U is (n*m, rank) -> (9, R1) and (9, R2) -> (9, R1+R2)
    # V is (m*p, rank) -> (15, R1) and (10, R2) ?? 
    # Wait, V for 3x3x3 is (3*3, R1) = (9, R1).
    # V for 3x3x2 is (3*2, R2) = (6, R2).
    # But the full V is (m*p, rank) = (3*5, R) = (15, R).
    # The first 9 rows of V correspond to the first 3 columns of B (p1=3).
    # The next 6 rows of V correspond to the last 2 columns of B (p2=2).
    # So we can concatenate V1 and V2 vertically?
    # V1 is 9xR1. V2 is 6xR2.
    # We need a 15x(R1+R2) matrix.
    # V_full[0:9, 0:R1] = V1
    # V_full[9:15, R1:R1+R2] = V2
    # The rest is 0.
    
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U_full = np.zeros((9, R))
    V_full = np.zeros((15, R))
    W_full = np.zeros((15, R)) # W is (p*n, rank) = (5*3, R) = (15, R)
    
    # Fill U
    U_full[:, 0:R1] = U1
    U_full[:, R1:R] = U2
    
    # Fill V
    V_full[0:9, 0:R1] = V1
    V_full[9:15, R1:R] = V2
    
    # Fill W
    # W1 is (3*3, R1) = (9, R1). Indices k*n+i. k in 0..2, i in 0..2.
    # W2 is (2*3, R2) = (6, R2). Indices k*n+i. k in 0..1 (local), i in 0..2.
    # Global k for W2 is 3,4. So global index (3+i)*3 + i? No.
    # W index is k*n + i.
    # For W1, k is 0,1,2. Indices 0..8.
    # For W2, k is 3,4. Indices 9..14.
    # W2 local index is k_local * n + i.
    # Global index is (k_local + 3) * n + i.
    # So we place W2 in rows 9..14 of W_full.
    
    W_full[0:9, 0:R1] = W1
    W_full[9:15, R1:R] = W2
    
    # Verify
    if not verify(U_full, V_full, W_full, 3, 3, 5):
        # Fallback to full schoolbook
        U_f, V_f, W_f = schoolbook(3, 3, 5)
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
