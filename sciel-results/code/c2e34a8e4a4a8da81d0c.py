"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i,j,k] = 1 if i*m+j, j*p+k, k*n+i
    # The decomposition says: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # i.e., T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check a set of random integer matrices
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r]  # shape (n*m,)
            v_vec = V[:, r]  # shape (m*p,)
            w_vec = W[:, r]  # shape (p*n,)
            
            # Reshape
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # The contribution to C[i,k] from rank r is:
            # sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    C_computed[i, k] += s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _schoolbook():
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from best_rank-1 down, using structured patterns
    
    # For 3x5x5, let's try a few structured approaches
    
    # Approach 1: Try rank 74, 73, ... with small perturbations
    # The tensor T[i,j,k] = delta_{i*m+j, j*p+k, k*n+i} in the decomposition sense
    
    # Let's try a more systematic search for small rank reductions
    # Key: the entries must be integer or half-integer
    
    # Try structured decompositions based on grouping
    # For n=3, we have 3 rows of A. Maybe we can share some computation.
    
    # Let's try: can we do it in rank < 75?
    # One idea: for each (j,k) pair, we need to compute A[i,j]*B[j,k] for i=0,1,2
    # That's 3 values. If we could compute all 3 with fewer than 3 "rank-1" terms...
    # But each rank-1 term contributes u[i,j]*v[j,k]*w[k,i] to C[i,k]
    
    # Let me try a random search with structured initialization
    np.random.seed(123)
    
    for target_rank in range(best_rank - 1, max(best_rank - 20, 60), -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts at this rank
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook-like structure but with some entries zeroed
            # or with small integer/half-integer values
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Start from schoolbook and try to merge some columns
            # Pick random pairs of schoolbook columns and try to combine them
            sb = _schoolbook()
            sb_U = np.array(sb['U'], dtype=float)
            sb_V = np.array(sb['V'], dtype=float)
            sb_W = np.array(sb['W'], dtype=float)
            
            # We need to represent 75 basis vectors as target_rank vectors
            # Each basis vector r has U[:,r], V[:,r], W[:,r] with single 1s
            # We want to write these as linear combinations of target_rank vectors
            
            # This is hard in general. Let's try a different approach:
            # Random search with verification
            
            # Generate random U, V, W with small integer/half-integer entries
            # and check if they decompose the tensor
            
            # Actually, let's try a more targeted approach:
            # The tensor T is sparse (75 ones). We need sum_r u_r * v_r * w_r = T
            # where u_r is a vector in R^(n*m), etc.
            
            # Let's try: for each of the 75 positions, assign it to a rank component
            # and see if we can merge some
            
            # Simpler: try rank 74 by merging two schoolbook terms
            if target_rank == best_rank - 1:
                # Try merging two random schoolbook columns
                idx1, idx2 = np.random.choice(best_rank, 2, replace=False)
                
                U_new = np.zeros((n * m, target_rank))
                V_new = np.zeros((m * p, target_rank))
                W_new = np.zeros((p * n, target_rank))
                
                # Copy all columns except idx1, idx2
                cols_keep = [c for c in range(best_rank) if c not in (idx1, idx2)]
                for new_col, old_col in enumerate(cols_keep):
                    U_new[:, new_col] = sb_U[:, old_col]
                    V_new[:, new_col] = sb_V[:, old_col]
                    W_new[:, new_col] = sb_W[:, old_col]
                
                # Now try to represent the sum of columns idx1 and idx2 as one column
                # u1*v1*w1 + u2*v2*w2 = u_new * v_new * w_new
                # This is generally not possible with a single rank-1 term unless
                # the two terms are "compatible"
                
                # Check if the two schoolbook terms can be merged
                # Term idx1: (i1, j1, k1) -> U[i1*m+j1, idx1]=1, V[j1*p+k1, idx1]=1, W[k1*n+i1, idx1]=1
                # Term idx2: (i2, j2, k2)
                
                # Decode indices
                i1, j1 = divmod(np.where(sb_U[:, idx1] != 0)[0][0], m)
                k1 = np.where(sb_V[:, idx1] != 0)[0][0] - j1 * p
                # Verify
                i2, j2 = divmod(np.where(sb_U[:, idx2] != 0)[0][0], m)
                k2 = np.where(sb_V[:, idx2] != 0)[0][0] - j2 * p
                
                # For merging: we need u_new*v_new*w_new = u1*v1*w1 + u2*v2*w2
                # This requires the two tensors to be rank 1 together, which they're not
                # unless they share structure. Generally impossible.
                pass
            
            # For now, just keep schoolbook if no improvement found
            pass
    
    # Return the best found
    return best

# EVOLVE-BLOCK-END
