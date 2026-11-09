"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def make_schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 128."""
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    def verify_fast(U, V, W, tol=1e-9):
        """Fast verification using vectorized operations."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        
        rank = U.shape[1]
        if rank == 0:
            return False
        
        # Compute all (i,j,k) values efficiently
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # For each c, we add U[:,c] * V[:,c] * W[:,c] to the tensor
        
        # Build the tensor as a 3D array
        tensor = np.zeros((n, m, p), dtype=np.float64)
        
        for c in range(rank):
            u_col = U[:, c]  # (n*m,)
            v_col = V[:, c]  # (m*p,)
            w_col = W[:, c]  # (p*n,)
            
            # Reshape and multiply
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            
            # The contribution of term c to T[i,j,k] is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # = u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            # We can compute this as: for each (i,j,k), sum over c
            # Let's compute it more efficiently
            
            # Build a 3D array for this term
            # term[i,j,k] = u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            # We can do: for each i, u_reshaped[i,:] gives (m,), 
            #            for each j, v_reshaped[j,:] gives (p,),
            #            for each k, w_reshaped[k,i] gives scalar
            # So term[i,j,k] = u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            
            # Compute via outer products
            # For fixed i: term[i,:,:] = u_reshaped[i,:] * (v_reshaped * w_reshaped[:,i])
            # where v_reshaped * w_reshaped[:,i] means element-wise: v_reshaped[j,k] * w_reshaped[k,i]
            
            for i in range(n):
                u_row = u_reshaped[i, :]  # (m,)
                # v_reshaped[j,k] * w_reshaped[k,i] for all j,k
                vw = v_reshaped * w_reshaped[:, i]  # (m, p) element-wise
                term_i = u_row[:, np.newaxis] * vw  # (m, p)
                tensor[i, :, :] += term_i
        
        # Check if tensor equals 1 everywhere
        if not np.allclose(tensor, 1.0, atol=tol):
            return False
        return True
    
    def verify_sparse(U, V, W, tol=1e-9):
        """Verification that early-terminates on first failure."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try structured search for lower rank
    # Key idea: use block structure and algebraic identities
    
    import random
    rng = random.Random(42)
    
    # Strategy 1: Try to find a decomposition with rank < 128
    # using structured patterns
    
    # For 4x4x8, we can try to exploit the fact that p=8=2*4
    # Maybe we can use a 4x4x4 decomposition (rank 64) in a clever way
    
    # Actually, let's try a more systematic approach:
    # Start with the schoolbook and try to merge terms
    
    # For now, let's try a few structured random searches
    # with better initialization
    
    # Try ranks from 127 down to 100
    for target_rank in range(best_rank - 1, max(96, best_rank - 32), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few structured attempts
        for attempt in range(2):
            if time.time() > deadline - 15:
                break
            
            # Initialize with schoolbook pattern and then perturb
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with structured pattern: each column corresponds to a (i,j,k) triple
            # but we have fewer columns than n*m*p, so we need to merge
            
            # Strategy: randomly assign each (i,j,k) to one of target_rank columns
            # and set the corresponding entries
            col_assign = {}
            col_counter = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col_counter < target_rank:
                            col = col_counter
                            col_counter += 1
                        else:
                            # Assign to an existing column
                            col = rng.randint(0, target_rank - 1)
                        col_assign[(i,j,k)] = col
            
            # Now fill U, V, W
            for (i,j,k), col in col_assign.items():
                if U[i*m+j, col] == 0:
                    U[i*m+j, col] = 1
                if V[j*p+k, col] == 0:
                    V[j*p+k, col] = 1
                if W[k*n+i, col] == 0:
                    W[k*n+i, col] = 1
            
            # Try local search: perturb some entries
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Randomly perturb a few entries
                for _ in range(3):
                    mat = rng.choice([U, V, W])
                    row = rng.randint(0, mat.shape[0] - 1)
                    col = rng.randint(0, mat.shape[1] - 1)
                    if mat[row, col] != 0:
                        mat[row, col] *= -1  # Flip sign
                    else:
                        mat[row, col] = rng.choice([-1, 1])
                
                # Quick check: verify a random subset
                sample_indices = rng.sample(range(n*m*p), min(20, n*m*p))
                valid = True
                for idx in sample_indices:
                    i = idx // (m*p)
                    j = (idx // p) % m
                    k = idx % p
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 0.1:
                        valid = False
                        break
                
                if valid:
                    # Full verification
                    if verify_sparse(U, V, W):
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        best_rank = target_rank
                        break
        
        if best_rank < best_rank + 1:  # Found something better
            continue
    
    return best_result
# EVOLVE-BLOCK-END

