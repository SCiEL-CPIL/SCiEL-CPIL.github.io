"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a low-rank exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Uses random search with NumPy-verified candidates for ranks below the schoolbook bound.
    Falls back to the schoolbook decomposition (rank 54) if no better solution is found.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Precompute indices for verification
    # i: 0..n-1, j: 0..m-1, k: 0..p-1
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Target ranks to check. We check lower ranks first.
    # Based on lower bounds and known results, we check down to 45.
    # If the DFT strategy implies a specific reduction, it might be around 45 or 48.
    target_ranks = [45, 48, 50, 52, 54]
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Define a set of candidate values for entries
    # Integer and half-integer entries
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook as fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Verification function using NumPy
    def verify(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
        
        # Create meshgrid for i, j, k
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        # Map to matrix indices
        u_rows = i_idx * m + j_idx
        v_rows = j_idx * p + k_idx
        w_rows = k_idx * n + i_idx
        
        # Extract rows
        # To be efficient, we can do this in chunks or just use advanced indexing
        # U[u_rows, :] has shape (n*m*p, rank)
        U_sub = U[u_rows, :]
        V_sub = V[v_rows, :]
        W_sub = W[w_rows, :]
        
        # Compute product and sum over rank axis
        # Product element-wise: (n*m*p, rank)
        prod = U_sub * V_sub * W_sub
        sums = np.sum(prod, axis=1)
        
        # Check if all sums are 1 (within tolerance)
        return np.allclose(sums, 1.0, atol=1e-6)

    # Search for better ranks
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
            
        if target_rank >= best_rank:
            continue
            
        # Random search attempts
        # Number of attempts depends on remaining time
        # Let's try a fixed number of attempts per rank, adjusted by time
        attempts = 500
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Using small integers/halves increases chance of exact solution if one exists
            # but limits search space.
            U = np.random.choice(values, size=(n * m, target_rank))
            V = np.random.choice(values, size=(m * p, target_rank))
            W = np.random.choice(values, size=(p * n, target_rank))
            
            # Quick check: ensure non-zero rows/cols? 
            # For now, just verify.
            if verify(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break # Found one, move to next lower rank? No, we want the MINIMAL.
                # Since we iterate from low to high, if we find rank 45, we should stop?
                # The goal is rank-minimal. So if we find 45, we can stop searching for 48, 50, etc.
                # But we should continue checking if we can go lower? 
                # The loop goes 45, 48, 50... So if we find 45, we break the inner loop.
                # We should then break the outer loop as well because 45 is the lowest we are checking.
                # If we found 48, we should still check 45? No, we check 45 first.
                # So if we find a solution for 45, we are done (assuming 45 is the lowest we try).
                # If we don't find 45, we try 48.
                pass

        if best_rank == target_rank:
            # Found a decomposition for this rank. 
            # Since we are checking in increasing order of rank, this is the minimal found so far.
            # We can break if this is the lowest rank we are interested in.
            # However, maybe we can find 44? We didn't try 44.
            # Let's assume 45 is a good lower bound target.
            break

    # Return the best found
    # Convert to lists of lists as required by interface? 
    # The interface says "factor matrices of shapes... with integer/half-integer entries".
    # It doesn't strictly specify list vs numpy array, but "U", "V", "W" keys are expected.
    # The seed program returned lists. Let's return lists to be safe, or numpy arrays if allowed.
    # "factor matrices of shapes (n*m, rank)" usually implies arrays.
    # The seed code: `return {"U": best_U, "V": best_V, "W": best_W}` where best_U was a list.
    # Let's return numpy arrays, they are often accepted. If not, convert to list.
    # To be consistent with the seed's "list" style but efficient:
    # The prompt says "Keep the function names and signatures... return interface EXACTLY as in the current program".
    # Current program returns a dict with lists.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
