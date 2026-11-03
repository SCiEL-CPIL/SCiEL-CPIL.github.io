import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a rank-minimal decomposition
    # Schoolbook rank is n*m*p = 75
    # Try to use block structure to reduce rank
    
    # Strategy: Partition A (3x5) into rows, B (5x5) into blocks
    # A = [A1; A2; A3] where A1, A2, A3 are 1x5
    # B is 5x5
    
    # For 1x5 * 5x5, the rank is 5 (each output is a dot product)
    # But we can share computations across rows
    
    # Let's try a systematic search for sparse decompositions
    # using the structure of the problem
    
    # First, let's verify the schoolbook decomposition works
    def verify_decomposition(U, V, W, rank):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_comp = np.zeros((n, p))
        for c in range(rank):
            scalar_u = np.dot(U[:, c], A.flatten())
            scalar_v = np.dot(V[:, c], B.flatten())
            w_vec = W[:, c]
            w_mat = w_vec.reshape(p, n)
            C_comp += w_mat.T * (scalar_u * scalar_v)
        
        return np.allclose(C_true, C_comp, atol=1e-9)
    
    # Schoolbook decomposition (rank 75)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # Use a greedy approach: try to combine terms
    
    # For 3x5 * 5x5, we can try to use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try rank values from low to high and see if we can find a valid decomposition
    # Start with a reasonable upper bound
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a decomposition with rank < 75
    # Use a randomized search with structured patterns
    
    # Pattern 1: For each output C[i,k], we need sum_j A[i,j] B[j,k]
    # We can try to share the A[i,j] factors across different k values
    
    # Let's try a systematic construction
    # Group terms by the A[i,j] factor
    
    # For fixed i and j, we have terms A[i,j] * B[j,k] for all k
    # If we can compute B[j,k] for all k with fewer terms, we save
    
    # Actually, let's try a different approach:
    # Use the fact that B is 5x5, and try to decompose it using
    # a lower-rank bilinear form
    
    # For now, let's try to verify if rank 75 works and return it
    # as a baseline, then try to improve
    
    if verify_decomposition(U_school, V_school, W_school, rank_schoolbook):
        best_rank = rank_schoolbook
        best_U = U_school.copy()
        best_V = V_school.copy()
        best_W = W_school.copy()
    
    # Try to find a lower rank using structured patterns
    # Pattern: For each i, share the j-index across k values
    
    # Let's try rank = 60 (3 rows * 5 cols * 4 shared terms)
    # This is just a heuristic; we need to actually find valid decompositions
    
    # Try a few candidate ranks
    candidate_ranks = [75, 70, 65, 60, 55, 50]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        
        # Try to construct a decomposition with this rank
        # This is a simplified attempt; a full search would be needed
        
        # For now, just verify if the schoolbook rank works
        # and return the best found
        
        elapsed = time.time() - start_time
        if elapsed > 120:
            break
    
    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
