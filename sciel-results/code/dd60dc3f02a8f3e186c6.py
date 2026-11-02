"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    # First, construct the known rank-14 solution
    rank_14 = _construct_rank_14(n, m, p)
    
    # Try to find a rank-13 solution by merging terms
    if time.time() - start_time < deadline - 30:
        rank_13 = _try_rank_13(n, m, p, deadline)
        if rank_13 is not None:
            return rank_13
    
    return rank_14

def _construct_rank_14(n, m, p):
    """Construct the known rank-14 decomposition using Strassen on two 2x2 blocks."""
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (0,0,-1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for r in range(7):
        u_entries, v_entries, w_entries = strassen_terms[r]
        
        # Block 1: columns 0 and 1 of B (k=0,1)
        col1 = r
        for i, j, val in u_entries:
            U[i * m + j][col1] += val
        for j, k, val in v_entries:
            V[j * p + k][col1] += val
        for k, i, val in w_entries:
            W[k * n + i][col1] += val
            
        # Block 2: columns 2 and 3 of B (k=2,3)
        col2 = 7 + r
        for i, j, val in u_entries:
            U[i * m + j][col2] += val
        for j, k, val in v_entries:
            k_shifted = k + 2
            V[j * p + k_shifted][col2] += val
        for k, i, val in w_entries:
            k_shifted = k + 2
            W[k_shifted * n + i][col2] += val
    
    return {"U": U, "V": V, "W": W}

def _try_rank_13(n, m, p, deadline):
    """Try to find a rank-13 decomposition by merging Strassen terms."""
    # The idea: if we can express one of the 14 rank-1 terms as a linear combination
    # of the other 13, we can reduce the rank.
    # 
    # We'll set up a system where we try to find coefficients that allow us to
    # merge terms. Specifically, we'll look for a decomposition of the form:
    # T = sum_{r=1}^{13} U_r * V_r * W_r
    #
    # by trying to absorb one term into the others.
    
    # Build the tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try different merging strategies
    # Strategy 1: Try to merge term 7 (last Strassen term) with others
    # by expressing it as a combination of the first 13 terms.
    
    # For each pair of terms to merge, solve for the coefficients
    # This is a nonlinear problem, so we'll use a heuristic approach.
    
    # Alternative: Try to find a rank-13 solution by solving a system of equations
    # where we fix most entries and solve for the remaining ones.
    
    # Given the complexity, let's try a simpler approach:
    # Check if there's a linear dependency among the 14 rank-1 tensors.
    
    # Flatten each rank-1 tensor U_r * V_r * W_r into a vector
    rank_14 = _construct_rank_14(n, m, p)
    U14 = np.array(rank_14['U'], dtype=float)
    V14 = np.array(rank_14['V'], dtype=float)
    W14 = np.array(rank_14['W'], dtype=float)
    
    # Check if the 14 rank-1 tensors are linearly independent
    # If not, we can potentially reduce the rank.
    
    # Build the matrix where each column is a flattened rank-1 tensor
    tensors = []
    for r in range(14):
        u_col = U14[:, r]
        v_col = V14[:, r]
        w_col = W14[:, r]
        # Outer product and flatten
        tensor = np.outer(np.outer(u_col, v_col), w_col).flatten()
        tensors.append(tensor)
    
    tensors_matrix = np.column_stack(tensors)
    
    # Check rank of this matrix
    rank_of_tensors = np.linalg.matrix_rank(tensors_matrix)
    
    if rank_of_tensors < 14:
        # There's a linear dependency, try to find it
        # Use SVD to find the null space
        U_svd, s, Vt = np.linalg.svd(tensors_matrix, full_matrices=False)
        
        # The null space vectors are in the last rows of Vt
        null_vec = Vt[-1]
        
        # If we can express one tensor as a combination of the others,
        # we might be able to reduce the rank.
        # However, this doesn't directly give us a rank-13 decomposition.
        # We need to check if the merged decomposition is valid.
        
        # For now, return None since we haven't found a valid rank-13 solution
        return None
    
    # No linear dependency found, return None
    return None

# EVOLVE-BLOCK-END
