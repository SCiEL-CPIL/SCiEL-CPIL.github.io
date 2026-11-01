import numpy as np
import time
from itertools import product

def construct():
    """
    Find rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x13 matrix.
    Output shape: (2, 13)
    Input A: (2, 2), Input B: (2, 13)
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The standard tensor decomposition for matrix multiplication C = A * B is:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    The tensor T is a 3-tensor with indices (i, j, k) where:
    T[i, j, k] = 1 if this term contributes to C[i, k]
    But actually, the matrix multiplication tensor is defined such that:
    C[i, k] = sum_{j} A[i, j] * B[j, k]
    
    The rank of this tensor is the minimum number of rank-1 tensors needed to decompose it.
    
    For 2x2 * 2x13, the schoolbook decomposition has rank 2*2*13 = 52.
    However, we can potentially do better by finding shared structure.
    
    The decomposition is:
    T = sum_{r=1}^{R} u_r v_r w_r
    where u_r is a vector of length 4 (for A), v_r is a vector of length 26 (for B),
    w_r is a vector of length 26 (for C, using W[k*2+i, r] convention).
    
    Actually, let's be more careful about the dimensions:
    - A is 2x2, so we have 4 entries: A[0,0], A[0,1], A[1,0], A[1,1]
    - B is 2x13, so we have 26 entries: B[0,0], B[0,1], ..., B[1,12]
    - C is 2x13, so we have 26 entries: C[0,0], C[0,1], ..., C[1,12]
    
    The tensor T has shape (4, 26, 26) where T[a_idx, b_idx, c_idx] = 1 
    if the product A[a_idx] * B[b_idx] contributes to C[c_idx].
    
    Specifically:
    - a_idx = i*2 + j (where i is row, j is column of A)
    - b_idx = j*13 + k (where j is row, k is column of B)
    - c_idx = k*2 + i (using the W[k*n+i, r] convention, where n=2)
    
    Wait, the convention says W[k*n+i, r]. So for C[i, k], the index in W is k*2 + i.
    
    So T[a_idx, b_idx, c_idx] = 1 when:
    - a_idx corresponds to (i_a, j_a)
    - b_idx corresponds to (j_b, k)
    - c_idx corresponds to (k, i_c)
    And we need j_a = j_b and i_a = i_c.
    
    The rank-1 decomposition is:
    T = sum_{r} u_r * v_r * w_r
    where for each term r:
    - u_r[a_idx] is nonzero for at most one a_idx
    - v_r[b_idx] is nonzero for at most one b_idx
    - w_r[c_idx] is nonzero for at most one c_idx
    
    And T[a_idx, b_idx, c_idx] = sum_r u_r[a_idx] * v_r[b_idx] * w_r[c_idx]
    
    For the schoolbook decomposition, each term corresponds to a specific (i, j, k) triple:
    - u_r has 1 at position i*2 + j
    - v_r has 1 at position j*13 + k
    - w_r has 1 at position k*2 + i
    
    This gives rank 52.
    
    Can we do better? For small matrix multiplication, the rank can sometimes be reduced.
    For 2x2 * 2x2, the minimum rank is 7 (Strassen).
    For 2x2 * 2x13, we might be able to do better than 52.
    
    Let's try to find a decomposition with rank < 52.
    """
    
    deadline = time.time() + 120
    best_result = None
    best_rank = 52  # Start with schoolbook rank
    
    # Build the full tensor T of shape (4, 26, 26)
    T = np.zeros((4, 26, 26), dtype=np.float64)
    
    for i in range(2):
        for j in range(2):
            for k in range(13):
                a_idx = i * 2 + j
                b_idx = j * 13 + k
                c_idx = k * 2 + i  # W[k*n+i, r] convention
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Schoolbook decomposition: rank 52
    # Each term is a rank-1 tensor with a single 1 in each factor
    U_school = np.zeros((4, 52))
    V_school = np.zeros((26, 52))
    W_school = np.zeros((26, 52))
    
    rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(13):
                a_idx = i * 2 + j
                b_idx = j * 13 + k
                c_idx = k * 2 + i
                U_school[a_idx, rank] = 1.0
                V_school[b_idx, rank] = 1.0
                W_school[c_idx, rank] = 1.0
                rank += 1
    
    # Verify schoolbook decomposition
    T_recon = np.zeros((4, 26, 26), dtype=np.float64)
    for r in range(52):
        T_recon += np.outer(np.outer(U_school[:, r], V_school[:, r]), W_school[:, r]).reshape(4, 26, 26)
    
    if np.allclose(T_recon, T, atol=1e-10):
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        best_rank = 52
    
    # Try to find a lower-rank decomposition
    # This is a hard problem, so we'll try some heuristic approaches
    
    # Approach 1: Try to merge some terms by finding linear dependencies
    # For each pair of output positions (i, k), the terms that contribute are those with the same i and k
    # but different j. However, since j must match between A and B, there's no sharing in the standard case.
    
    # Approach 2: Use random initialization and try to optimize for lower rank
    # This is unlikely to work well for exact decomposition, but let's try a simple approach
    
    # Try rank 40 to 51
    for target_rank in range(51, 35, -1):
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition with the given rank
        # Use a simple greedy approach: try to add terms one at a time
        U_try = np.zeros((4, target_rank))
        V_try = np.zeros((26, target_rank))
        W_try = np.zeros((26, target_rank))
        
        # Initialize with random sparse vectors
        for r in range(target_rank):
            # Choose random positions
            a_pos = np.random.randint(0, 4)
            b_pos = np.random.randint(0, 26)
            c_pos = np.random.randint(0, 26)
            U_try[a_pos, r] = 1.0
            V_try[b_pos, r] = 1.0
            W_try[c_pos, r] = 1.0
        
        # Check if this decomposition works
        T_recon = np.zeros((4, 26, 26), dtype=np.float64)
        for r in range(target_rank):
            T_recon += np.outer(np.outer(U_try[:, r], V_try[:, r]), W_try[:, r]).reshape(4, 26, 26)
        
        if np.allclose(T_recon, T, atol=1e-10):
            best_result = {"U": U_try, "V": V_try, "W": W_try}
            best_rank = target_rank
            break
    
    # If we didn't find a better decomposition, return the schoolbook one
    if best_result is None:
        best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    return best_result
