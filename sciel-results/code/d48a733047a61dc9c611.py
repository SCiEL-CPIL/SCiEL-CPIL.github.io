import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x3 using Schoolbook (Rank 48)
    
    # Generate Strassen decomposition for 4x4x4
    U_strassen, V_strassen, W_strassen = get_strassen_decomp(4)
    
    # Generate Schoolbook decomposition for 4x4x3
    U_school, V_school, W_school = get_schoolbook_decomp(4, 4, 3)
    
    # Combine
    # U: (16, 49) and (16, 48) -> (16, 97)
    # V: (28, 49) and (28, 48) -> (28, 97)
    # W: (28, 49) and (28, 48) -> (28, 97)
    
    # Note: The Strassen decomp is for 4x4x4.
    # U_strassen is 16x49.
    # V_strassen is 16x49 (but we need 28x49, so pad with zeros for the last 12 rows of B)
    # W_strassen is 16x49 (but we need 28x49, so pad with zeros for the last 12 rows of C)
    
    rank1 = U_strassen.shape[1]
    rank2 = U_school.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill Part 1 (Strassen)
    # U_strassen is 16x49. Fits in first 16 rows of U.
    U[:16, :rank1] = U_strassen
    
    # V_strassen is 16x49. Corresponds to B[:, 0:4].
    # In V matrix (28x49), rows 0-15 correspond to B[:, 0:4].
    V[:16, :rank1] = V_strassen
    
    # W_strassen is 16x49. Corresponds to C[:, 0:4].
    # In W matrix (28x49), rows 0-15 correspond to C[:, 0:4].
    W[:16, :rank1] = W_strassen
    
    # Fill Part 2 (Schoolbook)
    # U_school is 16x48.
    U[:16, rank1:total_rank] = U_school
    
    # V_school is 12x48? No, get_schoolbook_decomp(4,4,3) returns V of size (4*3, 48) = (12, 48).
    # But we need it in the full V space of size 28.
    # The schoolbook decomp for 4x4x3 uses B[:, 4:7].
    # So the V rows should be 16-27.
    V[16:28, rank1:total_rank] = V_school
    
    # W_school is 12x48? No, W is (p*n, rank) = (3*4, 48) = (12, 48).
    # The schoolbook decomp for 4x4x3 uses C[:, 4:7].
    # So the W rows should be 16-27.
    W[16:28, rank1:total_rank] = W_school
    
    # Convert to lists of lists as per interface?
    # The interface says "factor matrices of shapes...". It doesn't strictly say numpy arrays are forbidden,
    # but the seed uses lists. Let's return numpy arrays if accepted, or lists.
    # "integer/half-integer entries"
    # The evaluator likely accepts numpy arrays or lists.
    # To be safe, let's return numpy arrays. The seed used lists, but numpy is available.
    
    return {"U": U, "V": V, "W": W}

def get_strassen_decomp(N):
    # Returns U, V, W for NxNxN matrix multiplication using Strassen recursion.
    # N must be a power of 2.
    if N == 1:
        return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    
    # Get decomposition for N/2
    U_half, V_half, W_half = get_strassen_decomp(N // 2)
    R = U_half.shape[1]
    
    U_full = np.zeros((4 * N * N, 7 * R)) # Wait, size is (N*N, R). For 2N, it's (4*N*N, 7*R).
    # Actually, input size is (2N)^2 = 4N^2.
    U_full = np.zeros((4 * N * N, 7 * R))
    V_full = np.zeros((4 * N * N, 7 * R))
    W_full = np.zeros((4 * N * N, 7 * R))
    
    # Define the 7 Strassen steps
    # Each step has:
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    
    # Block indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # A is 2x2 blocks.
    
    steps = [
        { 'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)] }, # M1
        { 'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)] }, # M2 -> C10, C11
        # Correction on M2: C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So M2 contributes +1 to C10 and -1 to C11.
        # C10 is block 2. C11 is block 3.
        
        { 'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)] }, # M3 -> C01, C11
        # M3 = A00(B01 - B11). C01 = M3 + M5. C11 = ... + M3.
        # So +1 to C01 (block 1), +1 to C11 (block 3).
        
        { 'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)] }, # M4 -> C00, C10
        # M4 = A11(B10 - B00). C00 = M1 + M4 - M5 + M7. C10 = M2 + M4.
        # So +1 to C00 (block 0), +1 to C10 (block 2).
        
        { 'A': [(0, 1), (2, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)] }, # M5 -> C00, C01
        # M5 = (A00 + A10)B11. C00 = ... - M5. C01 = M3 + M5.
        # So -1 to C00 (block 0), +1 to C01 (block 1).
        
        { 'A': [(3, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)] }, # M6 -> C11
        # M6 = (A11 - A00)(B00 + B01). C11 = ... + M6.
        # So +1 to C11 (block 3).
        
        { 'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1), (3, 1)] }, # M7 -> C00, C11
        # M7 = (A01 - A11)(B10 + B11). C00 = ... + M7. C11 = ... + M7.
        # So +1 to C00 (block 0), +1 to C11 (block 3).
    ]
    
    # Fix M2 C blocks:
    # M2 contributes to C10 (+1) and C11 (-1).
    steps[1]['C'] = [(2, 1), (3, -1)]
    
    for step_idx, step in enumerate(steps):
        for r in range(R):
            u_base = U_half[:, r]
            v_base = V_half[:, r]
            w_base = W_half[:, r]
            
            u_new = np.zeros(4 * N * N)
            v_new = np.zeros(4 * N * N)
            w_new = np.zeros(4 * N * N)
            
            for block_idx, coeff in step['A']:
                # Place u_base into block block_idx of u_new
                start = block_idx * N * N
                u_new[start:start + N * N] += coeff * u_base
                
            for block_idx, coeff in step['B']:
                start = block_idx * N * N
                v_new[start:start + N * N] += coeff * v_base
                
            for block_idx, coeff in step['C']:
                start = block_idx * N * N
                w_new[start:start + N * N] += coeff * w_base
                
            col = step_idx * R + r
            U_full[:, col] = u_new
            V_full[:, col] = v_new
            W_full[:, col] = w_new
            
    return U_full, V_full, W_full

def get_schoolbook_decomp(n, m, p):
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
# EVOLVE-BLOCK-END
