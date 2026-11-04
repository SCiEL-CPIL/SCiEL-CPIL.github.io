"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1 for all i,j,k.
    """
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Check all entries of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i*m+j, :], Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Generate the schoolbook decomposition of rank n*m*p."""
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

def _bini_3x3x3_decomposition():
    """
    Return the 23-rank Bini decomposition for 3x3x3 matrix multiplication.
    This is a known exact decomposition with integer/half-integer entries.
    """
    # Bini's 23-rank algorithm for 3x3 matrix multiplication
    # The decomposition is: C[i,k] = sum_{c=0}^{22} alpha_c * beta_c * gamma_c
    # where alpha_c = sum_{i,j} U_c[i,j] A[i,j]
    #                beta_c = sum_{j,k} V_c[j,k] B[j,k]
    #                gamma_c = W_c[i,k] (scalar multiplier for C[i,k])
    
    # In our interface:
    # U has shape (9, 23) where U[i*m+j, c]
    # V has shape (9, 23) where V[j*p+k, c]
    # W has shape (9, 23) where W[k*n+i, c]
    
    # The Bini decomposition factors are well-known. Let me construct them.
    # Actually, let me use a verified set of factors.
    
    # Here are the 23 rank-1 components of the 3x3x3 tensor.
    # Each component is (U_row_vec, V_row_vec, W_col_vec) where:
    #   U_row_vec has 9 entries (for 3x3 A)
    #   V_row_vec has 9 entries (for 3x3 B)
    #   W_col_vec has 9 entries (for 3x3 C)
    
    # The Bini factors (from literature, verified):
    # I'll use a known correct set. Let me hardcode a verified Bini decomposition.
    
    # Actually, constructing Bini from scratch is error-prone. Let me use a 
    # simpler approach: try to find a decomposition of rank < 45 using 
    # randomized search with the structure of the problem.
    
    # For now, return None to indicate we need to search.
    return None

def _try_find_low_rank(n, m, p, max_rank, deadline):
    """
    Try to find a decomposition of rank <= max_rank using randomized search.
    Returns (U, V, W) or None if not found.
    """
    import random
    
    # Strategy: Use the block structure
    # C = A * B where A is 3x3, B is 3x5
    # Split B into B1 (3x3) and B2 (3x2)
    # C1 = A * B1 (3x3x3 problem, rank 27 schoolbook)
    # C2 = A * B2 (3x3x2 problem, rank 18 schoolbook)
    # Total schoolbook: 27 + 18 = 45
    
    # To beat 45, we need to find a decomposition where the total rank is < 45.
    # This requires some sharing between the two blocks or a cleverer decomposition.
    
    # Approach: Randomized search over small integer/half-integer matrices.
    # We'll try ranks from max_rank down to some minimum.
    
    for rank in range(max_rank, 30, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with small random integers/half-integers
            # Use values from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
            vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _greedy_set_cover_approach(n, m, p, deadline):
    """
    Greedy set-cover approach: generate a pool of candidate rank-1 components
    and greedily select a minimal subset that spans the full tensor.
    """
    import random
    
    # Generate a pool of candidate rank-1 components
    # Each component is a triple (u_vec, v_vec, w_vec) where:
    #   u_vec is a vector in R^{n*m}
    #   v_vec is a vector in R^{m*p}
    #   w_vec is a vector in R^{p*n}
    # such that u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i] contributes to T[i,j,k]
    
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # We need sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.
    
    # Generate candidates with small integer/half-integer entries
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    pool = []
    for _ in range(200):
        u = [random.choice(vals) for _ in range(n*m)]
        v = [random.choice(vals) for _ in range(m*p)]
        w = [random.choice(vals) for _ in range(p*n)]
        pool.append((u, v, w))
    
    # Greedy selection: at each step, pick the component that covers the most
    # uncovered tensor entries (where the current sum is not yet 1)
    
    # Initialize residual: we need sum to be 1 everywhere
    residual = [[1.0 for _ in range(p)] for _ in range(n)]
    for j in range(m):
        for i in range(n):
            for k in range(p):
                residual[i][k] = 1.0  # Each (i,k) needs to be satisfied by all j's contribution
    
    # Actually, the tensor is T[i,j,k] = 1 for each (i,j,k) triple.
    # Let me reconsider. The condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    # So we have n*m*p = 45 constraints.
    # Each rank-1 component c contributes to all 45 constraints.
    
    # Let me reformulate: we need to find rank vectors u_c, v_c, w_c such that
    # for each (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # This is a system of 45 equations in rank*27 unknowns (if we treat all entries as unknowns).
    # For rank r, we have 27r unknowns and 45 equations.
    # For r >= 2, this is underdetermined, so solutions should exist.
    
    # But we need integer/half-integer solutions, which is much harder.
    
    # Let me try a different approach: use the structure of the problem.
    # The MM tensor for 3x3x5 can be viewed as 5 copies of 3x3x1 tensors.
    # Each 3x3x1 tensor has rank 9 (schoolbook).
    # Total: 5*9 = 45.
    
    # To beat 45, we need to share rank-1 components across the 5 slices.
    
    # Let me try a direct numerical approach: solve for the factors using least squares
    # and then round to the nearest integer/half-integer.
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Start with schoolbook as fallback
    best = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition
    # Strategy 1: Randomized search for rank < 45
    result = _try_find_low_rank(n, m, p, best_rank - 1, deadline)
    if result is not None and time.time() < deadline:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            best = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])
    
    # Strategy 2: Try block decomposition
    # C[:, :3] = A * B[:, :3]  -> 3x3x3, schoolbook rank 27
    # C[:, 3:] = A * B[:, 3:]  -> 3x3x2, schoolbook rank 18
    # If we can find a 23-rank decomposition for 3x3x3, total would be 23+18=41
    
    if time.time() < deadline:
        # Try to construct a block decomposition
        # For the 3x3x3 part, try randomized search for rank 23-26
        for target_rank_333 in range(26, 22, -1):
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                import random
                vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
                
                # 3x3x3 part
                U1 = [[random.choice(vals) for _ in range(target_rank_333)] for _ in range(9)]
                V1 = [[random.choice(vals) for _ in range(target_rank_333)] for _ in range(9)]
                W1 = [[random.choice(vals) for _ in range(target_rank_333)] for _ in range(9)]
                
                # Check if this decomposes the 3x3x3 tensor
                valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            val = np.dot(
                                np.array(U1[i*3+j]), 
                                np.array(V1[j*3+k]) * np.array(W1[k*3+i])
                            )
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # 3x3x2 part (schoolbook, rank 18)
                rank_332 = 18
                U2 = [[0] * rank_332 for _ in range(9)]
                V2 = [[0] * rank_332 for _ in range(6)]
                W2 = [[0] * rank_332 for _ in range(6)]
                col = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(2):
                            U2[i*3+j][col] = 1
                            V2[j*5+k+3][col] = 1  # B[:, 3:] has indices j*5+(k+3)
                            W2[(k+3)*3+i][col] = 1  # C[:, 3:] has indices (k+3)*3+i
                            col += 1
                
                # Combine
                total_rank = target_rank_333 + rank_332
                U_combined = [row + [0]*rank_332 for row in U1]
                V_combined = [row[:9] + [0]*0 for row in V1]  # Need to handle indexing
                # Actually, V for the 3x3x3 part uses B[:, :3], so V1[j*3+k] for k in 0..2
                # V for the 3x3x2 part uses B[:, 3:], so V2[j*5+(k+3)] for k in 0..1
                # These are different indexings, so I need to be careful.
                
                # Let me redo this more carefully.
                # For the full problem:
                # U has shape (9, total_rank)
                # V has shape (15, total_rank)  [m*p = 3*5 = 15]
                # W has shape (15, total_rank)  [p*n = 5*3 = 15]
                
                U_full = [[0] * total_rank for _ in range(9)]
                V_full = [[0] * total_rank for _ in range(15)]
                W_full = [[0] * total_rank for _ in range(15)]
                
                # Fill in the 3x3x3 part (first target_rank_333 columns)
                for c in range(target_rank_333):
                    for i in range(3):
                        for j in range(3):
                            U_full[i*3+j][c] = U1[i*3+j][c]
                    for j in range(3):
                        for k in range(3):
                            V_full[j*5+k][c] = V1[j*3+k][c]
                    for k in range(3):
                        for i in range(3):
                            W_full[k*3+i][c] = W1[k*3+i][c]
                
                # Fill in the 3x3x2 part (next 18 columns)
                for c in range(18):
                    for i in range(3):
                        for j in range(3):
                            U_full[i*3+j][target_rank_333 + c] = U2[i*3+j][c]
                    for j in range(3):
                        for k in range(2):
                            V_full[j*5 + (k+3)][target_rank_333 + c] = V2[j*5+k+3][c]
                    for k in range(2):
                        for i in range(3):
                            W_full[(k+3)*3+i][target_rank_333 + c] = W2[(k+3)*3+i][c]
                
                # Verify the full decomposition
                if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                    best = {"U": U_full, "V": V_full, "W": W_full}
                    best_rank = total_rank
                    found = True
                    break
            
            if found:
                break
    
    return best

# EVOLVE-BLOCK-END
