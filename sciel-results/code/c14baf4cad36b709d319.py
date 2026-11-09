import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Use the known rank-19 decomposition for 3x3 matrix multiplication
    # This is a well-known result (Landsberg, Strassen, etc.)
    # We'll use a concrete rank-19 decomposition for 3x3 * 3x3
    # 
    # For 3x3 * 3x7, we can think of the 7 columns as:
    # - First 6 columns: two 3x3 blocks (but we only have 3 columns per block)
    # - Actually, 7 = 3 + 3 + 1, so we can decompose into:
    #   - 3x3 * 3x3 (rank 19)
    #   - 3x3 * 3x3 (rank 19)  
    #   - 3x3 * 3x1 (rank 9, since it's just matrix-vector)
    # Total: 19 + 19 + 9 = 47
    #
    # But wait, that's not quite right. Let me think more carefully.
    #
    # The multiplication is (3x3) * (3x7) -> (3x7)
    # We can write the 3x7 matrix as three column blocks:
    #   B = [B1, B2, B3] where B1 is 3x3, B2 is 3x3, B3 is 3x1
    # Then A*B = [A*B1, A*B2, A*B3]
    #
    # Rank of A*B1: 19 (3x3*3x3)
    # Rank of A*B2: 19 (3x3*3x3)
    # Rank of A*B3: 9  (3x3*3x1, which is just matrix-vector, rank = n*m = 9)
    #
    # Total rank: 19 + 19 + 9 = 47
    #
    # But we can do better! The three multiplications share the same A,
    # so we might be able to share some factors.
    #
    # Actually, let's just use the schoolbook approach first to get a valid
    # decomposition, then try to optimize.
    
    # Schoolbook: rank = 3*3*7 = 63
    rank_schoolbook = n * m * p
    
    # Let's construct the schoolbook decomposition as a baseline
    U = np.zeros((n * m, rank_schoolbook), dtype=int)
    V = np.zeros((m * p, rank_schoolbook), dtype=int)
    W = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Now let's try to use the structured approach
    # Use the known rank-19 decomposition for 3x3 matrix multiplication
    # I'll use a concrete one from the literature
    
    # Known rank-19 decomposition for 3x3 * 3x3
    # (This is one of the known minimal decompositions)
    # The decomposition has 19 terms
    
    # Let me use a specific rank-19 decomposition
    # Reference: https://en.wikipedia.org/wiki/Matrix_multiplication#3%C3%973_matrix_multiplication
    
    # I'll use the decomposition from:
    # "On the rank of the matrix multiplication tensor" 
    # The 19-term decomposition for 3x3
    
    # Here's a concrete rank-19 decomposition for 3x3 * 3x3:
    # Each term is (u, v, w) where u, v are 3x3 matrices (flattened to 9-vectors)
    # and w is a 3x3 matrix (flattened to 9-vectors)
    
    # Let me define the 19 terms. I'll use a known decomposition.
    # Actually, let me just try to find a good decomposition empirically.
    
    # For now, let's try a different approach: use the structure to reduce rank
    
    # The key insight: 7 = 3 + 3 + 1
    # We can write the multiplication as:
    # C = A * B where B = [B1 | B2 | B3]
    # C = [A*B1 | A*B2 | A*B3]
    
    # For the first two blocks (3x3 each), we can use the rank-19 decomposition
    # For the last block (3x1), we use the rank-9 decomposition
    
    # But we need to be careful about how we combine them.
    
    # Let's construct the decomposition step by step
    
    # First, get the rank-19 decomposition for 3x3 * 3x3
    # I'll use a known one
    
    # Here's a rank-19 decomposition for 3x3 matrix multiplication
    # (one of the known minimal decompositions)
    
    # The 19 terms (u_r, v_r, w_r) for r = 1..19
    # where u_r, v_r, w_r are 9-dimensional vectors (flattened 3x3 matrices)
    
    # Let me use a specific decomposition from the literature
    # I'll hardcode a known rank-19 decomposition
    
    # Actually, let me try a simpler approach first:
    # Use the schoolbook decomposition but try to eliminate redundant columns
    
    # Start with schoolbook
    U_current = U.copy()
    V_current = V.copy()
    W_current = W.copy()
    rank_current = rank_schoolbook
    
    # Try to eliminate columns that are linearly dependent
    # A column r is redundant if removing it still gives an exact decomposition
    
    # For each column, check if it's needed
    # This is expensive, so let's do it smartly
    
    # First, let's verify the schoolbook decomposition is correct
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r]
    
    # Let me verify: for schoolbook, each column r corresponds to (i,j,k)
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Then sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    # = sum_r [r corresponds to (i,j,k)] * 1 * 1 * 1
    # = 1 if there exists r with (i,j,k) = (i_r, j_r, k_r)
    # Since each (i,j,k) appears exactly once, this equals 1
    # But we need sum_j A[i,j] * B[j,k], not just 1!
    
    # I see the issue. The decomposition should be:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 (the multiplication tensor)
    # And the actual multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k] = sum_j A[i,j] * B[j,k]
    
    # So the tensor T is just the identity in the j index for each (i,k)
    # T[i,j,k] = 1 for all i,j,k
    
    # Wait, that's not right either. Let me reconsider.
    
    # The matrix multiplication tensor T_{n,m,p} is defined by:
    # T[i, j, k] = 1 for all i, j, k
    # And the multiplication is:
    # C[i, k] = sum_j A[i, j] * B[j, k] * T[i, j, k]
    
    # So T is just the all-ones tensor!
    
    # The tensor decomposition is:
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    # For schoolbook:
    # Each r corresponds to a unique (i, j, k)
    # U[i, j, r] = 1 if r = (i,j,k), else 0
    # V[j, k, r] = 1 if r = (i,j,k), else 0
    # W[k, i, r] = 1 if r = (i,j,k), else 0
    # Then sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 (since exactly one r matches)
    
    # So the schoolbook decomposition is correct!
    
    # Now, can we do better?
    # The rank of the 3x3*3x7 multiplication tensor
    
    # Known results:
    # - 3x3*3x3 has rank 19
    # - 3x3*3x1 has rank 9 (it's just matrix-vector multiplication)
    # - 3x3*3x7: we can write 7 = 3 + 3 + 1
    #   So rank <= 19 + 19 + 9 = 47
    
    # But we might be able to do better by sharing factors.
    
    # Let's try to construct a rank-47 decomposition
    
    # First, get a rank-19 decomposition for 3x3*3x3
    # I'll use a known one
    
    # Here's a rank-19 decomposition for 3x3 matrix multiplication
    # (from the literature)
    
    # The 19 terms are indexed by r = 0..18
    # Each term has u_r, v_r, w_r which are 9-dimensional vectors
    
    # Let me use a specific decomposition. I'll hardcode one.
    
    # Actually, let me try a different approach. Let me use the fact that
    # we can decompose the 7 columns into blocks and use known decompositions.
    
    # For now, let's just return the schoolbook decomposition and try to
    # optimize within the time budget.
    
    # Try to reduce rank by checking for linear dependencies
    
    # For each column, check if it's linearly dependent on the others
    # This is O(rank^3) per column, which is 63^3 = 250k operations per column
    # Total: 63 * 250k = 15.75M operations, which should be fast
    
    # But we need to be careful: removing a column changes the decomposition
    
    # A better approach: check if the column is in the span of the others
    # in the appropriate tensor product space
    
    # Actually, for tensor decompositions, the condition is more subtle.
    # A column r is redundant if the tensor T can still be expressed
    # without that column.
    
    # Let's try a greedy approach: for each column, try to remove it and
    # check if the decomposition is still exact
    
    # To check exactness, we need to verify:
    # T[i,j,k] = sum_{r != r0} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # for all i, j, k
    
    # This is O(n*m*p*rank) = 3*3*7*62 = 3822 operations per column
    # Total: 63 * 3822 = 240k operations, very fast
    
    # Let's do this
    
    U_test = U_current.copy()
    V_test = V_current.copy()
    W_test = W_current.copy()
    
    # Try to remove columns
    # We'll do this iteratively
    for col in range(rank_current - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        
        # Try removing column col
        # Check if the decomposition is still exact without it
        
        exact = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_{r != col} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = 0
                    for r in range(rank_current):
                        if r != col:
                            s += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                    if s != 1:  # T[i,j,k] should be 1
                        exact = False
                        break
                if not exact:
                    break
            if not exact:
                break
        
        if not exact:
            # Column col is needed, keep it
            continue
        
        # Column col is redundant, remove it
        # Shift columns to the right
        for r in range(col, rank_current - 1):
            U_test[:, r] = U_test[:, r+1]
            V_test[:, r] = V_test[:, r+1]
            W_test[:, r] = W_test[:, r+1]
        rank_current -= 1
    
    # Trim the arrays
    U_final = U_test[:, :rank_current].copy()
    V_final = V_test[:, :rank_current].copy()
    W_final = W_test[:, :rank_current].copy()
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
