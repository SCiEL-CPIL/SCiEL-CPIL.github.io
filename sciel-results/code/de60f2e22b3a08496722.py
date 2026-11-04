"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal decomposition for 2x4 x 4x5 matrix multiplication.
    
    Strategy: Use a rank-35 decomposition by exploiting shared structure.
    The key insight: for n=2, each output column k of C is:
      C[0,k] = sum_j A[0,j]*B[j,k]
      C[1,k] = sum_j A[1,j]*B[j,k]
    
    We can share multiplications across output columns by grouping B's columns.
    Using Strassen-like tricks on the 2-row structure combined with column grouping.
    
    Approach: Split the 5 B-columns into groups, use rank-7 Strassen for pairs
    and handle the remainder. But we can do better by using a 2x4x3 subproblem
    with rank 11 (vs schoolbook 12) + 2x4x2 with rank 7 (Strassen) = 18 per... 
    Actually let's try: 2x4x5 = 2x4x3 + 2x4x2.
    2x4x3: schoolbook is 24. Can we do better?
    2x4x2: schoolbook is 16. Strassen on 2x2x2 blocks: 2*7=14.
    
    Let me try a different split: 2x4x5 into four 2x2x? blocks won't work cleanly.
    
    New approach: Use the fact that we only have 2 rows. The tensor is:
    T[i,j,k] = delta_{i,j} for the multiplication... no.
    
    The multiplication tensor for (n,m,p) = (2,4,5) has entries:
    T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    
    So T is a 2x4x5 tensor where T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    Actually the tensor is: for each (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    
    The structure: U has 8 entries (2*4), V has 20 entries (4*5), W has 10 entries (5*2).
    
    Let me try rank 35: use Strassen for 4 of the 5 output columns (2 pairs) = 2*7=14,
    and schoolbook for the 5th column = 2*4=8. Total = 22? No that's too optimistic.
    
    Actually each Strassen 2x2x2 handles 2 rows of A, 2 cols of A, 2 rows of B, 2 cols of B.
    For our case: 2 rows of A, 4 cols of A, 4 rows of B, 5 cols of B.
    
    Split A's 4 cols into 2 groups of 2, B's 4 rows into 2 groups of 2, B's 5 cols into 2+3.
    
    For A_cols {0,1} x B_rows {0,1} x B_cols {0,1}: Strassen rank 7
    For A_cols {0,1} x B_rows {0,1} x B_cols {2,3}: Strassen rank 7  
    For A_cols {0,1} x B_rows {0,1} x B_cols {4}: schoolbook 2x2x1 rank 4
    For A_cols {2,3} x B_rows {2,3} x B_cols {0,1}: Strassen rank 7
    For A_cols {2,3} x B_rows {2,3} x B_cols {2,3}: Strassen rank 7
    For A_cols {2,3} x B_rows {2,3} x B_cols {4}: schoolbook 2x2x1 rank 4
    Total: 7+7+4+7+7+4 = 36 (same as before)
    
    To get below 36, I need cross-term sharing. Let me try a completely different approach:
    direct enumeration of a rank-35 solution using the structure that n=2 gives more freedom.
    
    With n=2, C has 10 entries. Each rank-1 term contributes to at most 2 entries (one per row).
    So we need at least 10 rank-1 terms just to cover all outputs, but they can overlap.
    
    Let me try constructing a rank-35 solution by modifying the rank-36 one to merge one pair.
    """
    t0 = time.time()
    DEADLINE = 140  # seconds, leave 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # First, build the known rank-36 solution, then try to reduce it
    rank36 = 36
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    
    def add_strassen_222(U, V, W, i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1
    
    def add_schoolbook_221(U, V, W, i1, i2, j1, j2, k1, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1
        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1
        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1
    
    col = 0
    add_strassen_222(U36, V36, W36, 0, 1, 0, 1, 0, 1, col); col += 7
    add_strassen_222(U36, V36, W36, 0, 1, 0, 1, 2, 3, col); col += 7
    add_schoolbook_221(U36, V36, W36, 0, 1, 0, 1, 4, col); col += 4
    add_strassen_222(U36, V36, W36, 0, 1, 2, 3, 0, 1, col); col += 7
    add_strassen_222(U36, V36, W36, 0, 1, 2, 3, 2, 3, col); col += 7
    add_schoolbook_221(U36, V36, W36, 0, 1, 2, 3, 4, col); col += 4
    
    # Verify rank-36 is correct
    U36_np = np.array(U36, dtype=float)
    V36_np = np.array(V36, dtype=float)
    W36_np = np.array(W36, dtype=float)
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test
    C_decomp = np.zeros((n, p))
    for c in range(rank36):
        scalar = U36_np[:, c] @ A_test.flatten() * V36_np[:, c] @ B_test.flatten()
        C_decomp += scalar * W36_np[:, c].reshape((p, n)).T
    assert np.allclose(C_decomp, C_expected, atol=1e-10), "Rank-36 verification failed"
    
    # Now try to find a rank-35 solution by attempting to eliminate one dimension
    # via a linear combination of existing terms
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 20:
        try:
            # Try to find if any column is a linear combination of others
            # This would indicate the rank can be reduced
            # Use SVD to check numerical rank
            # Stack all tensor components
            # The tensor T has components T[i,j,k] for the multiplication
            # For rank-r decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
            
            # Check if the rank-36 solution has any redundancy
            # by checking if removing any single term still gives a valid decomposition
            for remove_col in range(rank36):
                if time.time() - t0 > DEADLINE - 10:
                    break
                U_red = np.delete(U36_np, remove_col, axis=1)
                V_red = np.delete(V36_np, remove_col, axis=1)
                W_red = np.delete(W36_np, remove_col, axis=1)
                
                C_red = np.zeros((n, p))
                for c in range(rank36 - 1):
                    scalar = U_red[:, c] @ A_test.flatten() * V_red[:, c] @ B_test.flatten()
                    C_red += scalar * W_red[:, c].reshape((p, n)).T
                
                if np.allclose(C_red, C_expected, atol=1e-10):
                    # Found a redundant term!
                    U_final = U_red.tolist()
                    V_final = V_red.tolist()
                    W_final = W_red.tolist()
                    return {"U": U_final, "V": V_final, "W": W_final}
        except Exception:
            pass
    
    # If no reduction found, return rank-36
    return {"U": U36, "V": V36, "W": W36}
# EVOLVE-BLOCK-END
