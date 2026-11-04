"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal decomposition for 2x4 x 4x5 matrix multiplication.
    
    Strategy: Build a rank-36 decomposition using Strassen for 2x2x2 blocks
    and schoolbook for 2x2x1 blocks, then attempt to reduce the rank by
    finding linear dependencies among the rank-1 terms.
    
    The multiplication tensor T[i,j,k] = delta(i,k) for the (2,4,5) case
    has a natural block structure that allows Strassen's algorithm on
    2x2x2 sub-problems.
    """
    t0 = time.time()
    DEADLINE = 140  # seconds, leave 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Build the known rank-36 solution
    rank36 = 36
    U36 = np.zeros((n * m, rank36))
    V36 = np.zeros((m * p, rank36))
    W36 = np.zeros((p * n, rank36))
    
    def add_strassen_222(U, V, W, i1, i2, j1, j2, k1, k2, col_offset):
        """Add Strassen 2x2x2 decomposition for rows i1,i2 of A, cols j1,j2 of A,
           rows j1,j2 of B, cols k1,k2 of B."""
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
        
        # M1 = (A11+A22)(B11+B22)
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        
        # M2 = (A21+A22)B11
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        
        # M3 = A11(B12-B22)
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        # M4 = A22(B21-B11)
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        
        # M5 = (A11+A12)B22
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        
        # M6 = (A21-A11)(B11+B12)
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        
        # M7 = (A12-A22)(B21+B22)
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1
    
    def add_schoolbook_221(U, V, W, i1, i2, j1, j2, k1, col_offset):
        """Add schoolbook 2x2x1 decomposition."""
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
    
    # Verify rank-36 is correct using random test
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test
    
    def compute_C(U, V, W, A, B):
        """Compute C from decomposition."""
        r = U.shape[1]
        C = np.zeros((n, p))
        for c in range(r):
            scalar = np.dot(U[:, c], A.flatten()) * np.dot(V[:, c], B.flatten())
            C += scalar * W[:, c].reshape((p, n)).T
        return C
    
    C_decomp = compute_C(U36, V36, W36, A_test, B_test)
    assert np.allclose(C_decomp, C_expected, atol=1e-10), "Rank-36 verification failed"
    
    # Attempt rank reduction by finding linear dependencies
    # The key insight: if the rank-36 decomposition has any redundancy,
    # we can remove terms and get rank 35 or lower.
    
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 20:
        try:
            # Try removing each individual term to see if it's redundant
            for remove_col in range(rank36):
                if time.time() - t0 > DEADLINE - 10:
                    break
                U_red = np.delete(U36, remove_col, axis=1)
                V_red = np.delete(V36, remove_col, axis=1)
                W_red = np.delete(W36, remove_col, axis=1)
                
                C_red = compute_C(U_red, V_red, W_red, A_test, B_test)
                
                if np.allclose(C_red, C_expected, atol=1e-10):
                    # Found a redundant term!
                    return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
            
            # Try removing pairs of terms
            for remove_col1 in range(rank36):
                if time.time() - t0 > DEADLINE - 15:
                    break
                for remove_col2 in range(remove_col1 + 1, rank36):
                    if time.time() - t0 > DEADLINE - 10:
                        break
                    
                    mask = np.ones(rank36, dtype=bool)
                    mask[remove_col1] = False
                    mask[remove_col2] = False
                    
                    U_red = U36[:, mask]
                    V_red = V36[:, mask]
                    W_red = W36[:, mask]
                    
                    C_red = compute_C(U_red, V_red, W_red, A_test, B_test)
                    
                    if np.allclose(C_red, C_expected, atol=1e-10):
                        return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
            
            # Try a more systematic approach: use SVD to find the numerical rank
            # of the tensor and see if we can compress
            # Flatten the tensor and check its rank structure
            
            # Alternative: try to find a linear combination of terms that cancels out
            # This is equivalent to finding a null vector in the space of decompositions
            
            # For now, if no reduction found, return rank-36
        except Exception:
            pass
    
    # Return rank-36 solution
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
# EVOLVE-BLOCK-END
