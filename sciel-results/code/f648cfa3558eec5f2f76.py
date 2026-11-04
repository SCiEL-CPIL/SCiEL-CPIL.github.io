import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_4x4x4():
    """
    Construct 4x4x4 decomposition using Strassen recursively.
    Returns U, V, W of shape (16, 49).
    """
    U2, V2, W2 = _strassen_2x2()
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Mappings for the 7 Strassen products
    # Each product s has a mapping from 2x2 indices (i', j', k') to 4x4 indices (i, j, k)
    # and coefficients.
    
    # s=0: M1 = (A00+A11)(B00+B11)
    # A part: A[2i'+a, 2j'+a] for a in {0,1}
    # B part: B[2j'+b, 2k'+b] for b in {0,1}
    # This is a sum of 4 terms.
    # We can't just map (i',j',k') to (i,j,k) directly because of the sums.
    # However, the rank-7 decomposition of 2x2x2 handles the summation over j'.
    # The "A" and "B" matrices in the 2x2x2 decomposition are X and Y.
    # X[i', j'] = sum_a A[2i'+a, 2j'+a]
    # Y[j', k'] = sum_b B[2j'+b, 2k'+b]
    
    # This means U4, V4, W4 will have contributions from multiple (i,j) pairs.
    # For a fixed s and c (from 2x2x2):
    # U2[i', j', c] contributes to U4[i, j, s*7+c] for all i, j such that i' = i//2, j' = j//2?
    # No, X[i', j'] is a linear combination of A entries.
    # So U4[i, j, s*7+c] should be the coefficient of A[i, j] in X[i', j'].
    
    # Let's define the coefficient functions.
    def get_coeff_A(s, i_prime, j_prime, i, j):
        # Returns coeff of A[i, j] in X_s[i_prime, j_prime]
        if s == 0: # M1
            if i // 2 == i_prime and j // 2 == j_prime and (i % 2) == (j % 2):
                return 1
            if i // 2 == i_prime + 1 and j // 2 == j_prime + 1 and (i % 2) == (j % 2):
                return 1
            return 0
        elif s == 1: # M2
            if i // 2 == i_prime + 1 and j // 2 == j_prime and (i % 2) == (j % 2):
                return 1
            if i // 2 == i_prime + 1 and j // 2 == j_prime + 1 and (i % 2) == (j % 2):
                return 1
            return 0
        elif s == 2: # M3
            if i // 2 == i_prime and j // 2 == j_prime and (i % 2) == (j % 2):
                return 1
            return 0
        elif s == 3: # M4
            if i // 2 == i_prime + 1 and j // 2 == j_prime + 1 and (i % 2) == (j % 2):
                return 1
            return 0
        elif s == 4: # M5
            if i // 2 == i_prime and j // 2 == j_prime and (i % 2) == (j % 2):
                return 1
            if i // 2 == i_prime and j // 2 == j_prime + 1 and (i % 2) == (j % 2):
                return 1
            return 0
        elif s == 5: # M6
            if i // 2 == i_prime + 1 and j // 2 == j_prime and (i % 2) == (j % 2):
                return 1
            if i // 2 == i_prime and j // 2 == j_prime and (i % 2) == (j % 2):
                return -1
            return 0
        elif s == 6: # M7
            if i // 2 == i_prime and j // 2 == j_prime + 1 and (i % 2) == (j % 2):
                return 1
            if i // 2 == i_prime + 1 and j // 2 == j_prime + 1 and (i % 2) == (j % 2):
                return -1
            return 0
        return 0

    def get_coeff_B(s, j_prime, k_prime, j, k):
        # Returns coeff of B[j, k] in Y_s[j_prime, k_prime]
        if s == 0: # M1
            if j // 2 == j_prime and k // 2 == k_prime and (j % 2) == (k % 2):
                return 1
            if j // 2 == j_prime + 1 and k // 2 == k_prime + 1 and (j % 2) == (k % 2):
                return 1
            return 0
        elif s == 1: # M2
            if j // 2 == j_prime and k // 2 == k_prime and (j % 2) == (k % 2):
                return 1
            return 0
        elif s == 2: # M3
            if j // 2 == j_prime + 1 and k // 2 == k_prime and (j % 2) == (k % 2):
                return 1
            if j // 2 == j_prime + 1 and k // 2 == k_prime + 1 and (j % 2) == (k % 2):
                return -1
            return 0
        elif s == 3: # M4
            if j // 2 == j_prime and k // 2 == k_prime + 1 and (j % 2) == (k % 2):
                return 1
            if j // 2 == j_prime and k // 2 == k_prime and (j % 2) == (k % 2):
                return -1
            return 0
        elif s == 4: # M5
            if j // 2 == j_prime + 1 and k // 2 == k_prime + 1 and (j % 2) == (k % 2):
                return 1
            return 0
        elif s == 5: # M6
            if j // 2 == j_prime and k // 2 == k_prime and (j % 2) == (k % 2):
                return 1
            if j // 2 == j_prime and k // 2 == k_prime + 1 and (j % 2) == (k % 2):
                return 1
            return 0
        elif s == 6: # M7
            if j // 2 == j_prime + 1 and k // 2 == k_prime and (j % 2) == (k % 2):
                return 1
            if j // 2 == j_prime + 1 and k // 2 == k_prime + 1 and (j % 2) == (k % 2):
                return 1
            return 0
        return 0

    # W mapping is direct: k', i' -> k, i
    # But we need to map the 2x2 result indices to 4x4.
    # The result M_s is a 2x2 matrix.
    # It contributes to C blocks.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for a given s, and 2x2 indices i', k', we need to add to W4[k, i, s*7+c]
    # the coefficient from the C block formula.
    
    def get_coeff_W(s, i_prime, k_prime, i, k):
        # i, k are 4x4 indices. i' = i//2, k' = k//2?
        # No, the 2x2 result M_s[i', k'] is placed in the 4x4 C matrix.
        # M_s[0,0] -> C[0:2, 0:2]
        # M_s[0,1] -> C[0:2, 2:4]
        # M_s[1,0] -> C[2:4, 0:2]
        # M_s[1,1] -> C[2:4, 2:4]
        
        # So i' = i // 2, k' = k // 2.
        # And the offset within the block is (i%2, k%2).
        # But M_s is a 2x2 matrix, so it doesn't have sub-block structure.
        # Wait, M_s is the product of 2x2 matrices.
        # So M_s[i', k'] is a single scalar.
        # It contributes to C[i, k] where i = 2*i' + alpha, k = 2*k' + beta?
        # No, the Strassen algorithm computes the 2x2 blocks of C.
        # C_00 = M1 + ...
        # C_00 is a 2x2 matrix.
        # So M1[0,0] contributes to C[0,0] and C[1,1]?
        # No, M1 is a 2x2 matrix.
        # M1[0,0] is the top-left element of the 2x2 matrix M1.
        # It contributes to the top-left element of C_00.
        # So M1[i', k'] contributes to C[2*i', 2*k']?
        # And also C[2*i'+1, 2*k'+1]?
        # No, the 2x2 matrix M1 is added to the 2x2 block C_00.
        # So M1[i', k'] contributes to C[2*i' + i'', 2*k' + k''] for i'', k'' in {0,1}?
        # No, that would mean M1 is a 4x4 matrix.
        # But M1 is 2x2.
        
        # Ah, the Strassen algorithm for 4x4 uses 2x2 blocks.
        # The 7 products are 2x2 matrices.
        # The final C is 4x4.
        # C_00 (2x2) = M1 + M4 - M5 + M7.
        # So each element of M1 contributes to the corresponding element of C_00.
        # M1[0,0] -> C[0,0].
        # M1[0,1] -> C[0,1].
        # M1[1,0] -> C[1,0].
        # M1[1,1] -> C[1,1].
        
        # So the mapping is i = 2*i', k = 2*k'.
        # And the coefficient is from the C block formula.
        
        if i // 2 != i_prime or k // 2 != k_prime or i % 2 != 0 or k % 2 != 0:
            return 0
            
        if s == 0: # M1
            if i_prime == 0 and k_prime == 0: return 1 # C00
            if i_prime == 1 and k_prime == 1: return 1 # C11
            return 0
        elif s == 1: # M2
            if i_prime == 1 and k_prime == 0: return 1 # C10
            if i_prime == 1 and k_prime == 1: return -1 # C11
            return 0
        elif s == 2: # M3
            if i_prime == 0 and k_prime == 1: return 1 # C01
            if i_prime == 1 and k_prime == 1: return 1 # C11
            return 0
        elif s == 3: # M4
            if i_prime == 0 and k_prime == 0: return 1 # C00
            if i_prime == 1 and k_prime == 0: return 1 # C10
            return 0
        elif s == 4: # M5
            if i_prime == 0 and k_prime == 0: return -1 # C00
            if i_prime == 0 and k_prime == 1: return 1 # C01
            return 0
        elif s == 5: # M6
            if i_prime == 1 and k_prime == 1: return 1 # C11
            return 0
        elif s == 6: # M7
            if i_prime == 0 and k_prime == 0: return 1 # C00
            return 0
        return 0

    # Fill U4, V4, W4
    for s in range(7):
        for c in range(7):
            col = s * 7 + c
            for i_prime in range(2):
                for j_prime in range(2):
                    for k_prime in range(2):
                        u_val = U2[i_prime * 2 + j_prime, c]
                        v_val = V2[j_prime * 2 + k_prime, c]
                        w_val = W2[k_prime * 2 + i_prime, c]
                        
                        if u_val != 0:
                            for i in range(4):
                                for j in range(4):
                                    coeff = get_coeff_A(s, i_prime, j_prime, i, j)
                                    if coeff != 0:
                                        U4[i * 4 + j, col] += u_val * coeff
                                        
                        if v_val != 0:
                            for j in range(4):
                                for k in range(4):
                                    coeff = get_coeff_B(s, j_prime, k_prime, j, k)
                                    if coeff != 0:
                                        V4[j * 4 + k, col] += v_val * coeff
                                        
                        if w_val != 0:
                            for i in range(4):
                                for k in range(4):
                                    coeff = get_coeff_W(s, i_prime, k_prime, i, k)
                                    if coeff != 0:
                                        W4[k * 4 + i, col] += w_val * coeff
                                        
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Use 4x4x4 Strassen (rank 49) + schoolbook for rest (rank 16+16+4+16+4+4+1 = 61)
    # Total rank 110.
    
    U4, V4, W4 = _construct_4x4x4()
    
    U = np.zeros((25, 110))
    V = np.zeros((25, 110))
    W = np.zeros((25, 110))
    
    # Copy 4x4x4 part (indices 0..3)
    U[:16, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    # Add schoolbook for the remaining blocks
    # A12 (4x1), B21 (1x4) -> C11 (4x4) part? No, C11 is A11B11 + A12B21.
    # A11B11 is in U4.
    # A12B21: i in 0..3, j=4, k in 0..3.
    # Rank 16.
    col = 49
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 5 + i, col] = 1
            col += 1
            
    # A11B12: i in 0..3, j in 0..3, k=4.
    # Rank 16.
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 5 + i, col] = 1
            col += 1
            
    # A12B22: i in 0..3, j=4, k=4.
    # Rank 4.
    for i in range(4):
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 5 + i, col] = 1
        col += 1
        
    # A21B11: i=4, j in 0..3, k in 0..3.
    # Rank 16.
    for j in range(4):
        for k in range(4):
            U[4 * 5 + j, col] = 1
            V[j * 5 + k, col] = 1
            W[k * 5 + 4, col] = 1
            col += 1
            
    # A11B14? No, A21B12: i=4, j in 0..3, k=4.
    # Rank 4.
    for j in range(4):
        U[4 * 5 + j, col] = 1
        V[j * 5 + 4, col] = 1
        W[4 * 5 + 4, col] = 1
        col += 1
        
    # A22B21: i=4, j=4, k in 0..3.
    # Rank 4.
    for k in range(4):
        U[4 * 5 + 4, col] = 1
        V[4 * 5 + k, col] = 1
        W[k * 5 + 4, col] = 1
        col += 1
        
    # A22B22: i=4, j=4, k=4.
    # Rank 1.
    U[4 * 5 + 4, col] = 1
    V[4 * 5 + 4, col] = 1
    W[4 * 5 + 4, col] = 1
    col += 1
    
    # Total rank 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
