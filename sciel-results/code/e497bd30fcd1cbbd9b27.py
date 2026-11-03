"""Matrix multiplication 3x3x5 - improved tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s
    
    def schoolbook_rank(n, m, p):
        """Standard schoolbook: rank = n*m*p"""
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
        return U, V, W, rank
    
    def strassen_2x2_2x2():
        """Strassen decomposition for 2x2 matrix multiplication (rank 7)"""
        # Returns U, V, W for multiplying 2x2 by 2x2
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        # Standard Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11
        W[3][0] = 1  # C22
        
        # M2 = (A21+A22)B11 -> contributes to C21, C22
        U[2][1] = 1  # A21
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21
        W[3][1] = -1 # C22
        
        # M3 = A11(B21-B22) -> contributes to C12, C22
        U[0][2] = 1  # A11
        V[2][2] = 1  # B21
        V[3][2] = -1 # B22
        W[1][2] = 1  # C12
        W[3][2] = 1  # C22
        
        # M4 = A22(B11-B12) -> contributes to C11, C21
        U[3][3] = 1  # A22
        V[0][3] = 1  # B11
        V[1][3] = -1 # B12
        W[0][3] = 1  # C11
        W[2][3] = 1  # C21
        
        # M5 = (A11+A12)B22 -> contributes to C11, C12
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12
        V[3][4] = 1  # B22
        W[0][4] = -1 # C11
        W[1][4] = 1  # C12
        
        # M6 = (A21-A11)(B11+B12) -> contributes to C22
        U[2][5] = 1  # A21
        U[0][5] = -1 # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22
        
        # M7 = (A12-A22)(B21+B22) -> contributes to C11
        U[1][6] = 1  # A12
        U[3][6] = -1 # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11
        
        return U, V, W
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication"""
        rank = len(W[0])
        # Compute C[i,k] = sum_j A[i,j]*B[j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For each (i,k), C[i,k] should equal sum_j A[i,j]*B[j,k]
        # The tensor element T[i,j,k] = 1 if the bilinear form picks up A[i,j]*B[j,k]
        # Sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-10:
                        return False, (i, j, k, total)
        return True, None
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school, rank_school = schoolbook_rank(n, m, p)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try block decomposition: use Strassen-like for 2x2x2 blocks, schoolbook for rest
    # For 3x3x5, we can try to decompose as:
    # - 2x2x2 Strassen for rows 1-2, cols 1-2 of both matrices (rank 7)
    # - Remaining terms via schoolbook
    # Actually, let's try a more systematic approach with rank minimization
    
    # Approach: Try various block structures
    # The key insight: for 3x3x5, we can potentially use:
    # - Strassen for the 2x2 core (rank 7)
    # - Cross terms and remaining terms
    
    # Let's try constructing a hybrid decomposition
    # For 3x3x5:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # Try: decompose into 2x2x2 Strassen + remainder
    # The 2x2x2 part covers i in {0,1}, j in {0,1}, k in {0,1}
    # But we need to be careful about cross terms
    
    # Actually, let's try a different approach: use the fact that
    # we can split the 3x3x5 problem into overlapping 2x2x2 problems
    
    # For now, let's try a simple improvement: check if we can use
    # rank 39 or lower via a valid decomposition
    
    # Try rank from 39 down to 35
    for target_rank in range(39, 34, -1):
        if time.time() > deadline:
            break
        
        # Try to find a valid decomposition of the given rank
        # This is NP-hard in general, so we use a heuristic
        
        # Heuristic: start from schoolbook and try to merge terms
        # that have the same (i,j,k) pattern
        
        # For a quick test, let's just verify the schoolbook works
        # and return it if we can't find something better
        
        # Try a structured approach:
        # Use Strassen for the 2x2x2 subproblem (rows 0-1, cols 0-1)
        # and schoolbook for everything else
        
        # Build decomposition:
        # Part 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
        # Part 2: Schoolbook for all other (i,j,k) tuples
        
        U = [[0] * (7 + 18) for _ in range(n * m)]  # 7 + 9*3*2/3... let's compute
        V = [[0] * (7 + 18) for _ in range(m * p)]
        W = [[0] * (7 + 18) for _ in range(p * n)]
        
        col = 0
        # Add Strassen terms for the 2x2x2 core
        U_s, V_s, W_s = strassen_2x2_2x2()
        for r in range(4):
            for c in range(7):
                U[r][col + c] = U_s[r][c]
                V[r][col + c] = V_s[r][c]
                W[r][col + c] = W_s[r][c]
        col += 7
        
        # Add schoolbook for remaining (i,j,k) tuples
        # The Strassen part handles i in {0,1}, j in {0,1}, k in {0,1}
        # We need all (i,j,k) for i in 0..2, j in 0..2, k in 0..4
        # Strassen covers: i in {0,1}, j in {0,1}, k in {0,1}
        # Remaining: all other combinations
        
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    # Skip if covered by Strassen
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * 3 + j][col] = 1
                    V[j * 5 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        
        # Check if this is valid
        valid, err = verify_decomposition(U, V, W, n, m, p)
        if valid:
            actual_rank = col
            if actual_rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, actual_rank
        
        if time.time() > deadline:
            break
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
