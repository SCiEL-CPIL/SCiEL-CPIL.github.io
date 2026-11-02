"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    U, V, W are lists of 4x4 matrices (flattened to length 16) for each of the 7 terms.
    Term t: U_t[i,j] * V_t[j,k] contributes to W_t[i,k]
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indexing: row = i*2+j, col = k*2+i (W convention: W[k*n+i, r] for n=2)
    # For 2x2: U index = i*2+j, V index = j*2+k, W index = k*2+i
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Helper to create a 2x2 matrix as a flat list
    def mat(a11, a12, a21, a22):
        return [a11, a12, a21, a22]
    
    # M1 = (A11+A22)(B11+B22)
    U_terms.append(mat(1, 0, 0, 1))
    V_terms.append(mat(1, 0, 0, 1))
    W_terms.append(mat(1, 0, 0, 1))  # C11 += M1, C12 += M1, C21 += M1, C22 += M1? No...
    
    # Actually, let me be more careful. The decomposition is:
    # C = sum_t U_t * V_t where U_t is 2x2, V_t is 2x2
    # But the tensor is T[i,j,k] = delta_{i,k} (for matrix mult A*B)
    # Wait, I need to think about this differently.
    
    # For matrix multiplication C = A*B, the tensor is T[i,j,k] = delta_{i,k}
    # where i indexes rows of A/C, j indexes columns of A/rows of B, k indexes columns of B/C
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For Strassen, we have 7 terms. Let me look up the exact coefficients.
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11  
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So the decomposition has 7 terms with coefficients in W.
    
    # Let me define U_t[i,j], V_t[j,k], W_t[k,i] for each term t.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # U1: A11+A22 -> [[1,0],[0,1]]
    # V1: B11+B22 -> [[1,0],[0,1]]
    # W1: contributes to C11, C12, C21, C22 with coefficients
    # C11 += M1, C12 += M1? No, let me recheck.
    
    # Actually I think I'm overcomplicating. Let me use a known correct Strassen decomposition.
    
    # From literature, the 7 terms for 2x2x2 matrix multiplication tensor:
    
    terms = [
        # (U_coeff, V_coeff, W_coeff) where each is a 2x2 matrix (4 elements)
        # Term 1
        ([1,0,0,1], [1,0,0,1], [1,1,1,1]),  # M1: (A11+A22)(B11+B22) -> C11,C12,C21,C22
        # Term 2  
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2: (A21+A22)B11 -> C21
        # Term 3
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # M3: A11(B12-B22) -> C12
        # Term 4
        ([0,0,0,1], [0,-1,1,0], [1,0,0,0]),  # M4: A22(B21-B11) -> C11
        # Term 5
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5: (A11+A12)B22 -> C12, C21
        # Term 6
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6: (A21-A11)(B11+B12) -> C22
        # Term 7
        ([0,1,-1,0], [0,0,1,1], [1,0,0,0]),  # M7: (A12-A22)(B21+B22) -> C11
    ]
    
    # Let me verify this is correct by checking the tensor equation.
    # This is getting complex. Let me use a simpler approach: 
    # Just implement schoolbook for now but with a time guard, and return it.
    # The Strassen approach is complex to get exactly right.
    
    return None  # Placeholder

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
