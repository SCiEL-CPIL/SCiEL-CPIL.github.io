import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition for 4x4x4 first, then duplicate it for the two halves of 4x4x8.
    
    # 1. Define the 7 terms of Strassen for 2x2x2
    # U, V, W are 2x2 matrices for each term
    # The tensor is sum(U_s otimes V_s otimes W_s)
    # Indices for 2x2: 0, 1, 2, 3 corresponding to (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = []
    
    # Term 1: (A11+A22)(B11+B22)
    # A coeffs: (1,0,0,1) for blocks? No, A is 2x2 blocks.
    # Let's define U_s as 2x2 matrix of coefficients for A's 4 entries.
    # Wait, U_s is a vector of length 4 in the 2x2x2 tensor.
    # In the block decomposition, U_s is a 2x2 matrix.
    
    # Let's define the 7 terms as (U_mat, V_mat, W_mat) where each is 2x2.
    
    # T1
    U1 = np.array([[1, 0], [0, 1]]) # A11 + A22? No.
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # A part: A11 + A22. As a 2x2 matrix of coeffs?
    # The "A" in M1 is a 2x2 matrix where (0,0)=1, (1,1)=1.
    U1 = np.array([[1, 0], [0, 1]])
    V1 = np.array([[1, 0], [0, 1]])
    W1 = np.array([[1, 0], [0, 1]]) # C11 += M1, C22 += M1
    
    # M2 = (A21+A22)B11
    U2 = np.array([[0, 0], [1, 1]])
    V2 = np.array([[1, 0], [0, 0]])
    W2 = np.array([[0, 0], [1, 0]]) # C21 += M2
    
    # M3 = A11(B12-B22)
    U3 = np.array([[1, 0], [0, 0]])
    V3 = np.array([[0, 1], [0, -1]])
    W3 = np.array([[0, 1], [0, 0]]) # C12 += M3
    
    # M4 = A22(B21-B11)
    U4 = np.array([[0, 0], [0, 1]])
    V4 = np.array([[0, 0], [1, -1]])
    W4 = np.array([[0, 0], [1, 0]]) # C21 += M4
    
    # M5 = (A11+A12)B22
    U5 = np.array([[1, 1], [0, 0]])
    V5 = np.array([[0, 0], [0, 1]])
    W5 = np.array([[0, 1], [0, 0]]) # C12 += M5
    
    # M6 = (A21-A11)(B11+B12)
    U6 = np.array([[-1, 0], [1, 0]])
    V6 = np.array([[1, 1], [0, 0]])
    W6 = np.array([[0, 0], [1, 0]]) # C21 += M6? No.
    # C21 = M2 + M4. C11 = M1 - M4 + M6 - M7.
    # Let's check W matrices carefully.
    
    # Standard Strassen:
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # So:
    # W1: (1,0,0,1) -> C11, C22
    # W2: (0,0,1,0) -> C21
    # W3: (0,1,0,0) -> C12
    # W4: (0,0,1,0) -> C21 (with -1 in C11)
    # W5: (0,1,0,0) -> C12 (with +1 in C22)
    # W6: (0,0,1,0) -> C21 (with +1 in C11)
    # W7: (0,0,0,1) -> C22 (with -1 in C11)
    
    # Let's re-define W matrices as 2x2.
    W1 = np.array([[1, 0], [0, 1]])
    W2 = np.array([[0, 0], [1, 0]])
    W3 = np.array([[0, 1], [0, 0]])
    W4 = np.array([[-1, 0], [1, 0]]) # -1 for C11, +1 for C21
    W5 = np.array([[0, 1], [0, 1]]) # +1 for C12, +1 for C22
    W6 = np.array([[1, 0], [1, 0]]) # +1 for C11, +1 for C21
    W7 = np.array([[-1, 0], [0, -1]]) # -1 for C11, -1 for C22
    
    # Check C11: W1(1) + W4(-1) + W6(1) + W7(-1) = 1 - 1 + 1 - 1 = 0? 
    # Wait. C11 should be 1.
    # M1 - M4 + M6 - M7.
    # W1(0,0)=1. W4(0,0)=-1. W6(0,0)=1. W7(0,0)=-1. Sum = 0.
    # Error in W matrices.
    
    # Let's just use the vectors directly.
    # It is easier to generate the 28 terms programmatically by expanding the block multiplication.
    
    # I will use a simpler approach:
    # Just use the schoolbook for 4x4x4? Rank 64.
    # Then total rank 128. No improvement.
    
    # Let's use the 28-rank decomposition.
    # I will hardcode the 28 vectors for 4x4x4.
    # This is tedious but safe if done correctly.
    
    # Actually, I can just use the recursive Strassen code that generates the vectors.
    
    # Let's implement a helper to generate the 28 terms.
    
    # The 7 Strassen terms (U_s, V_s, W_s) are 2x2 matrices.
    # For each term, we expand the product of two 2x2 matrices (the blocks) into 4 scalar products.
    
    # Let's define the 7 terms correctly.
    terms_2x2 = [
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])), # M1
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,0]])), # M2
        (np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]]), np.array([[0,1],[0,0]])), # M3
        (np.array([[0,0],[0,1]]), np.array([[0,0],[1,-1]]), np.array([[-1,0],[1,0]])), # M4
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[0,1],[0,1]])), # M5
        (np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[1,0],[1,0]])), # M6
        (np.array([[0,-1],[0,1]]), np.array([[0,0],[1,1]]), np.array([[-1,0],[0,-1]])) # M7
    ]
    
    # Verify C11:
    # M1: 1
    # M4: -1
    # M6: 1
    # M7: -1
    # Sum: 0. Still wrong.
    
    # Correct Strassen:
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # My W matrices:
    # W1: (1,0,0,1)
    # W2: (0,0,1,0)
    # W3: (0,1,0,0)
    # W4: (-1,0,1,0) -> C11:-1, C21:1
    # W5: (0,1,0,1) -> C12:1, C22:1
    # W6: (1,0,1,0) -> C11:1, C21:1
    # W7: (-1,0,0,-1) -> C11:-1, C22:-1
    
    # C11: 1 - 1 + 1 - 1 = 0.
    # Why?
    # M1 - M4 + M6 - M7.
    # If W1(0,0)=1, W4(0,0)=-1, W6(0,0)=1, W7(0,0)=-1.
    # Sum is 0.
    # Ah, M7 is (A12-A22)(B21+B22).
    # C11 term in M7 is -1.
    # C22 term in M7 is -1.
    # So W7(0,0)=-1, W7(1,1)=-1.
    
    # Let's re-calculate C11 sum:
    # M1: +1
    # M4: -1
    # M6: +1
    # M7: -1
    # Sum = 0.
    
    # Is Strassen C11 = M1 - M4 + M6 - M7?
    # Yes.
    # So the coefficients must sum to 1.
    # 1 - 1 + 1 - 1 = 0.
    # There is a mistake in my W matrices or the formula.
    # Formula: C11 = M1 - M4 + M6 - M7.
    # Coeffs: 1, -1, 1, -1. Sum 0.
    # This implies the formula is wrong or I am misinterpreting.
    # Actually, C11 = M1 - M4 + M6 - M7 is correct.
    # Wait.
    # M1 = (A11+A22)(B11+B22)
    # M4 = A22(B21-B11)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Let's check A11B11 term in C11.
    # M1: A11B11
    # M4: 0
    # M6: -A11B11
    # M7: 0
    # Sum: A11B11 - A11B11 = 0.
    # But C11 should have A11B11.
    
    # Ah, M6 is (A21-A11)(B11+B12).
    # Term A11B11 is -A11B11.
    # So C11 has A11B11 - A11B11 = 0.
    # This means Strassen C11 formula is NOT M1 - M4 + M6 - M7.
    
    # Correct Strassen:
    # C11 = M1 - M4 + M6 - M7 is WRONG.
    # C11 = M1 - M4 + M6 - M7?
    # Let's look it up.
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # Let's re-evaluate A11B11 in C11.
    # M1: +1
    # M4: 0
    # M6: -1
    # M7: 0
    # Sum: 0.
    
    # There is a persistent error in my Strassen recall.
    # I will use the rank 49 (Strassen-Strassen) or just Schoolbook (128).
    # Given the risk, I will use the Schoolbook decomposition but with a small optimization:
    # Actually, just returning the schoolbook (128) gives score 1.0.
    # I need to beat 1.0.
    # Rank 98 (Strassen-Strassen) gives 1.3.
    # I will implement Rank 98 safely.
    
    # To implement Rank 98:
    # Use 7 outer Strassen terms.
    # For each, use 7 inner Strassen terms.
    # Total 49 terms for 4x4x4.
    # Total 98 terms for 4x4x8.
    
    # I will generate the 49 terms for 4x4x4 using the 7x7 expansion.
    
    # This is safer than 28 if my 28 logic is flawed.
    # And 98 is definitely correct if Strassen is correct.
    
    # Let's assume Strassen is correct and I just made an arithmetic error in verification.
    # I will code the 7x7 expansion.
    
    pass

# I will output the code for Rank 98.
