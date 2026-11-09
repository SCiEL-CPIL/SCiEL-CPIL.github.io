"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    n, m, p = DIMENSIONS
    
    # Rank-49 decomposition of 4x4x4
    # This is constructed by applying Strassen to 4x4, where 2x2 multiplications use Rank-7 Strassen.
    
    # Step 1: Define Rank-7 decomposition of 2x2x2
    # The 7 Strassen terms for 2x2x2
    # U, V, W are 2x2 matrices for each term
    # Flattened to 4x1 vectors for the factor matrices
    # But we need to track the linear combinations
    
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We represent the factors as 2x2 matrices
    # For each term t, U_t, V_t, W_t are 2x2 matrices
    # Such that sum_t U_t[i,j] V_t[j,k] W_t[k,i] = 1 for all i,j,k in {0,1}
    
    # Actually, it's easier to define the linear combinations directly.
    # Let's define the 7 terms as (U_mat, V_mat, W_mat)
    
    def get_strassen_2x2_factors():
        factors = []
        # Term 1
        # U: A00+A11 -> [[1,0],[0,1]] + [[0,0],[0,1]]? No.
        # A00 is top-left, A11 is bottom-right.
        # U1 = [[1,0],[0,1]]? No, U1 acts on A.
        # M1 = (A00+A11)(B00+B11)
        # U1[i,j] is coeff of A[i,j] in the first factor.
        # First factor is A00+A11.
        # A00 is A[0,0], A11 is A[1,1].
        # So U1[0,0]=1, U1[1,1]=1. Others 0.
        # V1[0,0]=1, V1[1,1]=1. Others 0.
        # W1[0,0]=1, W1[1,1]=1. Others 0.
        
        # Wait, the product is (A00+A11)(B00+B11).
        # This is a 2x2 matrix product? No, A00 is a scalar (entry of A).
        # In the 2x2x2 tensor, the "entries" are scalars.
        # So U1 is a 2x2 matrix with 1s at (0,0) and (1,1).
        
        # Let's verify.
        # Sum U1[i,j] V1[j,k] W1[k,i]
        # i=0, j=0, k=0: U1[0,0]V1[0,0]W1[0,0] = 1*1*1 = 1.
        # i=0, j=1, k=0: U1[0,1]=0.
        # i=1, j=0, k=1: U1[1,0]=0.
        # i=1, j=1, k=1: U1[1,1]V1[1,1]W1[1,1] = 1.
        # But we need 1 for ALL i,j,k.
        # The condition is sum_t U_t[i,j] V_t[j,k] W_t[k,i] = 1.
        
        # My manual check for Term 1 only gives 1 for (0,0,0) and (1,1,1).
        # The sum of all 7 terms must give 1 for all 8 combinations.
        
        # This is the standard Strassen decomposition.
        # I will use the known integer factors for Rank-7 2x2x2.
        
        # U, V, W are 2x2 matrices.
        # I will hardcode the 7 triplets.
        
        # Reference: Strassen's algorithm factors
        # t=0: M1
        # t=1: M2
        # ...
        
        # Let's generate them programmatically to be safe.
        # Define the 7 linear combinations for A, B, and the output C.
        
        # A_combos: list of 7 2x2 matrices (coefficients for A)
        # B_combos: list of 7 2x2 matrices (coefficients for B)
        # C_combos: list of 7 2x2 matrices (coefficients for C)
        
        # M1 = (A00+A11)(B00+B11)
        # A1 = [[1,0],[0,1]]
        # B1 = [[1,0],[0,1]]
        # C1 = [[1,0],[0,1]] (Contributes to C00 and C11)
        
        # M2 = (A10+A11)B00
        # A2 = [[0,0],[1,1]]
        # B2 = [[1,0],[0,0]]
        # C2 = [[0,0],[1,0]] (Contributes to C10)
        
        # M3 = A00(B01-B11)
        # A3 = [[1,0],[0,0]]
        # B3 = [[0,1],[0,-1]]
        # C3 = [[0,1],[0,0]] (Contributes to C01)
        
        # M4 = A11(B10-B00)
        # A4 = [[0,0],[0,1]]
        # B4 = [[-1,0],[1,0]]
        # C4 = [[0,0],[1,0]]? No.
        # C10 = A10B00 + A11B10.
        # M2 gives A11B00? No, M2 = (A10+A11)B00 = A10B00 + A11B00.
        # M4 = A11(B10-B00) = A11B10 - A11B00.
        # Sum M2+M4 = A10B00 + A11B10. Correct.
        # So C2 contributes to C10. C4 contributes to C10.
        # C2 = [[0,0],[1,0]]
        # C4 = [[0,0],[1,0]]
        
        # M5 = (A00+A01)B11
        # A5 = [[1,1],[0,0]]
        # B5 = [[0,0],[0,1]]
        # C5 = [[0,0],[0,1]] (Contributes to C01? No, C01 = A00B01 + A01B11)
        # M3 = A00B01 - A00B11.
        # M5 = A00B11 + A01B11.
        # Sum M3+M5 = A00B01 + A01B11. Correct.
        # C3 = [[0,1],[0,0]]
        # C5 = [[0,1],[0,0]]
        
        # M6 = (A10-A00)(B00+B01)
        # A6 = [[-1,0],[1,0]]
        # B6 = [[1,1],[0,0]]
        # C6 = [[0,0],[1,1]] (Contributes to C10 and C11)
        # C11 = A10B01 + A11B11.
        # M1 = A00B00 + A00B11 + A11B00 + A11B11.
        # M7 = (A01-A11)(B10+B11)
        # A7 = [[0,1],[-1,0]]
        # B7 = [[0,0],[1,1]]
        # C7 = [[1,0],[0,0]] (Contributes to C00 and C01?)
        # C00 = A00B00 + A01B10.
        # M1 gives A00B00.
        # M7 = A01B10 + A01B11 - A11B10 - A11B11.
        # Sum M1+M7 = A00B00 + A01B10. Correct.
        # So C1 = [[1,0],[0,1]]
        # C7 = [[1,0],[0,0]]
        
        # Let's list them:
        A_combos = [
            np.array([[1,0],[0,1]]),
            np.array([[0,0],[1,1]]),
            np.array([[1,0],[0,0]]),
            np.array([[0,0],[0,1]]),
            np.array([[1,1],[0,0]]),
            np.array([[-1,0],[1,0]]),
            np.array([[0,1],[-1,0]])
        ]
        B_combos = [
            np.array([[1,0],[0,1]]),
            np.array([[1,0],[0,0]]),
            np.array([[0,1],[0,-1]]),
            np.array([[-1,0],[1,0]]),
            np.array([[0,0],[0,1]]),
            np.array([[1,1],[0,0]]),
            np.array([[0,0],[1,1]])
        ]
        C_combos = [
            np.array([[1,0],[0,1]]),
            np.array([[0,0],[1,0]]),
            np.array([[0,1],[0,0]]),
            np.array([[0,0],[1,0]]),
            np.array([[0,1],[0,0]]),
            np.array([[0,0],[1,1]]),
            np.array([[1,0],[0,0]])
        ]
        
        # Verify: Sum_t A_t[i,j] B_t[j,k] C_t[k,i] = 1
        # This is the condition.
        
        return A_combos, B_combos, C_combos

    A_2, B_2, C_2 = get_strassen_2x2_factors()
    
    # Now, construct 4x4x4 factors
    # Outer Strassen on 4x4
    # A is 4x4. Blocks are 2x2.
    # A00 = A[0:2, 0:2], etc.
    
    # Outer Strassen combos for 4x4
    # Same structure as 2x2, but applied to blocks.
    
    def get_outer_strassen_combos():
        # Returns list of 7 4x4 matrices for A, B, C
        # Each is a 2x2 block matrix where blocks are 2x2 identity/zero matrices
        # representing the linear combination of the 2x2 blocks of A
        
        A_out = []
        B_out = []
        C_out = []
        
        # M1 = (A00+A11)(B00+B11)
        # A1 = [[I, 0], [0, I]]
        I2 = np.eye(2)
        Z2 = np.zeros((2,2))
        
        A_out.append(np.block([[I2, Z2], [Z2, I2]]))
        B_out.append(np.block([[I2, Z2], [Z2, I2]]))
        C_out.append(np.block([[I2, Z2], [Z2, I2]]))
        
        # M2 = (A10+A11)B00
        A_out.append(np.block([[Z2, Z2], [I2, I2]]))
        B_out.append(np.block([[I2, Z2], [Z2, Z2]]))
        C_out.append(np.block([[Z2, Z2], [I2, Z2]]))
        
        # M3 = A00(B01-B11)
        A_out.append(np.block([[I2, Z2], [Z2, Z2]]))
        B_out.append(np.block([[Z2, I2], [Z2, -I2]]))
        C_out.append(np.block([[Z2, I2], [Z2, Z2]]))
        
        # M4 = A11(B10-B00)
        A_out.append(np.block([[Z2, Z2], [Z2, I2]]))
        B_out.append(np.block([[-I2, Z2], [I2, Z2]]))
        C_out.append(np.block([[Z2, Z2], [I2, Z2]]))
        
        # M5 = (A00+A01)B11
        A_out.append(np.block([[I2, I2], [Z2, Z2]]))
        B_out.append(np.block([[Z2, Z2], [Z2, I2]]))
        C_out.append(np.block([[Z2, I2], [Z2, Z2]]))
        
        # M6 = (A10-A00)(B00+B01)
        A_out.append(np.block([[-I2, Z2], [I2, Z2]]))
        B_out.append(np.block([[I2, I2], [Z2, Z2]]))
        C_out.append(np.block([[Z2, Z2], [I2, I2]]))
        
        # M7 = (A01-A11)(B10+B11)
        A_out.append(np.block([[Z2, I2], [-I2, Z2]]))
        B_out.append(np.block([[Z2, Z2], [I2, I2]]))
        C_out.append(np.block([[I2, Z2], [Z2, Z2]]))
        
        return A_out, B_out, C_out

    A_outer, B_outer, C_outer = get_outer_strassen_combos()
    
    # Construct 49 factors for 4x4x4
    U_44 = []
    V_44 = []
    W_44 = []
    
    for s in range(7):
        # A'_s is a 4x4 matrix (linear combo of A blocks)
        # We need to decompose A'_s B'_s using Inner Strassen
        # A'_s is a 2x2 block matrix of 2x2 matrices
        # Let A'_s = [[A'_s_00, A'_s_01], [A'_s_10, A'_s_11]]
        # Each A'_s_ij is a 2x2 matrix
        
        # Apply Inner Strassen to A'_s, B'_s
        # The Inner Strassen uses the 2x2x2 Rank-7 factors
        # But the "scalars" are 2x2 matrices
        
        # For each inner term t (0..6):
        for t in range(7):
            # The inner Strassen term t corresponds to:
            # M_inner = (A'_s)_t (B'_s)_t
            # Where (A'_s)_t is a 2x2 matrix formed by linear combos of the 2x2 blocks of A'_s
            # Using A_2[t]
            
            # (A'_s)_t = sum_{i,j} A_2[t][i,j] * A'_s_ij
            # A'_s_ij is a 2x2 matrix.
            # So (A'_s)_t is a 2x2 matrix.
            
            # We need to expand this back to 4x4 for U_44
            # U_44 for this term is a 4x4 matrix
            
            # U_44 = sum_{i,j} A_2[t][i,j] * A_outer[s]_ij
            # Where A_outer[s]_ij is the 2x2 block of A_outer[s] at position (i,j)
            
            U_44_mat = np.zeros((4,4))
            for i in range(2):
                for j in range(2):
                    coeff = A_2[t][i,j]
                    if coeff != 0:
                        # A_outer[s] is 4x4. Block (i,j) is A_outer[s][2*i:2*i+2, 2*j:2*j+2]
                        block = A_outer[s][2*i:2*i+2, 2*j:2*j+2]
                        U_44_mat += coeff * block
                        
            V_44_mat = np.zeros((4,4))
            for i in range(2):
                for j in range(2):
                    coeff = B_2[t][i,j]
                    if coeff != 0:
                        block = B_outer[s][2*i:2*i+2, 2*j:2*j+2]
                        V_44_mat += coeff * block
                        
            W_44_mat = np.zeros((4,4))
            for i in range(2):
                for j in range(2):
                    coeff = C_2[t][i,j]
                    if coeff != 0:
                        block = C_outer[s][2*i:2*i+2, 2*j:2*j+2]
                        W_44_mat += coeff * block
                        
            U_44.append(U_44_mat.flatten())
            V_44.append(V_44_mat.flatten())
            W_44.append(W_44_mat.flatten())
            
    # U_44, V_44, W_44 are lists of 49 vectors of length 16
    # Convert to matrices 16x49
    U_44 = np.array(U_44).T
    V_44 = np.array(V_44).T
    W_44 = np.array(W_44).T
    
    # Construct 4x4x8 factors
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First 49 columns
    U[:, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    # Next 49 columns
    U[:, 49:98] = U_44
    V[16:32, 49:98] = V_44
    W[16:32, 49:98] = W_44
    
    # Check time
    if time.time() > deadline:
        # Return schoolbook if we are running out of time (unlikely)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
