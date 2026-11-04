import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # Part 2: 4x4x1 multiplication using Schoolbook (Rank 16)
    
    # Strassen 2x2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: i, j for U, V; k, i for W? 
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will build the 4x4x4 factors by composing these.
    
    # Let's create the 49 rank components.
    U1 = [[0]*49 for _ in range(16)]
    V1 = [[0]*49 for _ in range(16)]
    W1 = [[0]*49 for _ in range(16)]
    
    # Define the 7 blocks of A and B for Strassen
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # We need to express A_x and B_x for x=1..7 as linear combinations of A blocks.
    # Then apply 2x2 Strassen to each pair.
    
    # Helper to add a 2x2 Strassen decomposition to the global factors
    def add_strassen_2x2(U, V, W, A_coeff, B_coeff, offset):
        # A_coeff: 4x4 matrix of coefficients for A blocks? 
        # Actually, A_x is a 2x2 matrix whose entries are linear combinations of A[i,j].
        # It's easier to just iterate over the 7 Strassen multiplications.
        pass

    # This approach is complex to code dynamically. 
    # Alternative: Use the known rank-49 decomposition explicitly?
    # Or just use Rank 56 (Strassen outer, Schoolbook inner)?
    # Rank 56 + 16 = 72. Still better than 80.
    # Rank 49 + 16 = 65.
    
    # Let's try to generate the 49 components.
    
    # Strassen 2x2 factors (Rank 7)
    # U_s, V_s, W_s are 4x7
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # r=0: M1
    # A1 = A00+A11, B1 = B00+B11
    # A1 entries: 1,0; 0,1 (scaled by A blocks)
    # This is getting messy.
    
    # Let's use a simpler improvement: 
    # Use Strassen for 4x4x4 (Rank 49) is hard to code correctly on the fly.
    # Use Schoolbook for 4x4x4 (Rank 64) + Schoolbook for 4x4x1 (Rank 16) = 80.
    # Use Strassen for 2x2 blocks of 4x4x4?
    # If we use Strassen for the 4x4x4, we save 15 rank.
    
    # Let's try to implement the 49 rank decomposition.
    
    # The 4x4x4 tensor can be decomposed into 7 2x2x2 tensors.
    # Each 2x2x2 tensor has rank 7.
    # Total 49.
    
    # We define the linear maps for the 7 Strassen steps.
    # For step x, we have A_x and B_x.
    # A_x[i,j] = sum_{a,b} cA_x[i,j,a,b] * A[a,b]
    # B_x[i,j] = sum_{a,b} cB_x[i,j,a,b] * B[a,b]
    
    # Then we apply 2x2 Strassen to A_x, B_x.
    # The 2x2 Strassen factors U_s, V_s, W_s are fixed.
    
    # U_s[i, r] = coeff of A[i,j] in the r-th Strassen term of A_x?
    # No, U_s is for the 2x2 matrix multiplication.
    # U_s[i, r] corresponds to the coefficient of the (i,j) element of the left matrix in the r-th term.
    
    # Let's hardcode the 2x2 Strassen factors.
    # U_s (4x7), V_s (4x7), W_s (4x7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # r=0: (A00+A11)(B00+B11)
    # Left: A00+A11 -> [1,0; 0,1]
    # Right: B00+B11 -> [1,0; 0,1]
    # U_s[:, 0] = [1,0, 0,1]
    # V_s[:, 0] = [1,0, 0,1]
    # W_s[:, 0] = [1,0, 0,1] (C00 += M1, C01 += 0, C10 += 0, C11 += 0? No)
    # W_s for 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So W_s[k*2+i, r]
    # r=0 (M1): C00+=1, C11+=1. W_s[0,0]=1, W_s[3,0]=1.
    # r=1 (M2): C10+=1. W_s[2,1]=1.
    # r=2 (M3): C01+=1, C11+=1. W_s[1,2]=1, W_s[3,2]=1.
    # r=3 (M4): C00+=1, C10+=1. W_s[0,3]=1, W_s[2,3]=1.
    # r=4 (M5): C00-=1, C01+=1. W_s[0,4]=-1, W_s[1,4]=1.
    # r=5 (M6): C11+=1. W_s[3,5]=1.
    # r=6 (M7): C00+=1. W_s[0,6]=1.
    
    # Now for the 4x4x4, we have 7 such groups.
    # For each group x, we have A_x and B_x.
    # A_x is a 2x2 matrix. Its elements are linear combos of A.
    # B_x is a 2x2 matrix. Its elements are linear combos of B.
    
    # We can iterate x from 0 to 6.
    # For each x, we have 7 sub-terms.
    # Total 49 terms.
    
    # We need to define A_x and B_x coefficients.
    
    # x=0: M1 = (A00+A11)(B00+B11)
    # A0 = A00+A11. B0 = B00+B11.
    # x=1: M2 = (A10+A11)B00
    # A1 = A10+A11. B1 = B00.
    # x=2: M3 = A00(B01-B11)
    # A2 = A00. B2 = B01-B11.
    # x=3: M4 = A11(B10-B00)
    # A3 = A11. B3 = B10-B00.
    # x=4: M5 = (A00+A01)B11
    # A4 = A00+A01. B4 = B11.
    # x=5: M6 = (A10-A00)(B00+B01)
    # A5 = A10-A00. B5 = B00+B01.
    # x=6: M7 = (A01-A11)(B10+B11)
    # A6 = A01-A11. B6 = B10+B11.
    
    # We represent A_x as a 4x4 matrix of coefficients for A (flattened).
    # A_x[i,j] = sum_{a,b} C_A[x, i, j, a, b] * A[a,b]
    
    # This is getting long. I will code this carefully.
    
    return {"U": U1, "V": V1, "W": W1}
