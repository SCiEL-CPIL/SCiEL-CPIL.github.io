import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 23
    # We will try to construct the rank 23 decomposition.
    # If verification fails, we fall back to rank 27.
    
    # The 23 terms for 3x3 matrix multiplication
    # Format: (u_nonzeros, v_nonzeros, w_nonzeros)
    # u: A[i,j], v: B[j,k], w: C[k,i]
    
    # This is the Strassen 23-term decomposition.
    # Coefficients are hardcoded from known solutions.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # u_idx, v_idx, w_idx are lists of (index, value)
        pass

    # Instead of complex helpers, I will define the 23 vectors directly.
    # U: 9x23, V: 9x23, W: 9x23
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Term 0-6: Strassen for top-left 2x2
    # A00, A01, A10, A11
    # B00, B01, B10, B11
    
    # T0: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[1,0]=1
    W[0,0]=1; W[1,0]=1; W[3,0]=1; W[4,0]=1 # C00, C01, C10, C11 contributions
    
    # This is getting tedious and error prone.
    
    # SAFER APPROACH:
    # Use Rank 27 (Schoolbook) but check if we can reduce by 1 or 2 easily.
    # Actually, I will just return the Rank 27 solution to ensure validity,
    # UNLESS I can confidently code Rank 23.
    
    # I will code Rank 23 using a known valid set of coefficients.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    # The 3x3 case is less commonly detailed in simple wikis.
    
    # I will use a Rank 25 construction which is easier to derive:
    # 1. Strassen 2x2 (7 terms)
    # 2. Scalar 1x1 (1 term)
    # 3. Cross terms:
    #    C02 = A00B02 + A01B12 + A02B22
    #    C12 = A10B02 + A11B12 + A12B22
    #    C20 = A20B00 + A21B10 + A22B20
    #    C21 = A20B01 + A21B11 + A22B21
    #    C22 = A20B02 + A21B12 + A22B22
    
    # If I just add these 15 terms, rank is 22.
    # Why is it 23? Because the Strassen terms are linear combinations, 
    # and the "cross" terms might not be independent of them in a way that allows simple addition?
    # No, the indices are disjoint for the "new" elements.
    # The only overlap is A00, A01, A10, A11, B00, B01, B10, B11.
    # The cross terms use A00, A01, A10, A11 in combination with B02, B12, etc.
    # So they are distinct rank-1 tensors.
    # So 7 + 15 = 22 should be a valid decomposition?
    # If 22 is valid, then M3 <= 22. But M3 = 23.
    # Contradiction.
    # The error is that C02 = A00B02 + ... 
    # The term A00B02 is a rank 1 tensor.
    # Is it linearly independent from the Strassen terms?
    # The Strassen terms are M1..M7.
    # M1 = (A00+A11)(B00+B11).
    # A00B02 is not a linear combination of M1..M7 because it involves B02 which is 0 in M1..M7.
    # So yes, it is independent.
    # So why is the rank 23?
    # Maybe my count of 15 is wrong.
    # C02: 3 terms.
    # C12: 3 terms.
    # C20: 3 terms.
    # C21: 3 terms.
    # C22: 3 terms.
    # Total 15.
    # Total 22.
    # I must be missing a dependency or the Strassen terms don't cover the 2x2 block completely in a way that allows this simple addition?
    # No, Strassen covers the 2x2 block exactly.
    
    # There is a known result that M3 = 23.
    # This implies that the naive 22-term construction has a linear dependence or error.
    # Actually, C22 = A20B02 + A21B12 + A22B22.
    # The term A22B22 is the scalar term.
    # So C22 uses 2 cross terms + 1 scalar.
    # C02, C12, C20, C21 use 3 each.
    # Total 3*4 + 2 + 1 = 15.
    # Plus 7 Strassen.
    # Total 22.
    
    # I will implement the 22-term construction and verify it.
    # If it fails, I will return 27.
    # If it passes, I will return 22 (score 1.22).
    
    rank = 22
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    col = 0
    
    # Strassen 2x2
    # T0: (A00+A11)(B00+B11) -> C00, C01, C10, C11
    U[0][col]=1; U[3][col]=1
    V[0][col]=1; V[1][col]=1
    W[0][col]=1; W[1][col]=1; W[3][col]=1; W[4][col]=1
    col+=1
    
    # T1: (A10+A11)B00
    U[3][col]=1; U[4][col]=1
    V[0][col]=1
    W[3][col]=1; W[4][col]=1
    col+=1
    
    # T2: A00(B01-B11)
    U[0][col]=1
    V[1][col]=1; V[4][col]=-1
    W[1][col]=1; W[4][col]=-1
    col+=1
    
    # T3: A11(B10-B00)
    U[4][col]=1
    V[3][col]=1; V[0][col]=-1
    W[3][col]=1; W[0][col]=-1
    col+=1
    
    # T4: (A00+A01)B11
    U[0][col]=1; U[1][col]=1
    V[4][col]=1
    W[0][col]=1; W[1][col]=1
    col+=1
    
    # T5: (A10-A00)(B00+B01)
    U[3][col]=1; U[0][col]=-1
    V[0][col]=1; V[1][col]=1
    W[3][col]=1; W[4][col]=1
    col+=1
    
    # T6: (A11-A10)(B10+B11)
    U[4][col]=1; U[3][col]=-1
    V[3][col]=1; V[4][col]=1
    W[3][col]=1; W[4][col]=1
    col+=1
    
    # Scalar
    # T7: A22B22
    U[8][col]=1
    V[8][col]=1
    W[8][col]=1
    col+=1
    
    # Cross Terms
    # C02 = A00B02 + A01B12 + A02B22
    # T8: A00B02
    U[0][col]=1
    V[2][col]=1
    W[2][col]=1
    col+=1
    
    # T9: A01B12
    U[1][col]=1
    V[5][col]=1
    W[2][col]=1
    col+=1
    
    # T10: A02B22
    U[2][col]=1
    V[8][col]=1
    W[2][col]=1
    col+=1
    
    # C12 = A10B02 + A11B12 + A12B22
    # T11: A10B02
    U[3][col]=1
    V[2][col]=1
    W[5][col]=1
    col+=1
    
    # T12: A11B12
    U[4][col]=1
    V[5][col]=1
    W[5][col]=1
    col+=1
    
    # T13: A12B22
    U[5][col]=1
    V[8][col]=1
    W[5][col]=1
    col+=1
    
    # C20 = A20B00 + A21B10 + A22B20
    # T14: A20B00
    U[6][col]=1
    V[0][col]=1
    W[6][col]=1
    col+=1
    
    # T15: A21B10
    U[7][col]=1
    V[3][col]=1
    W[6][col]=1
    col+=1
    
    # T16: A22B20
    U[8][col]=1
    V[6][col]=1
    W[6][col]=1
    col+=1
    
    # C21 = A20B01 + A21B11 + A22B21
    # T17: A20B01
    U[6][col]=1
    V[1][col]=1
    W[7][col]=1
    col+=1
    
    # T18: A21B11
    U[7][col]=1
    V[4][col]=1
    W[7][col]=1
    col+=1
    
    # T19: A22B21
    U[8][col]=1
    V[7][col]=1
    W[7][col]=1
    col+=1
    
    # C22 = A20B02 + A21B12 + A22B22 (A22B22 already in T7)
    # T20: A20B02
    U[6][col]=1
    V[2][col]=1
    W[8][col]=1
    col+=1
    
    # T21: A21B12
    U[7][col]=1
    V[5][col]=1
    W[8][col]=1
    col+=1
    
    # Total 22 terms.
    
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check tensor identity
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We check if this equals delta[i,j,k] for the multiplication tensor?
    # No, the tensor is defined by C[k,i] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] should be 1 if the product contributes to C[k,i].
    # Actually, the tensor entries are 1 for all i,j,k.
    # Wait, the matrix multiplication tensor has entries 1 for all (i,j,k) in {0,1,2}^3.
    # So we check if sum_r U[i*3+j, r] V[j*3+k, r] W[k*3+i, r] == 1 for all i,j,k.
    
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(rank):
                    s += U_np[i*3+j, r] * V_np[j*3+k, r] * W_np[k*3+i, r]
                if abs(s - 1) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to 27
        rank = 27
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
