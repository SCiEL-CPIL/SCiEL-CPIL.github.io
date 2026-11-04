import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2
    # U2, V2, W2 are 4x7 matrices
    # Rows correspond to indices i*2+j, j*2+k, k*2+i
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00

    # Kronecker product to get 4x4 decomposition
    # U shape (16, 49), V shape (16, 49), W shape (16, 49)
    U = np.kron(U2, U2)
    V = np.kron(V2, V2)
    W = np.kron(W2, W2)
    
    # Convert to lists for return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Verify rank
    print(f"Rank: {U.shape[1]}")
    
    # Verify decomposition
    # T_ijk = sum_r U[i*4+j, r] V[j*4+k, r] W[k*4+i, r]
    # Should be 1 if indices are valid, 0 otherwise?
    # For 4x4, valid means any i,j,k?
    # Let's check a few values
    # C_00 = A_00 B_00 + A_01 B_10 + A_02 B_20 + A_03 B_30
    # So T_0,0,0 = 1, T_1,0,0 = 1 (A_10 B_00 -> C_10? No, A_10 B_00 -> C_10? No, A_1j B_j0 -> C_10)
    # A_10 B_00 contributes to C_10.
    # So T_1,0,0 should be 1?
    # Wait, T_i,j,k corresponds to A_ij, B_jk, C_ki?
    # No, C_ik.
    # So T_i,j,k = 1 if A_ij and B_jk contribute to C_ik.
    # They always do.
    # So T_i,j,k = 1 for all i,j,k?
    # Let's check T_0,0,0.
    val = 0
    for r in range(U.shape[1]):
        val += U[0, r] * V[0, r] * W[0, r]
    print(f"T_0,0,0 = {val}") # Should be 1
    
    val = 0
    for r in range(U.shape[1]):
        val += U[1, r] * V[0, r] * W[0, r] # A_01, B_00, C_00? No, A_01 B_00 -> C_00? No.
    # A_01 B_00 is not a term in C_00. C_00 = sum_j A_0j B_j0.
    # So A_01 B_00 is not in C_00.
    # So T_0,1,0 should be 0?
    # But my index mapping:
    # U row i*m+j. V row j*p+k. W row k*n+i.
    # For A_01 (i=0, j=1), B_00 (j=0, k=0).
    # j must match. Here j=1 and j=0. Mismatch.
    # So the product is 0?
    # In the tensor, the index j is shared.
    # So T_i,j,k is non-zero only if the j in U and V match?
    # But U and V are just vectors.
    # The sum is over r.
    # The condition is that the decomposition reconstructs the tensor.
    # The tensor T has T_i,j,k = 1 if j is the summation index?
    # No, the tensor is fixed.
    # T_i,j,k = 1 if the indices form a valid multiplication step?
    # Actually, T_i,j,k = 1 for all i,j,k is WRONG.
    # T_i,j,k = 1 if we are computing C_ik from A_ij and B_jk.
    # This is always true for any i,j,k.
    # Wait, C_ik = sum_j A_ij B_jk.
    # So for a fixed i,k, we sum over j.
    # The term A_ij B_jk is present.
    # So T_i,j,k = 1 for all i,j,k.
    # But then why did T_0,0,0 = 0 in my manual check of Strassen?
    # Let's re-evaluate Strassen T_0,0,0.
    # U_2[0] = [1, 0, 1, 0, 1, -1, 0]
    # V_2[0] = [1, 1, 0, -1, 0, 1, 0]
    # W_2[0] = [1, 0, 1, 1, -1, 0, 1]
    # Dot: 1*1*1 + 0*1*0 + 1*0*1 + 0*(-1)*1 + 1*0*(-1) + (-1)*1*0 + 0*0*1 = 1.
    # Ah, I made an arithmetic error before.
    # 1*1*1 = 1.
    # (-1)*1*0 = 0.
    # So it is 1.
    # So T_0,0,0 = 1.
    # What about T_0,1,0? (A_01, B_10, C_00? No, B_10 is j=1, k=0. A_01 is i=0, j=1. Match. C_00 is k=0, i=0. Match.)
    # So T_0,1,0 should be 1.
    # Let's check.
    # U_2[1] (row 1) = [0, 0, 0, 0, 1, 0, 1]
    # V_2[2] (row 2, j=1, k=0? No, V row is j*p+k. p=2. j=1, k=0 -> row 2)
    # V_2[2] = [0, 0, 0, 1, 0, 0, 1]
    # W_2[0] (row 0, k=0, i=0) = [1, 0, 1, 1, -1, 0, 1]
    # Dot: 0*0*1 + 0*0*0 + 0*0*1 + 0*1*1 + 1*0*(-1) + 0*0*0 + 1*1*1 = 1.
    # So T_0,1,0 = 1.
    # So it seems T_i,j,k = 1 for all i,j,k.
    
    pass
