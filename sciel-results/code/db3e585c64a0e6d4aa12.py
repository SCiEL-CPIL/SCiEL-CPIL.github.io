import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Matrix multiplication tensor for n x m times m x p.
    # We seek a rank-minimal exact decomposition.
    # Schoolbook rank is n*m*p = 3*4*5 = 60.
    # We can reduce the rank by using Strassen's algorithm on a 2x2x2 sub-block.
    # The 2x2x2 block has 8 terms. Strassen reduces this to 7 terms.
    # Total rank = 60 - 8 + 7 = 59.
    
    n, m, p = DIMENSIONS
    rank = n * m * p - 1  # 59
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # We use Fraction for exact arithmetic to ensure exactness.
    U = [[Fraction(0) for _ in range(rank)] for _ in range(n * m)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(m * p)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(p * n)]
    
    # We will fill the tensor terms.
    # The tensor T is defined by T[i, j, k] = 1 if we have the term A[i, j] * B[j, k] contributing to C[i, k].
    # The decomposition is sum_c U[:, c] V[:, c] W[:, c] such that
    # (A * B)[i, k] = sum_{i,j,k} A[i, j] B[j, k] = sum_c (sum_j U[i*m+j, c] A[i, j]) * (sum_j V[j*p+k, c] B[j, k]) * W[k*n+i, c]
    # Wait, the standard tensor decomposition for matrix multiplication is:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where (u_c)_{i,j} = U[i*m+j, c], (v_c)_{j,k} = V[j*p+k, c], (w_c)_{k,i} = W[k*n+i, c]
    # And the identity is:
    # sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} = sum_c u_c ⊗ v_c ⊗ w_c
    # This means that for each (i, j, k), the coefficient of A[i, j] B[j, k] C[i, k] in the tensor is 1.
    # In the decomposition, the coefficient of A[i, j] B[j, k] C[i, k] is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # So we need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
    
    # Strategy:
    # For all (i, j, k) NOT in the Strassen block (i=0,1; j=0,1; k=0,1), we use the standard basis vector.
    # For the Strassen block, we use the 7 Strassen vectors.
    
    # Define the Strassen block indices
    i_block = [0, 1]
    j_block = [0, 1]
    k_block = [0, 1]
    
    # Strassen vectors for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) ? No.
    # Let's use the standard Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # The vectors u_c, v_c, w_c are:
    # M1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> (A00+A11)(B00+B11) contributes to all C?
    # No, the w vector determines which C element it contributes to.
    # Actually, the decomposition is:
    # T_block = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^4 (for A indices 00,01,10,11), v_c is in R^4 (for B indices 00,01,10,11), w_c is in R^4 (for C indices 00,01,10,11).
    #
    # Let's map indices:
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #
    # M1 = (A00 + A11)(B00 + B11)
    # u1 = (1, 0, 0, 1)
    # v1 = (1, 0, 0, 1)
    # This product is (A00B00 + A00B11 + A11B00 + A11B11).
    # It contributes to C00, C01, C10, C11?
    # In the tensor, the term A00B00 contributes to C00. A00B11 contributes to C01. A11B00 contributes to C10. A11B11 contributes to C11.
    # So w1 must be (1, 1, 1, 1)?
    # Let's check: sum_c u_c[i,j] v_c[j,k] w_c[k,i] = 1 for all (i,j,k) in block.
    # If w1 = (1,1,1,1), then M1 contributes 1 to all (i,j,k) in the block?
    # u1[0,0]=1, v1[0,0]=1, w1[0,0]=1 -> A00B00C00 gets 1.
    # u1[0,0]=1, v1[0,1]=0, w1[0,1]=1 -> A00B01C01 gets 0.
    # This is not working. The Strassen vectors are not simply Kronecker products of sums.
    #
    # Let's look up the explicit vectors for Strassen's 7-term decomposition of the 2x2x2 tensor.
    # The 8 standard terms are:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    #
    # Strassen's decomposition:
    # 1. (e00 + e11) ⊗ (e00 + e11) ⊗ (e00 + e01 + e10 + e11)  -- This is not rank 1.
    #
    # Actually, the 7 vectors are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,1,1,1)
    # 2. u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
    # 3. u=(1,0,0,0), v=(0,0,1,-1), w=(0,1,0,0)
    # 4. u=(0,0,0,1), v=(0,1,-1,0), w=(1,0,0,0)
    # 5. u=(1,1,0,0), v=(0,0,0,1), w=(0,0,0,1)
    # 6. u=(0,-1,1,0), v=(1,1,0,0), w=(0,0,0,1)
    # 7. u=(0,1,-1,0), v=(0,0,1,1), w=(0,0,0,1)
    #
    # Let's verify term 1:
    # u1 = (1,0,0,1), v1 = (1,0,0,1), w1 = (1,1,1,1)
    # (i=0,j=0,k=0): 1*1*1 = 1
    # (i=0,j=0,k=1): 1*0*1 = 0
    # (i=0,j=1,k=0): 0*1*1 = 0
    # (i=0,j=1,k=1): 0*0*1 = 0
    # (i=1,j=0,k=0): 0*1*1 = 0
    # (i=1,j=0,k=1): 0*0*1 = 0
    # (i=1,j=1,k=0): 1*1*1 = 1
    # (i=1,j=1,k=1): 1*0*1 = 0
    # This gives 1 for (0,0,0) and (1,1,1). But we need 1 for all 8.
    #
    # It seems I need the correct explicit vectors.
    # From a reliable source (e.g. Wikipedia or a textbook):
    # The 7 rank-1 tensors are:
    # 1. (1,0,0,1) ⊗ (1,0,0,1) ⊗ (1,1,1,1)
    # 2. (0,0,1,1) ⊗ (1,0,0,0) ⊗ (0,0,1,0)
    # 3. (1,0,0,0) ⊗ (0,0,1,-1) ⊗ (0,1,0,0)
    # 4. (0,0,0,1) ⊗ (0,1,-1,0) ⊗ (1,0,0,0)
    # 5. (1,1,0,0) ⊗ (0,0,0,1) ⊗ (0,0,0,1)
    # 6. (0,-1,1,0) ⊗ (1,1,0,0) ⊗ (0,0,0,1)
    # 7. (0,1,-1,0) ⊗ (0,0,1,1) ⊗ (0,0,0,1)
    #
    # Let's re-verify term 1.
    # u1=(1,0,0,1), v1=(1,0,0,1), w1=(1,1,1,1)
    # The product u1 ⊗ v1 ⊗ w1 has components:
    # (i,j,k) -> u1[i,j] * v1[j,k] * w1[k,i]
    # (0,0,0): 1*1*1 = 1
    # (0,0,1): 1*0*1 = 0
    # (0,1,0): 0*1*1 = 0
    # (0,1,1): 0*0*1 = 0
    # (1,0,0): 0*1*1 = 0
    # (1,0,1): 0*0*1 = 0
    # (1,1,0): 1*1*1 = 1
    # (1,1,1): 1*0*1 = 0
    # This is clearly not the identity tensor.
    #
    # I will use a known correct set of vectors for the 2x2x2 tensor.
    # The 8 standard terms are e_1...e_8.
    # Strassen's 7 terms are linear combinations.
    # Let's just use the standard 60 terms. It is safe and correct.
    # The task is to find a rank-minimal decomposition. 60 is the schoolbook rank.
    # If I cannot find a valid 59-rank decomposition quickly, I should return 60.
    # But the inspiration says rank 52 is known.
    # However, implementing a complex 52-rank decomposition from scratch is risky.
    # Let's try to implement the 59-rank decomposition correctly.
    
    # The correct Strassen vectors for the 2x2x2 tensor T = sum e_{i,j} e_{j,k} e_{k,i} are:
    # Let's denote the basis vectors for A as a00, a01, a10, a11.
    # The 7 vectors (u, v, w) are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,1,1,1)  -- This was wrong.
    
    # Let's look at the tensor identity:
    # sum_{i,j,k} a_{i,j} b_{j,k} c_{k,i} = sum_c (sum_{i,j} u_c[i,j] a_{i,j}) (sum_{j,k} v_c[j,k] b_{j,k}) (sum_{k,i} w_c[k,i] c_{k,i})
    #
    # The 7 multipliers are:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # And the outputs are:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    #
    # This means:
    # u1 = (1,0,0,1), v1 = (1,0,0,1), w1 = (1,0,0,1)  -> M1 contributes to c00?
    # No, M1 contributes to c00, c01, c10, c11?
    # c00 has M1. c11 has M1.
    # So w1 should have 1 at (0,0) and (1,1).
    # w1 = (1,0,0,1)
    # Let's check:
    # u1=(1,0,0,1), v1=(1,0,0,1), w1=(1,0,0,1)
    # (0,0,0): 1*1*1 = 1
    # (1,1,1): 1*1*1 = 1
    # (0,0,1): 1*0*0 = 0
    # (0,1,0): 0*1*0 = 0
    # (1,0,0): 0*1*1 = 0
    # (1,0,1): 0*0*0 = 0
    # (0,1,1): 0*0*0 = 0
    # (1,1,0): 1*1*0 = 0
    # This gives 1 for (0,0,0) and (1,1,1).
    #
    # M2 = (a10+a11)b00
    # u2 = (0,0,1,1), v2 = (1,0,0,0), w2 = (0,0,1,0)
    # c10 has M2.
    # w2 should have 1 at (1,0).
    # w2 = (0,0,1,0)
    # Check:
    # u2=(0,0,1,1), v2=(1,0,0,0), w2=(0,0,1,0)
    # (1,0,0): 1*1*1 = 1
    # (1,1,0): 1*0*1 = 0
    # (0,0,0): 0*1*0 = 0
    # This gives 1 for (1,0,0).
    #
    # M3 = a00(b10-b11)
    # u3 = (1,0,0,0), v3 = (0,0,1,-1), w3 = (0,1,0,0)
    # c01 has M3.
    # w3 should have 1 at (0,1).
    # w3 = (0,1,0,0)
    # Check:
    # u3=(1,0,0,0), v3=(0,0,1,-1), w3=(0,1,0,0)
    # (0,1,0): 1*1*1 = 1
    # (0,1,1): 1*(-1)*1 = -1
    # This gives 1 for (0,1,0) and -1 for (0,1,1).
    #
    # M4 = a11(b01-b00)
    # u4 = (0,0,0,1), v4 = (0,1,-1,0), w4 = (1,0,0,0)
    # c00 has M4.
    # w4 should have 1 at (0,0).
    # w4 = (1,0,0,0)
    # Check:
    # u4=(0,0,0,1), v4=(0,1,-1,0), w4=(1,0,0,0)
    # (1,0,0): 1*(-1)*1 = -1
    # (1,1,0): 1*1*1 = 1
    # This gives -1 for (1,0,0) and 1 for (1,1,0).
    #
    # M5 = (a00+a01)b11
    # u5 = (1,1,0,0), v5 = (0,0,0,1), w5 = (0,0,0,1)
    # c00 has -M5. c01 has M5. c11 has M5.
    # w5 should have -1 at (0,0), 1 at (0,1), 1 at (1,1).
    # w5 = (-1, 1, 0, 1)
    # Check:
    # u5=(1,1,0,0), v5=(0,0,0,1), w5=(-1,1,0,1)
    # (0,0,0): 1*0*(-1) = 0
    # (0,1,0): 1*0*1 = 0
    # (0,0,1): 1*1*(-1) = -1
    # (0,1,1): 1*1*1 = 1
    # This gives -1 for (0,0,1) and 1 for (0,1,1).
    #
    # M6 = (a10-a00)(b00+b01)
    # u6 = (-1,0,1,0), v6 = (1,1,0,0), w6 = (0,0,0,1)
    # c11 has M6.
    # w6 should have 1 at (1,1).
    # w6 = (0,0,0,1)
    # Check:
    # u6=(-1,0,1,0), v6=(1,1,0,0), w6=(0,0,0,1)
    # (0,0,1): -1*1*1 = -1
    # (0,1,1): -1*1*1 = -1
    # (1,0,1): 1*1*1 = 1
    # (1,1,1): 1*1*1 = 1
    # This gives -1 for (0,0,1), (0,1,1) and 1 for (1,0,1), (1,1,1).
    #
    # M7 = (a01-a11)(b10+b11)
    # u7 = (0,1,0,-1), v7 = (0,0,1,1), w7 = (1,0,0,0)
    # c00 has M7.
    # w7 should have 1 at (0,0).
    # w7 = (1,0,0,0)
    # Check:
    # u7=(0,1,0,-1), v7=(0,0,1,1), w7=(1,0,0,0)
    # (0,0,0): 0*0*1 = 0
    # (0,1,0): 1*1*1 = 1
    # (1,0,0): -1*1*1 = -1
    # (1,1,0): -1*1*1 = -1
    # This gives 1 for (0,1,0) and -1 for (1,0,0), (1,1,0).
    #
    # Now let's sum all 7 vectors and see if we get the identity tensor.
    # Identity tensor has 1 at (i,j,k) if i,j,k are consistent with matrix multiplication.
    # For 2x2x2, all 8 (i,j,k) are valid.
    #
    # (0,0,0):
    # M1: 1
    # M2: 0
    # M3: 0
    # M4: 0
    # M5: 0
    # M6: 0
    # M7: 0
    # Sum: 1. Correct.
    #
    # (0,0,1):
    # M1: 0
    # M2: 0
    # M3: 0
    # M4: 0
    # M5: -1
    # M6: -1
    # M7: 0
    # Sum: -2. Incorrect. Should be 1.
    #
    # It seems my w vectors are not correct.
    # I will fall back to the schoolbook decomposition of rank 60.
    # It is always correct.
    
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
