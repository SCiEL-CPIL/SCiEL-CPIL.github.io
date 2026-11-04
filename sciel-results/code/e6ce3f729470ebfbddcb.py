import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    def verify_fast(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        # Check if sum_c U_ijk V_jkc W_kic == 1 for all i,j,k
        # Actually, the tensor T_ijk = 1 for all i,j,k is NOT correct.
        # The tensor T_ijk should be 1 if the indices match the multiplication structure?
        # No, the tensor is defined such that <T, A \otimes B \otimes C> = trace(C A B) ?
        # The condition is:
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k?
        # Let's check:
        # If we set A_ik = delta_ik, B_jk = delta_jk, C_ki = delta_ki?
        # The standard check is:
        # result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # This should equal 1 for all i, j, k?
        # Let's test with schoolbook.
        # U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for c corresponding to (i,j,k).
        # Sum over c: only one c is non-zero for each (i,j,k) in schoolbook?
        # In schoolbook, c is index for (i,j,k).
        # U[i,j,c_ijk]=1, V[j,k,c_ijk]=1, W[k,i,c_ijk]=1.
        # Sum is 1.
        # So yes, result should be ones((n,m,p)).
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, np.ones((n,m,p)), atol=1e-8, rtol=1e-8)

    # Attempt Rank 13
    rank_target = 13
    U_try = np.zeros((n*m, rank_target), dtype=np.float64)
    V_try = np.zeros((m*p, rank_target), dtype=np.float64)
    W_try = np.zeros((p*n, rank_target), dtype=np.float64)

    # Part 1: 3x3x3 Rank 7 (cols 0-6)
    # Hardcoded Strassen-like 3x3x3
    # This is a known decomposition.
    # U (9x7), V (9x7), W (9x7) for the 3x3x3 block.
    # Indices for V and W in this block: j in 0..2, k in 0..2.
    
    # Coefficients for U (i, j, c)
    U_333 = np.zeros((9, 7), dtype=np.float64)
    # c=0: A00
    U_333[0, 0] = 1
    # c=1: A00 + A01
    U_333[0, 1] = 1
    U_333[1, 1] = 1
    # c=2: A20 + A21
    U_333[6, 2] = 1
    U_333[7, 2] = 1
    # c=3: A22
    U_333[8, 3] = 1
    # c=4: A00 + A22
    U_333[0, 4] = 1
    U_333[8, 4] = 1
    # c=5: A01 - A22
    U_333[1, 5] = 1
    U_333[8, 5] = -1
    # c=6: A20 - A00
    U_333[6, 6] = 1
    U_333[0, 6] = -1

    V_333 = np.zeros((9, 7), dtype=np.float64)
    # c=0: B11 - B12
    V_333[4, 0] = 1  # j=1, k=1 -> 1*3+1=4
    V_333[5, 0] = -1 # j=1, k=2 -> 1*3+2=5
    # c=1: B22
    V_333[8, 1] = 1  # j=2, k=2 -> 2*3+2=8
    # c=2: B00
    V_333[0, 2] = 1  # j=0, k=0 -> 0
    # c=3: B00 - B01
    V_333[0, 3] = 1
    V_333[1, 3] = -1
    # c=4: B00 + B11
    V_333[0, 4] = 1
    V_333[4, 4] = 1
    # c=5: B12 + B22
    V_333[5, 5] = 1
    V_333[8, 5] = 1
    # c=6: B00 + B01
    V_333[0, 6] = 1
    V_333[1, 6] = 1

    W_333 = np.zeros((9, 7), dtype=np.float64)
    # We need to solve for W such that the identity holds.
    # Instead of hardcoding W which is error-prone, we can solve for W given U and V?
    # Or use a known W.
    # Let's try to derive W or use a search for W if U,V are fixed.
    # But for now, let's use a placeholder and verify.
    # If verification fails, we fallback.
    
    # Actually, I can just use the schoolbook for the 3x3x3 part if the hardcoded one is not verified easily.
    # But rank 9 for 3x3x3 is easy.
    # Let's use Rank 9 for the first part to be safe, and Rank 6 for the second. Total 15.
    # Rank 15 is better than 45.
    
    # Let's construct Rank 15 decomposition.
    rank_15 = 15
    U_15 = np.zeros((n*m, rank_15), dtype=np.float64)
    V_15 = np.zeros((m*p, rank_15), dtype=np.float64)
    W_15 = np.zeros((p*n, rank_15), dtype=np.float64)
    
    # Part 1: 3x3x3 Rank 9 (cols 0-8)
    # Terms: A_ij * B_jk for j in 0..2, i in 0..2, k in 0..2?
    # No, Rank 9 means 9 terms.
    # The 9 terms are A_i0 B_0k, A_i1 B_1k, A_i2 B_2k?
    # For each j, we have a rank 1 term in i,k space?
    # Term j: (A_col_j) * (B_row_j).
    # This is a rank 1 matrix.
    # So C = A_col0 B_row0 + A_col1 B_row1 + A_col2 B_row2.
    # This is rank 3 in matrix sense, but in tensor sense?
    # Tensor rank of sum of 3 outer products is 3?
    # No, the tensor is 3x3x3.
    # Each term A_i0 B_0k contributes to C_ik.
    # This is a rank 1 tensor in the (i,k) plane for fixed j?
    # The tensor T_ijk = delta_j0 for term 0?
    # Yes.
    # So we can use 3 terms for the 3x3x3 part?
    # Let's check.
    # Term 0: U[i,0,c0]=1, V[0,k,c0]=1, W[k,i,c0]=1.
    # Sum: U[i,j,c] V[j',k,c] W[k',i',c].
    # For c0: U[i,0,c0]=1, V[0,k,c0]=1, W[k,i,c0]=1.
    # Product: 1 * 1 * 1 = 1 if j=0, j'=0, k'=k, i'=i.
    # This gives 1 for T_i0k.
    # We want T_ijk = 1 for all? No.
    # We want the decomposition to represent the multiplication tensor.
    # The multiplication tensor has T_ijk = 1 for all i,j,k?
    # No, T_ijk is the coefficient of A_ik B_jk? No.
    # The tensor is defined by T_ijk = 1 if we consider the basis?
    # The condition sum_c ... = 1 for all i,j,k is correct for the structure tensor.
    # So if we use 3 terms for j=0,1,2:
    # c0 (j=0): T_i0k = 1.
    # c1 (j=1): T_i1k = 1.
    # c2 (j=2): T_i2k = 1.
    # This covers all i,j,k with T_ijk=1.
    # So Rank 3 is sufficient for 3x3x3?
    # No, Rank of 3x3x3 matrix multiplication is 7.
    # Why?
    # Because U, V, W must be linear combinations?
    # No, the rank is the minimum number of rank-1 tensors to sum to T.
    # T_ijk = 1 for all i,j,k.
    # This tensor is rank 1? No.
    # T_ijk = 1 is a rank 1 tensor?
    # T = u v w? u_i v_j w_k = 1?
    # If u_i=1, v_j=1, w_k=1, then T_ijk=1.
    # So Rank 1?
    # No, the tensor for matrix multiplication is NOT T_ijk=1.
    # The tensor is T_ijk = 1 if the indices are compatible?
    # The structure tensor of matrix multiplication has T_ijk = 1 for all i,j,k?
    # Let's check the definition.
    # <T, A \otimes B \otimes C> = trace(C A B).
    # If T_ijk = 1, then sum_ijk A_ik B_jk C_ki?
    # This is not trace(CAB).
    # The structure tensor T has T_ijk = 1 if we index A by (i,j), B by (j,k), C by (k,i)?
    # Then sum_ijk A_ij B_jk C_ki = trace(C A B).
    # So YES, T_ijk = 1 for all i,j,k.
    # So why is the rank 7?
    # Because U, V, W are matrices of size n x m, m x p, p x n.
    # The rank 1 tensor is U \otimes V \otimes W.
    # (U \otimes V \otimes W)_ijk = U_ij V_jk W_ki.
    # We want sum_c U_ij^c V_jk^c W_ki^c = 1 for all i,j,k.
    # This is possible with rank 1?
    # U_ij=1, V_jk=1, W_ki=1.
    # Then product is 1.
    # So Rank 1?
    # No, U, V, W must be such that they form a valid decomposition?
    # Is there a constraint?
    # The rank of the matrix multiplication tensor is 7.
    # My logic that T_ijk=1 is rank 1 must be wrong.
    # Ah, U, V, W are fixed matrices.
    # The tensor is the sum of rank 1 tensors formed by outer products of columns of U, V, W.
    # If I choose U=1, V=1, W=1, then the sum is 1.
    # So Rank 1?
    # No, the rank of 2x2x2 matrix multiplication is 7.
    # T_ijk = 1 for all i,j,k in 2x2x2.
    # If Rank 1, then U_ij V_jk W_ki = 1 for all i,j,k.
    # U_00 V_00 W_00 = 1
    # U_00 V_01 W_10 = 1
    # ...
    # This implies U_ij V_jk W_ki is constant.
    # This is possible.
    # But the rank is 7.
    # Why?
    # Because the tensor is NOT T_ijk=1.
    # The tensor is T_ijk = 1 if the indices match the multiplication?
    # No, the structure tensor is indeed T_ijk = 1 for all i,j,k.
    # Wait, I recall now.
    # The tensor is $T \in \mathbb{R}^{n \times m \times p}$.
    # $T_{i,j,k} = 1$ for all $i,j,k$ is NOT the matrix multiplication tensor.
    # The matrix multiplication tensor is defined by the bilinear map.
    # The structure tensor has entries $T_{i,j,k} = 1$?
    # Let's check 2x2x2.
    # Rank 7.
    # If T_ijk=1, rank is 1.
    # So T_ijk is NOT 1 for all i,j,k.
    # What is T_ijk?
    # It is 1 if the term A_ij B_jk contributes to C_ki?
    # Yes, every term contributes to exactly one C entry?
    # No, A_ij B_jk contributes to C_ik.
    # So T_ijk is the coefficient of A_ij B_jk in C_ki?
    # C_ki = sum_j A_ij B_jk? No, C_ik = sum_j A_ij B_jk.
    # So C_ki is not standard.
    # If C is p x n, then C_ki.
    # Then C_ki = sum_j A_ij B_jk.
    # So T_ijk = 1 for all i,j,k.
    # Then Rank 1?
    # There is a contradiction.
    # The rank of 2x2x2 matrix multiplication is 7.
    # Therefore, my assumption that T_ijk=1 is rank 1 must be flawed.
    # The flaw: U, V, W are not arbitrary.
    # They are matrices.
    # The rank 1 tensor is $u v w$ where $u \in \mathbb{R}^{n \times m}$, etc.
    # If $u_{ij}=1, v_{jk}=1, w_{ki}=1$, then $u_{ij}v_{jk}w_{ki}=1$.
    # So it IS rank 1.
    # Why is the rank 7?
    # Ah! The tensor is NOT $T_{ijk}=1$.
    # The tensor is $T_{ijk} = \delta_{i,k}$? No.
    # The structure tensor of matrix multiplication $M_{n,m,p}$ has entries $T_{i,j,k} = 1$?
    # I will trust the literature: Rank 7 for 2x2x2.
    # And Rank 7 for 3x3x3.
    
    # I will use the schoolbook fallback to ensure correctness, as deriving the exact coefficients is risky.
    # Schoolbook rank 45.
    
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }

BET: {mechanism: "Fallback to schoolbook ensures validity, but no improvement expected over seed.", q: 0.0, delta: [0.0, 0.0]}
