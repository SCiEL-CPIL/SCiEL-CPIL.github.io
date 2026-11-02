import numpy as np

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_term(i, j, k, u_val, v_val, w_val, idx):
        U[i * m + j, idx] += u_val
        V[j * p + k, idx] += v_val
        W[k * n + i, idx] += w_val

    idx = 0

    # Block j=2: M_{2,1,7} -> 14 terms
    for i in range(2):
        for k in range(7):
            add_term(i, 2, k, 1.0, 1.0, 1.0, idx)
            idx += 1
    # idx is now 14

    # Block j in {0,1}: M_{2,2,7}
    # Split k into {0,1}, {2,3}, {4,5}, {6}
    # Strassen for 2x2x2 blocks:
    # M00 = M11 + M12 + M21 - M22
    # M01 = M12 + M22
    # M10 = M21
    # M11 = M11
    # where M11 = (A00+A11)(B00+B11)
    # M12 = (A00+A11)(B10-B11)
    # M21 = (A10-A00)(B00+B10)
    # M22 = (A10-A01)(B11-B01)

    def add_strassen_2x2x2(j_base, k_base, start_idx):
        # j_base in {0,1} is the base for j index (0 or 1)
        # k_base in {0,1,2,...} is the base for k index
        # We are computing C_{i,k} for i in {0,1}, k in {k_base, k_base+1}
        # A is 2x2: rows i in {0,1}, cols j in {j_base, j_base+1}
        # B is 2x2: rows j in {j_base, j_base+1}, cols k in {k_base, k_base+1}
        # C is 2x2: rows i in {0,1}, cols k in {k_base, k_base+1}
        
        # M11 = (A00+A11)(B00+B11)
        # A00 = A[i=0, j=j_base], A11 = A[i=1, j=j_base+1]
        # B00 = B[j=j_base, k=k_base], B11 = B[j=j_base+1, k=k_base+1]
        # Term: U[i, j] * V[j, k] * W[k, i]
        # For M11, we need to add 4 terms to U, V, W? No, Strassen terms are products of sums.
        # Each Strassen term M_r is a product of two linear forms.
        # In the tensor decomposition, each term r contributes:
        # U[i*m+j, r] = coeff_U
        # V[j*p+k, r] = coeff_V
        # W[k*n+i, r] = coeff_W
        # such that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = C[i,k]
        
        # Wait, the tensor is T_{i,j,k} = delta_{j,j} (always 1 for valid j).
        # The decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) * W[k*n+i, r] ?
        # No, the standard form is:
        # C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
        # Actually, the tensor T is defined such that:
        # T_{i,j,k} = 1 if we multiply A[i,j] and B[j,k] to get C[i,k].
        # The decomposition is:
        # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # For Strassen 2x2x2:
        # C00 = M11 + M12 + M21 - M22
        # C01 = M12 + M22
        # C10 = M21
        # C11 = M11
        
        # Let's map the Strassen terms to the tensor indices.
        # The 2x2x2 tensor has indices i in {0,1}, j in {0,1}, k in {0,1}.
        # The 7 terms are:
        # r=0: M11 = (A00+A11)(B00+B11)
        #   U[0,0]=1, U[1,1]=1
        #   V[0,0]=1, V[1,1]=1
        #   W[0,0]=1, W[1,1]=1? No.
        #   The term contributes to C00, C01, C10, C11.
        #   Specifically, M11 contributes +1 to C00, +1 to C11.
        #   So W[0,0] (k=0, i=0) = 1, W[1,1] (k=1, i=1) = 1.
        #   Let's check:
        #   sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        #   For i=0, j=0, k=0: U[0,0] V[0,0] W[0,0] = 1*1*1 = 1.
        #   For i=0, j=1, k=1: U[1,1] V[1,1] W[1,1] = 1*1*1 = 1.
        #   For i=1, j=0, k=0: U[0,0] V[0,0] W[0,1] = 1*1*0 = 0.
        #   For i=1, j=1, k=1: U[1,1] V[1,1] W[1,1] = 1*1*1 = 1.
        #   This doesn't look right. The tensor T_{i,j,k} is 1 for all i,j,k in the 2x2x2 case?
        #   No, T_{i,j,k} is 1 if j is the shared index. In 2x2x2, j is 0 or 1.
        #   The tensor is:
        #   T_{0,0,0}=1, T_{0,1,0}=0, T_{0,0,1}=0, T_{0,1,1}=1
        #   T_{1,0,0}=0, T_{1,1,0}=1, T_{1,0,1}=1, T_{1,1,1}=0
        #   Wait, C[i,k] = sum_j A[i,j] B[j,k].
        #   So T_{i,j,k} = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
        #   This is always true for any i,j,k. So T_{i,j,k} = 1 for all i,j,k?
        #   No, the tensor is defined on the indices of the inputs and output.
        #   The multiplication tensor is:
        #   T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
        #   So T_{(i,j),(j,k),(i,k)} = 1.
        #   In the flattened indices:
        #   U index: i*m + j
        #   V index: j*p + k
        #   W index: k*n + i
        #   So T_{i*m+j, j*p+k, k*n+i} = 1.
        #   For 2x2x2:
        #   i=0, j=0, k=0: U[0], V[0], W[0] -> 1
        #   i=0, j=1, k=0: U[1], V[2], W[1] -> 1
        #   i=0, j=0, k=1: U[0], V[1], W[2] -> 1
        #   i=0, j=1, k=1: U[1], V[3], W[3] -> 1
        #   i=1, j=0, k=0: U[2], V[0], W[1] -> 1
        #   i=1, j=1, k=0: U[3], V[2], W[2] -> 1
        #   i=1, j=0, k=1: U[2], V[1], W[3] -> 1
        #   i=1, j=1, k=1: U[3], V[3], W[0] -> 1
        
        # Strassen decomposition for 2x2x2:
        # M11 = (A00+A11)(B00+B11)
        #   U[0,0]=1, U[3,0]=1 (A00, A11)
        #   V[0,0]=1, V[3,0]=1 (B00, B11)
        #   W[0,0]=1, W[3,0]=1 (C00, C11)
        #   Check:
        #   i=0, j=0, k=0: U[0,0] V[0,0] W[0,0] = 1*1*1 = 1. (Correct)
        #   i=0, j=1, k=1: U[3,0] V[3,0] W[3,0] = 1*1*1 = 1. (Correct)
        #   i=1, j=0, k=0: U[2,0] V[0,0] W[1,0] = 0*1*0 = 0. (Correct, T is 0)
        #   i=1, j=1, k=1: U[3,0] V[3,0] W[0,0] = 1*1*0 = 0. (Correct, T is 0)
        
        # M12 = (A00+A11)(B10-B11)
        #   U[0,1]=1, U[3,1]=1
        #   V[1,1]=1, V[3,1]=-1 (B10, -B11)
        #   W[1,1]=1, W[3,1]=1 (C01, C11)
        #   Check:
        #   i=0, j=1, k=0: U[1,1] V[1,1] W[0,1] = 0*1*0 = 0. (T is 0)
        #   i=0, j=1, k=1: U[3,1] V[3,1] W[1,1] = 1*(-1)*1 = -1. (T is 1, but M12 contributes to C01, C11)
        #   Wait, C01 = M12 + M22. C11 = M11.
        #   So M12 should contribute +1 to C01 and +1 to C11?
        #   No, C01 = M12 + M22. So M12 contributes to C01.
        #   C11 = M11. So M12 does not contribute to C11.
        #   My W indices for M12 were wrong.
        #   M12 = (A00+A11)(B10-B11).
        #   This product contributes to C01 and C11?
        #   C01 = sum_j A[0,j] B[j,1] = A[0,0]B[0,1] + A[0,1]B[1,1].
        #   C11 = sum_j A[1,j] B[j,1] = A[1,0]B[0,1] + A[1,1]B[1,1].
        #   M12 = (A00+A11)(B10-B11) = A00B10 - A00B11 + A11B10 - A11B11.
        #   This doesn't match C01 or C11 directly.
        #   The standard Strassen formulas are:
        #   M1 = (A11 - A21)(B11 + B12)
        #   M2 = (A21 + A22)B11
        #   M3 = A11(B21 - B11)
        #   M4 = A22(B22 - B12)
        #   M5 = (A11 + A22)(B11 + B22)
        #   M6 = (A12 - A22)(B11 + B21)
        #   M7 = (A21 + A11)(B12 - B22)
        #   C11 = M5 + M4 - M3 - M7
        #   C12 = M1 + M7
        #   C21 = M2 + M6
        #   C22 = M5 + M3 - M1 - M2
        
        # Let's use the standard 7-term Strassen decomposition.
        # Indices: A is 2x2, B is 2x2, C is 2x2.
        # A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}.
        
        # M5 = (A00+A11)(B00+B11)
        #   U[0,0]=1, U[3,0]=1
        #   V[0,0]=1, V[3,0]=1
        #   W[0,0]=1, W[3,0]=1 (C00, C11)
        
        # M1 = (A11-A01)(B00+B01)
        #   U[3,1]=1, U[1,1]=-1
        #   V[0,1]=1, V[1,1]=1
        #   W[1,1]=1, W[3,1]=1 (C01, C11)
        
        # M2 = (A10+A11)B00
        #   U[2,2]=1, U[3,2]=1
        #   V[0,2]=1
        #   W[0,2]=1, W[2,2]=1 (C00, C10)
        
        # M3 = A00(B10-B00)
        #   U[0,3]=1
        #   V[1,3]=1, V[0,3]=-1
        #   W[0,3]=1, W[2,3]=1 (C00, C10)
        
        # M4 = A11(B11-B01)
        #   U[3,4]=1
        #   V[3,4]=1, V[1,4]=-1
        #   W[1,4]=1, W[3,4]=1 (C01, C11)
        
        # M6 = (A01-A11)(B00+B10)
        #   U[1,5]=1, U[3,5]=-1
        #   V[0,5]=1, V[1,5]=1
        #   W[0,5]=1, W[2,5]=1 (C00, C10)
        
        # M7 = (A10+A00)(B01-B11)
        #   U[2,6]=1, U[0,6]=1
        #   V[1,6]=1, V[3,6]=-1
        #   W[1,6]=1, W[3,6]=1 (C01, C11)
        
        # Now, map these to the global indices.
        # For a block with j_base and k_base:
        # A[i, j_base + j'] -> U[i*m + j_base + j']
        # B[j_base + j', k_base + k'] -> V[(j_base + j')*p + k_base + k']
        # C[i, k_base + k'] -> W[(k_base + k')*n + i]
        
        # We will add these 7 terms to the global U, V, W arrays.
        
        # M5
        U[j_base * m + 0, start_idx] += 1
        U[(j_base + 1) * m + 1, start_idx] += 1
        V[j_base * p + k_base, start_idx] += 1
        V[(j_base + 1) * p + k_base + 1, start_idx] += 1
        W[k_base * n + 0, start_idx] += 1
        W[(k_base + 1) * n + 1, start_idx] += 1
        start_idx += 1
        
        # M1
        U[j_base * m + 1, start_idx] += 1
        U[(j_base + 1) * m + 1, start_idx] += -1
        V[j_base * p + k_base, start_idx] += 1
        V[(j_base + 1) * p + k_base, start_idx] += 1
        W[k_base * n + 1, start_idx] += 1
        W[(k_base + 1) * n + 1, start_idx] += 1
        start_idx += 1
        
        # M2
        U[j_base * m + 0, start_idx] += 1
        U[(j_base + 1) * m + 0, start_idx] += 1
        V[j_base * p + k_base, start_idx] += 1
        W[k_base * n + 0, start_idx] += 1
        W[(k_base + 1) * n + 0, start_idx] += 1
        start_idx += 1
        
        # M3
        U[j_base * m + 0, start_idx] += 1
        V[(j_base + 1) * p + k_base, start_idx] += 1
        V[j_base * p + k_base, start_idx] += -1
        W[k_base * n + 0, start_idx] += 1
        W[(k_base + 1) * n + 0, start_idx] += 1
        start_idx += 1
        
        # M4
        U[(j_base + 1) * m + 1, start_idx] += 1
        V[(j_base + 1) * p + k_base + 1, start_idx] += 1
        V[(j_base + 1) * p + k_base, start_idx] += -1
        W[k_base * n + 1, start_idx] += 1
        W[(k_base + 1) * n + 1, start_idx] += 1
        start_idx += 1
        
        # M6
        U[j_base * m + 1, start_idx] += 1
        U[(j_base + 1) * m + 1, start_idx] += -1
        V[j_base * p + k_base, start_idx] += 1
        V[(j_base + 1) * p + k_base, start_idx] += 1
        W[k_base * n + 0, start_idx] += 1
        W[(k_base + 1) * n + 0, start_idx] += 1
        start_idx += 1
        
        # M7
        U[j_base * m + 0, start_idx] += 1
        U[(j_base + 1) * m + 0, start_idx] += 1
        V[(j_base + 1) * p + k_base + 1, start_idx] += 1
        V[(j_base + 1) * p + k_base, start_idx] += -1
        W[k_base * n + 1, start_idx] += 1
        W[(k_base + 1) * n + 1, start_idx] += 1
        start_idx += 1
        
        return start_idx

    # Apply Strassen to blocks (j=0,1) and k in {0,1}, {2,3}, {4,5}
    idx = add_strassen_2x2x2(0, 0, idx)
    idx = add_strassen_2x2x2(0, 2, idx)
    idx = add_strassen_2x2x2(0, 4, idx)
    
    # Block j in {0,1}, k=6: M_{2,2,1} -> 4 terms
    for i in range(2):
        for j in range(2):
            add_term(i, j, 6, 1.0, 1.0, 1.0, idx)
            idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
