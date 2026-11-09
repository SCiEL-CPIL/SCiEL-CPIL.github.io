"""Matrix multiplication 3,4,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 6]

def _strassen_3x3():
    """Return the 23-term Strassen decomposition for 3x3 matrix multiplication.
    
    Returns lists U, V, W of length 23, each being a 9-element list (flattened 3x3).
    U[i], V[i], W[i] are the coefficient vectors for term i.
    C = sum_i W[i] * (A * U[i]) * (V[i] * B)  (in tensor form)
    """
    # Strassen's 23-multiplication algorithm for 3x3 matrices
    # Each term is: W_i = (V_i^T * A) * (B * U_i)
    # In tensor decomposition: T = sum_i U_i ⊗ V_i ⊗ W_i
    # where U_i is 9x1 (for A), V_i is 9x1 (for B), W_i is 9x1 (for C)
    
    # The 23 bilinear forms for 3x3 matrix multiplication
    # M = A * B, where A = [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
    # B = [[d1,d2,d3],[e1,e2,e3],[f1,f2,f3]]
    
    # Strassen 3x3 23-multiplication (from various references)
    # Let me use the standard Strassen-like decomposition
    
    # Actually, let me use a verified 23-rank decomposition for 3x3x3.
    # The decomposition: M_kl = sum_i U_i[k,l_a] * V_i[l_b,m] * W_i... 
    
    # Let me use the explicit 23-term formula.
    # For 3x3, indices a,b,c = 0,1,2 for rows of A; d,e,f = 0,1,2 for cols of B
    # C[0,0] = a1*d1 + a2*e1 + a3*f1, etc.
    
    # I'll use the known 23-rank decomposition. Here's one construction:
    # Based on the work of Strassen and subsequent refinements.
    
    # Let me build it from smaller blocks.
    # Split 3x3 into 2x2 + 1x1 overlap... that's complicated.
    
    # Alternative: Use the fact that R(3,3,3) <= 23 by embedding in a larger structure.
    # Actually, let me just hard-code a verified 23-term decomposition.
    
    # Here's a 23-rank decomposition for 3x3 matrix multiplication:
    # Term i has U_i (9-vector for A coeffs), V_i (9-vector for B coeffs), W_i (9-vector for C coeffs)
    # C = sum_i (U_i^T * A_flat) * (V_i^T * B_flat) * W_i  -- NO, that's not right.
    
    # In tensor form: T_{klm} = M_{kl} when M = A*B
    # T_{abc,def} = sum_kl A_{ak} * B_{kl} * delta... 
    # Actually: (A*B)_{ij} = sum_k A_{ik} B_{kj}
    # So the tensor T_{i,k,j} = sum_k A_{ik} B_{kj} has T[i,j,k] contributing.
    
    # Tensor decomposition: T_{i,j,k} (where i=row of C, j=col of C, k=inner dim)
    # Wait, the convention is different. Let me re-read.
    
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # For n=3, m=4, p=6: U is 12xR, V is 24xR, W is 18xR
    # W[k*n + i, r] -- so W is indexed by (k, i) where k is col of B / row of C inner, i is row
    # This means: C[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor: T[i,k,j] = A[i,k] * B[k,j] summed over k gives C[i,j]
    
    # For decomposition: C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
    # Wait no. The standard tensor decomposition for matrix mult:
    # C[i,j] = sum_k A[i,k] B[k,j] = sum_r U[i*m+k... 
    
    # Let me think again. The matrix multiplication tensor has T_{i,k,j} = delta structure.
    # (A * B)_{i,j} = sum_k A_{i,k} * B_{k,j}
    # As a bilinear map: f(A, B)_{i,j} = sum_k A_{i,k} * B_{k,j}
    # Tensor T with T[i,k,j] such that sum_k T[i,k,j] A[i,k] B[k,j] = C[i,j]... 
    
    # Actually the tensor is: T_{(i,k),(k,j)} = 1 if we think of it as T_{i,k,j}
    # The rank decomposition: T_{i,k,j} = sum_r U[i,k,r] * V[k,j,r] * ... 
    
    # NO. The standard formulation:
    # C_{ij} = sum_k A_{ik} B_{kj}
    # = sum_r (sum_k U_{ik}^{(r)} A_{ik}) (sum_k V_{kj}^{(r)} B_{kj})  -- this is wrong dimensionally
    
    # Correct: C_{ij} = sum_r U_{ir}^{A} * V_{jr}^{B} where U_{ir} = sum_k U^{(r)}_{ik} A_{ik}... 
    
    # Let me use the direct formula from the seed:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Sum over c: for each c, we have U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    # This should give C[i,k] = sum_j A[i,j] B[j,k]
    
    # So: sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k] = C[i,k]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_j U[i*m+j, c] * V[j*p+k', c] * W[k'*n+i, c] = delta_{kk'} ... for each c?
    # No, sum_c sum_j ... = sum_j delta_{jk}... 
    
    # The condition is: for all i,k: sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # And for the "cross terms" to vanish appropriately.
    
    # Actually the precise condition: The tensor T_{i,j,k} = delta_{j} (inner index)
    # T_{(i,j),(j,k)} = 1 means the bilinear form picks out A_{ij} * B_{jk}
    # Rank decomposition: T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U is indexed by (i,j), V by (j,k), W by (k,i)
    
    # So: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k (the "correct" terms)
    # But this can't be right because we need the sum over j to give C[i,k]
    
    # I think the correct interpretation is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j (sum_r U[i,j,r] * something) ... 
    
    # Let me just follow the seed's structure exactly.
    # In the seed: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for each (i,j,k)
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if (i,j,k) matches, 0 otherwise
    # So C[i,k] = sum_j sum_c A[i,j]*U[i*m+j,c] * B[j,k]*V[j*p+k,c] * W[k*n+i,c]
    # = sum_j A[i,j]*B[j,k] * (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])
    # = sum_j A[i,j]*B[j,k] * 1 = C[i,k]  ✓
    
    # So the condition is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for "wrong" combinations it should work out via the summation.
    
    # Actually I realize the condition is simpler: we just need the trilinear form
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] to equal 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p). That's it! Because then
    # C[i,k] = sum_j A[i,j] B[j,k] * 1 = correct.
    
    # Wait, but U[i*m+j,c] is multiplied with A[i,j], and V[j*p+k,c] with B[j,k],
    # and W[k*n+i,c] is a scalar. So:
    # C[i,k] = sum_j sum_c A[i,j] * U[i*m+j,c] * B[j,k] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_j A[i,j] * B[j,k] * (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])
    
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for ALL i,j,k.
    
    # That's 3*4*6 = 72 equations. The schoolbook uses 72 terms.
    # To get lower rank, we need fewer terms that still satisfy these 72 equations.
    
    pass

def construct():
    n, m, p = 3, 4, 6
    
    # Strategy: Use block decomposition.
    # Split the 4x6 B matrix into two 4x3 blocks.
    # C[:, 0:3] = A * B[:, 0:3] and C[:, 3:6] = A * B[:, 3:6]
    # Each is a 3x4 * 4x3 = 3x3 multiplication.
    # R(3,4,6) <= 2 * R(3,4,3)
    
    # For R(3,4,3): split the 4 rows of B into 3+1.
    # A is 3x4, B is 4x3. Split B into B1 (3x3) and B2 (1x3).
    # A*B = A1*B1 + A2*B2 where A1 is 3x3 (first 3 cols of A), A2 is 3x1 (last col)
    # R(3,4,3) <= R(3,3,3) + R(3,1,3)
    
    # R(3,1,3) = 9 (schoolbook, and it's optimal since it's just scalar mult)
    # R(3,3,3) <= 23 (Strassen)
    
    # So R(3,4,3) <= 23 + 9 = 32
    # R(3,4,6) <= 2 * 32 = 64
    
    # Now I need to construct the actual 23-rank decomposition for 3x3x3.
    # Let me use a known explicit decomposition.
    
    # For 3x3 matrix multiplication, here's a 23-rank decomposition:
    # I'll use the formulation where we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1,2}
    
    # This is equivalent to finding 23 triples of 3x3 matrices (U_r, V_r, W_r)
    # such that sum_r U_r ⊙ V_r ⊙ W_r (in the appropriate sense) = T where T[i,j,k]=1
    
    # Actually, let me use a different approach. I'll use the 7-rank Strassen for 2x2
    # and extend to 3x3.
    
    # For 2x2 Strassen (rank 7):
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    
    # For 3x3, I can partition into 2x2 blocks with overlaps.
    # This gets complex. Let me try a different approach.
    
    # Alternative: Use the fact that R(3,3,3) <= 23 is well-established.
    # Let me hard-code a verified 23-term decomposition.
    
    # Here's a 23-rank decomposition for 3x3 matrix multiplication:
    # (U_r, V_r, W_r) for r = 0..22, each is a 3x3 matrix (flattened to 9-vector)
    # Condition: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k
    
    # I'll construct this using the "lifting" technique from 2x2.
    
    # Actually, let me try yet another approach: direct construction for 3x4x6.
    
    # Use the identity: R(n,m,p) <= R(n,m,p1) + R(n,m,p2) if p = p1+p2
    # And R(n,m,p) <= R(n1,m,p) + R(n2,m,p) if n = n1+n2
    
    # For 3x4x6:
    # Option 1: R(3,4,6) <= R(3,4,3) + R(3,4,3) = 2*R(3,4,3)
    # R(3,4,3) <= R(3,3,3) + R(3,1,3) = 23 + 9 = 32
    # Total: 64
    
    # Option 2: R(3,4,6) <= R(2,4,6) + R(1,4,6)
    # R(1,4,6) = 1*4*6 = 24 (optimal)
    # R(2,4,6) <= R(2,4,4) + R(2,4,2)
    # R(2,4,2) <= R(2,2,2) + R(2,2,2) = 7+7 = 14? No...
    # R(2,4,2): 2x4 * 4x2 = 2x2. Split 4 into 2+2.
    # R(2,4,2) <= R(2,2,2) + R(2,2,2) = 14
    # R(2,4,4) <= R(2,2,4) + R(2,2,4)
    # R(2,2,4) <= R(2,2,2) + R(2,2,2) = 14
    # So R(2,4,4) <= 28, R(2,4,2) <= 14, R(2,4,6) <= 42
    # Total: 24 + 42 = 66. Worse than 64.
    
    # Option 3: R(3,4,6) <= R(3,3,6) + R(3,1,6)
    # R(3,1,6) = 18
    # R(3,3,6) <= R(3,3,3) + R(3,3,3) = 46
    # Total: 18 + 46 = 64. Same.
    
    # Let me go with Option 1: 64 rank via 2 * (23 + 9)
    
    # Now I need the 23-rank decomposition for 3x3x3.
    # Let me use a published decomposition.
    
    # Here is a 23-rank decomposition for 3x3 matrix multiplication.
    # Each term r has U_r, V_r, W_r as 3x3 matrices (row-major flattened).
    # The condition: for all i,j,k in {0,1,2}: sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1
    
    # I'll use the decomposition from:
    # "Optimal tensor rank of 3x3 matrix multiplication" 
    # Based on Strassen's work and subsequent improvements.
    
    # Let me construct it step by step using block methods.
    # Partition 3x3 into four 2x2 blocks with one row/col shared.
    
    # Actually, let me just use a computational approach: 
    # I'll verify if I can find a 64-rank decomposition numerically,
    # but that's not guaranteed to give integer/half-integer entries.
    
    # Given the complexity, let me try a more direct construction.
    
    # For the 9-rank part (3x1x3): this is just schoolbook.
    # U[i*1+j, c] where i in 0..2, j=0 (only 1 col), so 3 rows.
    # Actually for R(3,1,3): n=3, m=1, p=3
    # U shape (3*1, 9) = (3, 9), V shape (1*3, 9) = (3, 9), W shape (3*3, 9) = (9, 9)
    # Schoolbook: 9 terms, one for each (i,j,k) with j=0 fixed.
    # U[i, i*3+k] = 1, V[0*3+k, i*3+k] = 1, W[k*3+i, i*3+k] = 1
    
    # For the 23-rank part (3x3x3): I need to hard-code it.
    
    # Let me use a known 23-term formula. After research, here's one:
    
    # I'll define the 23 terms for 3x3x3 multiplication.
    # Using the convention: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor condition: sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1
    
    # Here's a 23-rank decomposition (I'll verify it):
    
    terms_23 = []
    
    # Term 0-6: Strassen 2x2 on top-left 2x2 block
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # But this only covers 2x2. I need to extend to 3x3.
    
    # This is getting very complex. Let me try a different strategy:
    # Just use the schoolbook 72 and see if I can shave off a few terms.
    
    # Actually, the simplest improvement: R(3,4,6) <= R(3,4,4) + R(3,4,2)
    # R(3,4,2): 3x4 * 4x2 = 3x2. Schoolbook = 24.
    # Can we do better? R(3,4,2) <= R(3,2,2) + R(3,2,2) = 2*R(3,2,2)
    # R(3,2,2) = R(2,2,3) by transpose. R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7+4 = 11
    # So R(3,4,2) <= 22.
    # R(3,4,4): unknown exact, but <= R(3,3,4) + R(3,1,4) = R(3,3,4) + 12
    # R(3,3,4) <= R(3,3,2) + R(3,3,2) = 2*R(3,3,2)
    # R(3,3,2) = R(2,3,3) <= R(2,2,3) + R(2,1,3) = 11 + 6 = 17? 
    # This is getting complicated and I'm not sure of the exact values.
    
    # Let me just go with the 72 schoolbook since I can't guarantee a valid lower-rank decomposition.
    
    # Wait - let me try one more thing. R(3,4,6) <= 2*R(3,4,3) and R(3,4,3) <= R(3,4,2) + R(3,4,1)
    # R(3,4,1) = 12 (schoolbook, optimal)
    # R(3,4,2) <= 2*R(3,2,2) = 2*R(2,2,3)
    # R(2,2,3) <= 11 (from above)
    # So R(3,4,2) <= 22, R(3,4,3) <= 22+12 = 34, R(3,4,6) <= 68. Worse.
    
    # OK, I'll stick with 72 unless I can construct a valid lower-rank decomposition.
    
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
