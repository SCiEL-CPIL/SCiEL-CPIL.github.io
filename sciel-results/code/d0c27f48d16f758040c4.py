import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # Indices: i in {0,1}, k in {0,1}, j in {0,1}
    # U_222[i, r], V_222[k, r], W_222[j, r]
    # Strassen formulas:
    # M1 = (A11 - A22) * (B22 + B11) -> i=0, k=1, j=1 ? No.
    # Let's use the standard rank-7 decomposition tensors.
    # There are 7 rank-1 terms.
    # We can define them explicitly.
    
    # Term 1: M1 = (A00 - A11)(B01 + B11)
    # A indices: (0,0) and (1,1). B indices: (0,1) and (1,1).
    # This is not a single rank-1 term in the tensor sense of A[i,k]B[k,j].
    # The tensor is sum of A[i,k] B[k,j].
    # Strassen rewrites the sum using 7 products.
    # The decomposition of the tensor M_222 is:
    # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
    
    # Let's use the known integer decomposition for M_222.
    # U, V, W matrices of size 2x7.
    
    # One standard decomposition:
    # r=1: U[0,0]=1, U[1,0]=1; V[0,0]=1, V[1,0]=1; W[0,0]=1, W[1,0]=1 ? No.
    
    # Let's construct the rank-7 decomposition explicitly.
    # The 7 products are:
    # P1 = (A00 - A11)(B01 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B11)
    # P4 = A11(B00 - B10)
    # P5 = (A00 + A11)(B11 - B10)
    # P6 = (A01 - A00)(B00 + B10)
    # P7 = (A10 + A00)(B11 + B10)
    
    # We need to express C00, C01, C10, C11 as sums of Pr.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # This gives us the W matrix for M_222.
    # W[j, r] where j is output index (0,1) and r is term index (0..6).
    
    W_222 = np.zeros((2, 7), dtype=int)
    # P1
    W_222[0, 0] = 1
    W_222[1, 0] = 1
    # P2
    W_222[1, 1] = -1
    # P3
    W_222[0, 2] = 1
    W_222[1, 2] = 1
    # P4
    W_222[0, 3] = 1
    W_222[1, 3] = 1
    # P5
    W_222[0, 4] = -1
    W_222[1, 4] = -1
    # P6
    W_222[1, 5] = 1
    # P7
    W_222[0, 6] = 1
    
    # Now U and V.
    # P1 = (A00 - A11)(B01 + B11)
    # U[0,0] = 1, U[1,0] = -1
    # V[0,0] = 1, V[1,0] = 1  (B01 is k=0,j=1? No, B indices are k,j. B01 is k=0,j=1. B11 is k=1,j=1.)
    # Wait, B is 2x2. B[k,j].
    # B01 is k=0, j=1. B11 is k=1, j=1.
    # So V[0,0]=1, V[1,0]=1.
    
    # P2 = (A10 + A11)B00
    # U[0,1] = 0, U[1,1] = 1
    # V[0,1] = 1, V[1,1] = 0
    
    # P3 = A00(B10 - B11)
    # U[0,2] = 1, U[1,2] = 0
    # V[0,2] = 0, V[1,2] = -1
    
    # P4 = A11(B00 - B10)
    # U[0,3] = 0, U[1,3] = 1
    # V[0,3] = 1, V[1,3] = -1
    
    # P5 = (A00 + A11)(B11 - B10)
    # U[0,4] = 1, U[1,4] = 1
    # V[0,4] = -1, V[1,4] = 1
    
    # P6 = (A01 - A00)(B00 + B10)
    # U[0,5] = -1, U[1,5] = 0
    # V[0,5] = 1, V[1,5] = 1
    
    # P7 = (A10 + A00)(B11 + B10)
    # U[0,6] = 1, U[1,6] = 1
    # V[0,6] = 1, V[1,6] = 1
    
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 1],
        [-1, 1, 0, 1, 1, 0, 1]
    ], dtype=int)
    
    V_222 = np.array([
        [1, 1, 0, 1, -1, 1, 1],
        [1, 0, -1, -1, 1, 1, 1]
    ], dtype=int)
    
    # Verify M_222 decomposition
    # C = U @ diag(1) @ V^T? No.
    # The tensor is sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
    # This means C[i,j] = sum_k sum_r U[i,r] V[k,r] W[j,r] ? No.
    # The multiplication is C = A B.
    # A[i,k] = sum_r U[i,r] x_r ? No.
    # The decomposition is:
    # A[i,k] B[k,j] = sum_r U[i,r] V[k,r] W[j,r] ?
    # This is not correct. The rank-1 term is u \otimes v \otimes w.
    # The tensor T[i,k,j] = sum_r U[i,r] V[k,r] W[j,r].
    # And we require T[i,k,j] = delta_{i,i} delta_{k,k} delta_{j,j} ? No.
    # The tensor for matrix multiplication is T[i,k,j] = 1 if we are summing A[i,k]B[k,j].
    # Actually, the tensor is just the identity in the middle?
    # No, the tensor $M_{n,m,p}$ is defined such that $\langle M, A \otimes B \rangle = AB$.
    # $M_{i,k,j} = 1$ for all $i,k,j$? No.
    # $M_{i,k,j} = 1$ if the indices match the summation?
    # The tensor is $T = \sum_{i,k,j} e_i \otimes e_k \otimes e_j$.
    # So $T[i,k,j] = 1$.
    # The decomposition is $T = \sum_{r} u_r \otimes v_r \otimes w_r$.
    # So $1 = \sum_{r} U[i,r] V[k,r] W[j,r]$.
    # This must hold for all $i,k,j$.
    
    # Let's check P1 term for i=0, k=0, j=0.
    # U[0,0]=1, V[0,0]=1, W[0,0]=1. Product = 1.
    # Sum over r.
    # r=1: 1*1*1 = 1
    # r=2: 0*1*0 = 0
    # r=3: 1*0*1 = 0
    # r=4: 0*1*1 = 0
    # r=5: 1*-1*-1 = 1
    # r=6: -1*1*0 = 0
    # r=7: 1*1*1 = 1
    # Sum = 3. But T[0,0,0] should be 1.
    
    # My Strassen coefficients are for the *output* C, not the tensor T itself.
    # The tensor T is fixed. The decomposition expresses T as a sum of rank-1 tensors.
    # The standard Strassen algorithm expresses the *output* C as a linear combination of 7 products.
    # This corresponds to a decomposition of the tensor T.
    # The rank-1 tensors are $u_r \otimes v_r \otimes w_r$.
    # The coefficients in the Strassen formulas are the entries of U, V, W.
    
    # Let's re-verify the Strassen tensor decomposition.
    # The 7 rank-1 components are:
    # 1. u1 = [1, -1]^T, v1 = [1, 1]^T, w1 = [1, 1]^T ?
    #    Term: (A00-A11)(B01+B11)
    #    This product contributes to C00 and C11.
    #    C00 += 1 * P1. C11 += 1 * P1.
    #    So w1 = [1, 1]^T.
    #    u1 = [1, -1]^T.
    #    v1 = [1, 1]^T.
    #    Check T[0,0,0]: u1[0]v1[0]w1[0] = 1*1*1 = 1.
    #    Check T[0,1,0]: u1[0]v1[1]w1[0] = 1*1*1 = 1.
    #    Check T[1,0,0]: u1[1]v1[0]w1[0] = -1*1*1 = -1.
    #    Check T[1,1,0]: u1[1]v1[1]w1[0] = -1*1*1 = -1.
    
    # We need sum_r u_r[i] v_r[k] w_r[j] = 1 for all i,k,j.
    
    # Let's use a known valid rank-7 decomposition of M_222.
    # From literature (e.g., Landsberg):
    # The decomposition is not unique.
    # One common one is:
    # r=1: u=[1,1], v=[1,1], w=[1,1] ? No, that's rank 1.
    
    # Let's use the Python code to generate a valid one or use a hardcoded one that sums to 1.
    # Actually, the schoolbook is rank 8. Strassen is rank 7.
    # I will use a hardcoded rank-7 decomposition that is known to be correct.
    
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    # The 7 products are M1..M7.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The rank-1 tensors are defined by the factors of M1..M7.
    # M1 = (A00 - A11)(B01 + B11)
    # u1 = [1, -1], v1 = [1, 1], w1 = [1, 1]
    # M2 = (A10 + A11)B00
    # u2 = [0, 1], v2 = [1, 0], w2 = [0, -1]  (since C11 -= M2)
    # M3 = A00(B10 - B11)
    # u3 = [1, 0], v3 = [0, -1], w3 = [1, 1]
    # M4 = A11(B00 - B10)
    # u4 = [0, 1], v4 = [1, -1], w4 = [1, 1]
    # M5 = (A00 + A11)(B11 - B10)
    # u5 = [1, 1], v5 = [-1, 1], w5 = [-1, -1] (since C00 -= M5, C11 -= M5)
    # M6 = (A01 - A00)(B00 + B10)
    # u6 = [-1, 0], v6 = [1, 1], w6 = [0, 1]
    # M7 = (A10 + A00)(B11 + B10)
    # u7 = [1, 1], v7 = [1, 1], w7 = [1, 0]
    
    # Let's check if sum_r u_r[i] v_r[k] w_r[j] = 1.
    # i=0, k=0, j=0:
    # r1: 1*1*1 = 1
    # r2: 0*1*0 = 0
    # r3: 1*0*1 = 0
    # r4: 0*1*1 = 0
    # r5: 1*-1*-1 = 1
    # r6: -1*1*0 = 0
    # r7: 1*1*1 = 1
    # Sum = 3. Still 3.
    
    # There is a mistake in my assumption that the Strassen products directly give the tensor decomposition with integer coefficients summing to 1.
    # The tensor $T$ has entries 1.
    # The decomposition $T = \sum u_r \otimes v_r \otimes w_r$ requires $\sum u_r[i] v_r[k] w_r[j] = 1$.
    # The Strassen algorithm computes $C = AB$ using 7 scalar multiplications.
    # This implies that the bilinear map $A, B \to AB$ has rank 7.
    # This is equivalent to the tensor rank being 7.
    # The coefficients $u_r, v_r, w_r$ are the ones from the Strassen formulas.
    # Why did my sum give 3?
    # Because the Strassen formulas compute $C_{ij}$, which is $\sum_k A_{ik} B_{kj}$.
    # The tensor $T$ is such that $\langle T, A \otimes B \rangle = \sum_{i,k,j} T_{i,k,j} A_{ik} B_{kj}$.
    # If $T_{i,k,j} = 1$, then $\langle T, A \otimes B \rangle = \sum_{i,k,j} A_{ik} B_{kj}$.
    # This is NOT equal to $AB$. $AB$ is a matrix.
    # The tensor for matrix multiplication is $M_{n,m,p} = \sum_{i,k,j} e_i \otimes e_k \otimes e_j$.
    # The rank of this tensor is the rank of the matrix multiplication.
    # The Strassen algorithm shows that this tensor has rank 7 for $n=m=p=2$.
    # The decomposition is $M = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
    # This means $\sum_{r} u_r[i] v_r[k] w_r[j] = 1$ for all $i,k,j$.
    
    # Let's re-evaluate the sum for i=0, k=0, j=0 with the correct w vectors.
    # The w vectors in the Strassen algorithm are the coefficients for C.
    # C00 = M1 + M4 - M5 + M7.
    # So w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1. Others 0.
    # C11 = M1 - M2 + M3 + M6.
    # So w1[1]=1, w2[1]=-1, w3[1]=1, w6[1]=1. Others 0.
    
    # Let's check sum for i=0, k=0, j=0.
    # r1: u1[0]=1, v1[0]=1, w1[0]=1. Prod=1.
    # r2: u2[0]=0, v2[0]=1, w2[0]=0. Prod=0.
    # r3: u3[0]=1, v3[0]=0, w3[0]=0. Prod=0.
    # r4: u4[0]=0, v4[0]=1, w4[0]=1. Prod=0.
    # r5: u5[0]=1, v5[0]=-1, w5[0]=-1. Prod=1.
    # r6: u6[0]=-1, v6[0]=1, w6[0]=0. Prod=0.
    # r7: u7[0]=1, v7[0]=1, w7[0]=1. Prod=1.
    # Sum = 3.
    
    # It seems the Strassen decomposition does NOT satisfy $\sum u_r[i] v_r[k] w_r[j] = 1$.
    # Why?
    # Because the tensor $M$ is not the identity tensor.
    # The tensor $M_{2,2,2}$ has $M_{i,k,j} = 1$ if $i,j$ are output indices and $k$ is the summation index?
    # No, the tensor is $M = \sum_{i,k,j} e_i \otimes e_k \otimes e_j$.
    # This tensor, when contracted with $A \otimes B$, gives $\sum_{i,k,j} A_{ik} B_{kj}$.
    # This is a scalar, not the matrix $AB$.
    # The matrix multiplication tensor is usually defined as $M_{n,m,p} = \sum_{i=1}^n \sum_{k=1}^m \sum_{j=1}^p e_i \otimes e_k \otimes e_j$.
    # The rank of this tensor is the rank of the bilinear map $(A,B) \mapsto AB$.
    # The Strassen algorithm computes $AB$ using 7 multiplications.
    # This means the bilinear map has rank 7.
    # The decomposition of the tensor is $M = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
    # This implies $\sum_{r} u_r[i] v_r[k] w_r[j] = 1$ for all $i,k,j$.
    
    # My calculation gave 3. This implies my u,v,w vectors are wrong or the tensor definition is different.
    # Actually, the tensor $M$ is such that $M_{i,k,j} = 1$.
    # If I use the Strassen vectors, the sum is not 1.
    # This means the Strassen vectors are for a *different* tensor?
    # No, they are for matrix multiplication.
    # Let's check the sum for i=0, k=0, j=1.
    # r1: 1*1*1 = 1 (w1[1]=1)
    # r5: 1*-1*-1 = 1 (w5[1]=-1)
    # r7: 1*1*0 = 0 (w7[1]=0)
    # Sum = 2.
    
    # There is a known issue: The Strassen decomposition is for the *output* matrix.
    # The tensor rank is the minimum number of multiplications.
    # The decomposition $M = \sum u_r \otimes v_r \otimes w_r$ is correct.
    # My vectors must be wrong.
    
    # I will use a simple rank-8 schoolbook decomposition for $2 \times 2 \times 2$ to be safe, then multiply by rank-4 for $1 \times 2 \times 2$.
    # Rank 8 * 4 = 32. This is the same as the seed.
    
    # To get an improvement, I MUST use rank 7 for $2 \times 2 \times 2$.
    # I will hardcode a verified rank-7 decomposition.
    
    # Verified decomposition from a reliable source:
    # u1 = [1, 1], v1 = [1, 1], w1 = [1, 1]
    # u2 = [1, -1], v2 = [1, 1], w2 = [1, -1]
    # u3 = [1, 0], v3 = [1, 1], w3 = [1, 0]
    # u4 = [0, 1], v4 = [1, 1], w4 = [0, 1]
    # u5 = [1, 1], v5 = [1, 0], w5 = [0, 1]
    # u6 = [1, -1], v6 = [1, 0], w6 = [1, 0]
    # u7 = [1, 0], v7 = [0, 1], w7 = [1, 1]
    
    # Let's check sum for i=0, k=0, j=0.
    # r1: 1*1*1 = 1
    # r2: 1*1*1 = 1
    # r3: 1*1*1 = 1
    # r4: 0*1*0 = 0
    # r5: 1*1*0 = 0
    # r6: 1*1*1 = 1
    # r7: 1*0*1 = 0
    # Sum = 4.
    
    # I will use the seed program's schoolbook decomposition for the $2 \times 2 \times 2$ part as well, but I will try to find a rank 14 decomposition for the whole $2 \times 4 \times 4$?
    # No, rank 28 is a safe improvement.
    # I will use the schoolbook rank 8 for $2 \times 2 \times 2$ and rank 4 for $1 \times 2 \times 2$.
    # Total rank 32.
    
    # Wait, the seed is rank 32.
    # If I use rank 7 for $2 \times 2 \times 2$, I get rank 28.
    # I will use a known correct rank-7 decomposition.
    
    # Correct rank-7 decomposition of M_222:
    # See https://www.researchgate.net/publication/220354978
    # I will use the one from the Wikipedia article on Strassen's algorithm, carefully mapped.
    
    # If I cannot verify the rank-7 decomposition in code, I will stick to rank 32.
    # But rank 28 is better.
    
    # Let's try to construct the rank 28 decomposition using the Kronecker product of the schoolbook rank 8 and rank 4.
    # This gives 32.
    
    # I will output the schoolbook decomposition for now, as I cannot guarantee the correctness of the Strassen vectors in this short time.
    # However, the task asks for a stronger algorithm.
    # I will use the Strassen decomposition for $2 \times 2 \times 2$ with the vectors:
    
    # U_222 = [[1, 1, 1, 0, 1, 1, 1],
    #           [1, -1, 0, 1, 1, -1, 0]]
    # V_222 = [[1, 1, 1, 1, 1, 1, 0],
    #           [1, 1, 1, 1, 0, 0, 1]]
    # W_222 = [[1, 1, 1, 0, 0, 1, 1],
    #           [1, -1, 0, 1, 1, 0, 1]]
    
    # This is a guess.
    
    # I will use the seed program's logic but with the Kronecker product of rank 7 and rank 4.
    # If the rank 7 decomposition is invalid, the result will be invalid.
    # To be safe, I will use the schoolbook decomposition (rank 32).
    
    return construct()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
