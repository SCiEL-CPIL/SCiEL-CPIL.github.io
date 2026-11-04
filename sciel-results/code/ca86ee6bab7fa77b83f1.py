"""Matrix multiplication 2x3 by 3x5 with rank-28 decomposition using Strassen blocks."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 28
    
    # U: shape (n*m, rank) = (6, 28)
    # V: shape (m*p, rank) = (15, 28)
    # W: shape (p*n, rank) = (10, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Column mapping:
    # Columns 0-6: Strassen block for (A[0:2, 0:2] * B[0:2, 0:2])
    # Columns 7-10: Strassen block for (A[0:2, 0:2] * B[2, 0:5])
    # Columns 11-14: Strassen block for (A[0:2, 2] * B[0:2, 0:2])
    # Columns 15-27: Schoolbook for remaining terms
    
    col = 0
    
    # --- Strassen Block 1: A[0:2, 0:2] * B[0:2, 0:2] ---
    # A rows 0,1; A cols 0,1 -> U rows: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    # B rows 0,1; B cols 0,1,2,3 -> V rows: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # W: output C[0:2, 0:2] -> W rows: k*n+i
    #   C[0,0]: k=0,i=0 -> W[0]
    #   C[0,1]: k=1,i=0 -> W[2]
    #   C[0,2]: k=2,i=0 -> W[4]
    #   C[0,3]: k=3,i=0 -> W[6]
    #   C[1,0]: k=0,i=1 -> W[1]
    #   C[1,1]: k=1,i=1 -> W[3]
    #   C[1,2]: k=2,i=1 -> W[5]
    #   C[1,3]: k=3,i=1 -> W[7]
    
    # Strassen M_1 = (A11 + A22) * (B11 + B22)
    # A11=A[0,0]=0, A22=A[1,1]=4
    # B11=B[0,0]=0, B22=B[1,1]=6
    U[0][col] = 1   # A[0,0]
    U[4][col] = 1   # A[1,1]
    V[0][col] = 1   # B[0,0]
    V[6][col] = 1   # B[1,1]
    # Output to all of C[0:2, 0:2]
    W[0][col] = 1   # C[0,0]
    W[2][col] = 1   # C[0,1]
    W[4][col] = 1   # C[0,2]
    W[6][col] = 1   # C[0,3]
    W[1][col] = 1   # C[1,0]
    W[3][col] = 1   # C[1,1]
    W[5][col] = 1   # C[1,2]
    W[7][col] = 1   # C[1,3]
    col += 1  # col=1
    
    # Strassen M_2 = A22 * (B21 - B11)
    # A22=A[1,1]=4
    # B21=B[1,0]=5, B11=B[0,0]=0
    U[4][col] = 1   # A[1,1]
    V[5][col] = 1   # B[1,0]
    V[0][col] = -1  # B[0,0]
    # Output to C[1, 0:2]
    W[1][col] = 1   # C[1,0]
    W[3][col] = 1   # C[1,1]
    W[5][col] = 1   # C[1,2]
    W[7][col] = 1   # C[1,3]
    col += 1  # col=2
    
    # Strassen M_3 = (A11 - A21) * B22
    # A11=A[0,0]=0, A21=A[1,0]=3
    # B22=B[1,1]=6
    U[0][col] = 1   # A[0,0]
    U[3][col] = -1  # A[1,0]
    V[6][col] = 1   # B[1,1]
    # Output to C[0:2, 0:2]
    W[0][col] = 1   # C[0,0]
    W[2][col] = 1   # C[0,1]
    W[4][col] = 1   # C[0,2]
    W[6][col] = 1   # C[0,3]
    W[1][col] = 1   # C[1,0]
    W[3][col] = 1   # C[1,1]
    W[5][col] = 1   # C[1,2]
    W[7][col] = 1   # C[1,3]
    col += 1  # col=3
    
    # Strassen M_4 = A11 * (B12 - B22)
    # A11=A[0,0]=0
    # B12=B[0,1]=1, B22=B[1,1]=6
    U[0][col] = 1   # A[0,0]
    V[1][col] = 1   # B[0,1]
    V[6][col] = -1  # B[1,1]
    # Output to C[0, 0:2]
    W[0][col] = 1   # C[0,0]
    W[2][col] = 1   # C[0,1]
    W[4][col] = 1   # C[0,2]
    W[6][col] = 1   # C[0,3]
    col += 1  # col=4
    
    # Strassen M_5 = (A12 + A22) * (B11 + B12)
    # A12=A[0,1]=1, A22=A[1,1]=4
    # B11=B[0,0]=0, B12=B[0,1]=1
    U[1][col] = 1   # A[0,1]
    U[4][col] = 1   # A[1,1]
    V[0][col] = 1   # B[0,0]
    V[1][col] = 1   # B[0,1]
    # Output to C[0:2, 0:2]
    W[0][col] = 1   # C[0,0]
    W[2][col] = 1   # C[0,1]
    W[4][col] = 1   # C[0,2]
    W[6][col] = 1   # C[0,3]
    W[1][col] = 1   # C[1,0]
    W[3][col] = 1   # C[1,1]
    W[5][col] = 1   # C[1,2]
    W[7][col] = 1   # C[1,3]
    col += 1  # col=5
    
    # Strassen M_6 = (A11 - A21) * (B21 + B22)
    # A11=A[0,0]=0, A21=A[1,0]=3
    # B21=B[1,0]=5, B22=B[1,1]=6
    U[0][col] = 1   # A[0,0]
    U[3][col] = -1  # A[1,0]
    V[5][col] = 1   # B[1,0]
    V[6][col] = 1   # B[1,1]
    # Output to C[1, 0:2]
    W[1][col] = 1   # C[1,0]
    W[3][col] = 1   # C[1,1]
    W[5][col] = 1   # C[1,2]
    W[7][col] = 1   # C[1,3]
    col += 1  # col=6
    
    # Strassen M_7 = (A12 - A22) * (B11 - B21)
    # A12=A[0,1]=1, A22=A[1,1]=4
    # B11=B[0,0]=0, B21=B[1,0]=5
    U[1][col] = 1   # A[0,1]
    U[4][col] = -1  # A[1,1]
    V[0][col] = 1   # B[0,0]
    V[5][col] = -1  # B[1,0]
    # Output to C[0:2, 0:2]
    W[0][col] = 1   # C[0,0]
    W[2][col] = 1   # C[0,1]
    W[4][col] = 1   # C[0,2]
    W[6][col] = 1   # C[0,3]
    W[1][col] = 1   # C[1,0]
    W[3][col] = 1   # C[1,1]
    W[5][col] = 1   # C[1,2]
    W[7][col] = 1   # C[1,3]
    col += 1  # col=7
    
    # --- Strassen Block 2: A[0:2, 0:2] * B[2, 0:5] ---
    # A rows 0,1; A cols 0,1 -> U rows: 0, 1, 3, 4
    # B row 2; B cols 0,1,2,3,4 -> V rows: 2*5+0=10, 2*5+1=11, 2*5+2=12, 2*5+3=13, 2*5+4=14
    # W: output C[0:2, 0:5] -> W rows: k*n+i
    #   C[0,0]: k=0,i=0 -> W[0]
    #   C[0,1]: k=1,i=0 -> W[2]
    #   C[0,2]: k=2,i=0 -> W[4]
    #   C[0,3]: k=3,i=0 -> W[6]
    #   C[0,4]: k=4,i=0 -> W[8]
    #   C[1,0]: k=0,i=1 -> W[1]
    #   C[1,1]: k=1,i=1 -> W[3]
    #   C[1,2]: k=2,i=1 -> W[5]
    #   C[1,3]: k=3,i=1 -> W[7]
    #   C[1,4]: k=4,i=1 -> W[9]
    
    # For 2x1 matrix A and 1x5 matrix B, Strassen gives rank 4 (vs 2*1*5=10 schoolbook)
    # Actually for 2x1 * 1x5, we can just use 2 terms:
    # C[0, :] = A[0,0] * B[2, :]
    # C[1, :] = A[1,0] * B[2, :]
    # But that's only 2 terms for 10 outputs. Let's just use schoolbook for this part.
    # Actually, 2x1 * 1x5 has rank at most 2 (since min(2,1,5)=1... no, rank of tensor is different)
    # The tensor for 2x1 * 1x5 multiplication: C[i,k] = A[i,0] * B[0,k]
    # This is a sum of 2*5 = 10 terms schoolbook, but can be done in 2 terms?
    # No, each output C[i,k] depends on A[i,0] and B[0,k]. The tensor is:
    # T[i,k,j] = delta[i,j] * delta[k,j'] ... actually T[i,k, (j=0, k')] where j indexes A col, k' indexes B row
    # For 2x1 * 1x5: A is 2x1, B is 1x5, C is 2x5
    # T[i, k, (a, b)] where a in {0} (A col), b in {0} (B row)
    # T[i, k, (0, 0)] = 1 if we're computing C[i,k] = A[i,0]*B[0,k]
    # So the tensor has only 1 nonzero entry in the "input" dimension? No...
    # The multiplication tensor T: C[i,k] = sum_{j} A[i,j] B[j,k]
    # For n=2, m=1, p=5: C[i,k] = A[i,0] * B[0,k]
    # T[i, k, (j=0, k'=0)] = 1
    # So there's only 1 term! Rank 1.
    # Wait, that's not right. Let me reconsider.
    
    # The tensor for matrix multiplication M_{n,m,p} is in F^{n x m} * F^{m x p} * F^{p x n}
    # T[i, j, (k, l)] = 1 if... no.
    # T is a 3-tensor: T[i, j, (k, l)] where i in [n], j in [m], (k,l) in [p]x[n]
    # T[i, j, (k, l)] = 1 if the term A[i,j]*B[j,k] contributes to C[k,l]
    # C[k,l] = sum_j A[l,j] * B[j,k]
    # So A[i,j]*B[j,k] contributes to C[k,i]
    # T[i, j, (k, l)] = 1 if l == i and ... j is free, k is free
    # Actually: T[i, j, (k, l)] = 1 if l = i
    # No wait. Let me re-index.
    
    # Standard: C = A * B, where A is n x m, B is m x p, C is n x p
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T is in F^{n*m} * F^{m*p} * F^{n*p} (or with the trace convention F^{p*n})
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # So T[(i,j), (j',k), (k',i')] = 1 if j=j', k=k', i=i'
    
    # For n=2, m=1, p=5:
    # A is 2x1, B is 1x5, C is 2x5
    # C[i, k] = A[i, 0] * B[0, k]
    # T[(i,0), (0,k), (k',i')] = 1 if i=i' and k=k'
    # So for each (i, k), there's exactly one term. The tensor has 2*5 = 10 nonzero entries.
    # Can we do better? The rank of this tensor...
    # T[(i,0), (0,k), (k',i')] = delta_{i,i'} * delta_{k,k'}
    # This factors as (sum_i delta_{i,i'} A[i,0]) * (sum_k delta_{k,k'} B[0,k])
    # No, it's a single product: A[i,0] * B[0,k] gives C[i,k]
    # The tensor is T = sum_{i,k} e_i^A * e_k^B * e_{(k,i)}^C
    # = (sum_i e_i^A * e_{i,*}^C) * (sum_k e_k^B * e_{*,k}^C) ... no, that's not right.
    
    # Actually, T[(i,0), (0,k), (k',i')] = 1 if i=i' and k=k'
    # This is a product of two factors: one depending on (i, i') and one on (k, k')
    # T = (sum_i e_i * e_i') (sum_k e_k * e_k')
    # But these are in different tensor factors, so...
    # In the tensor T ∈ F^{2*1} ⊗ F^{1*5} ⊗ F^{5*2},
    # T = sum_{i=0}^{1} sum_{k=0}^{4} e_{(i,0)} ⊗ e_{(0,k)} ⊗ e_{(k,i)}
    # This is a sum of 10 rank-1 terms.
    # Can we reduce? The tensor is:
    # For each i in {0,1}, k in {0,...,4}: e_{i}^{(A)} ⊗ e_k^{(B)} ⊗ e_{(k,i)}^{(C)}
    # Note that e_{i}^{(A)} appears with all k, and e_k^{(B)} appears with all i.
    # The tensor is (sum_i e_i^A ⊗ e_i^C) ⊗ (sum_k e_k^B ⊗ e_k^C) ... no.
    
    # Let's just use 10 schoolbook terms for this block. Or better, let's see if we can share.
    # Actually, I realize I should just use schoolbook for the non-2x2 parts.
    # Let me restart with a cleaner approach.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, :] -> contributes to C[0:2, 0:2]
    # Use Strassen for the 2x2 * 2x2 part (cols 0,1 of B), rank 7
    # Then schoolbook for A[0:2, 0:2] * B[0:2, 2:5] (cols 2,3,4 of B), rank 2*2*3=12
    # Total for this block: 7 + 12 = 19? No, that's worse than schoolbook 2*2*5=20.
    # Actually Strassen 2x2*2x2 is rank 7, and schoolbook 2x2*2x3 is rank 12. Total 19.
    # But schoolbook 2x2*2x5 is rank 20. So we save 1.
    
    # Let me just do full Strassen on the 2x2 x 2x2 sub-block and schoolbook elsewhere.
    # Full schoolbook: 2*3*5 = 30
    # Block A[0:2,0:2]*B[0:2,0:2]: Strassen rank 7 (vs schoolbook 8), save 1
    # Block A[0:2,0:2]*B[2,0:5]: schoolbook 2*1*5 = 10
    # Block A[0:2,2]*B[0:2,0:2]: schoolbook 2*1*2 = 4
    # Block A[0:2,2]*B[0:2,2:5]: schoolbook 2*1*3 = 6
    # Block A[0:2,2]*B[2,0:5]: schoolbook 2*1*5 = 10
    # Total: 7 + 10 + 4 + 6 + 10 = 37. That's worse!
    
    # OK so the Strassen only saves 1 on the 2x2*2x2 block. Not worth it.
    # Let me try a different partition.
    
    # How about: treat A as 2x3, split B into 3 blocks of 5 columns each?
    # B[:, 0:5] is 3x5. Split B into B[:, 0:1], B[:, 1:2], B[:, 2:3]... no, B is 3x5.
    
    # Alternative: split the middle dimension m=3 into parts.
    # A is 2x3, B is 3x5.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Group 1: A[i,0]*B[0,k] for all i,k -> 2*5 = 10 terms
    # Group 2: A[i,1]*B[1,k] for all i,k -> 2*5 = 10 terms
    # Group 3: A[i,2]*B[2,k] for all i,k -> 2*5 = 10 terms
    # Total: 30 (schoolbook)
    
    # Can we combine? For each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
    # This is a linear combination of 3 vectors in F^5, with coefficients A[i,0], A[i,1], A[i,2].
    # The tensor for fixed i is: C[i,:] = A[i,:] * B^T (where B^T is 5x3)
    # For each i, this is a 1x3 * 3x5 multiplication, which has rank 3 (outer product structure).
    # Actually, the tensor for 1x3 * 3x5 is: T[(0,j), (j',k), (k',0)] = delta_{j,j'} * delta_{k,k'}
    # = sum_j sum_k e_j^A ⊗ e_{(j,k)}^B ⊗ e_k^C ... 
    # For 1x3 * 3x5: n=1, m=3, p=5. Rank of M_{1,3,5}?
    # M_{1,m,p} has rank m (I think). Because C[0,k] = sum_j A[0,j]*B[j,k], and this can be done in m terms.
    # Actually rank(M_{1,m,p}) = m. 
    # So for each i, rank is 3. For 2 values of i, total rank 6? No, they share the B factors.
    
    # The full tensor: T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # = sum_{j,k} (sum_i delta_{i,i'} e_{(i,j)}^A) ⊗ e_{(j,k)}^B ⊗ (sum_{i'} delta_{i',i} e_{(k,i')}^C)
    # Hmm, this is getting complicated. Let me just stick with schoolbook rank 30.
    
    # Given the difficulty of finding a valid rank < 30 decomposition, I'll return the schoolbook.
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
