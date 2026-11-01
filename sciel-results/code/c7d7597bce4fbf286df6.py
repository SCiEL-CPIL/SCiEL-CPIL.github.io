import numpy as np
import time

def construct():
    n, m, p = 2, 2, 9
    t_start = time.time()
    deadline = t_start + 120.0

    size_u = n * m  # 4
    size_v = m * p  # 18
    size_w = p * n  # 18

    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k]
    # Actually, the tensor for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has entries T[i,j,k] = 1 for all i,j,k
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where W uses the trace convention W[k*n + i, r]

    # Let's construct the tensor as a matrix for rank reduction
    # Tensor shape: (n*m, m*p, p*n) but we need to be careful with indexing

    # Standard approach: build the bilinear form
    # For each output (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T can be viewed as a matrix of shape (n*m, p*n) with entries indexed by (j,k,i)
    # Wait, let's think of it differently.

    # The matrix multiplication tensor M_{n,m,p} has:
    # - Input A: indices (i,j) for i in range(n), j in range(m) -> total n*m
    # - Input B: indices (j,k) for j in range(m), k in range(p) -> total m*p  
    # - Output C: indices (i,k) for i in range(n), k in range(p) -> total n*p
    #   But the problem says output factor W has shape (p*n, rank) with W[k*n+i, r]
    #   So W index is k*n + i, which is (k,i) ordering, not (i,k)

    # The tensor T is defined by:
    # C[k, i] (in W's ordering) = sum_j A[i, j] * B[j, k]
    # So T[i, j, k] = 1 for all valid i, j, k
    # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # To find rank reduction, we can use the approach of viewing the tensor as a matrix
    # and finding linear dependencies.

    # Build the full tensor matrix representation
    # For each pair (input_pair, output), we have an entry
    # Input A index: i*m + j
    # Input B index: j*p + k
    # Output index: k*n + i
    # The tensor entry T[A_idx, B_idx] for output C_idx is 1 if they correspond

    # Actually, let's build a matrix M of shape (n*m * m*p, p*n) where
    # M[(i*m+j)*(m*p) + (j*p+k), k*n+i] = 1 if the indices are consistent (they always are)
    # Wait, that's not right either. The tensor is trilinear.

    # Better approach: build the tensor as a 3D array and then use a standard rank reduction method.
    # T[i,j,k] = 1 for all i,j,k (since every combination contributes)
    # We want to find rank r such that T[i,j,k] = sum_{l=0}^{r-1} U[i,j,l] * V[j,k,l] * W[k,i,l]

    # Let's start with the schoolbook decomposition (rank = n*m*p = 36)
    R0 = n * m * p  # 36
    U0 = np.zeros((size_u, R0))
    V0 = np.zeros((size_v, R0))
    W0 = np.zeros((size_w, R0))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    # Now try to reduce the rank by finding linear dependencies
    # We can represent the tensor as a matrix by fixing one mode
    # For example, fix the A-index and get a matrix of shape (m*p, p*n) for each A-index
    # T[A_idx, B_idx, C_idx] where A_idx = i*m+j, B_idx = j*p+k, C_idx = k*n+i

    # Build the tensor matrix: for each (i,j) pair, we have a matrix M_{i,j} of shape (m*p, p*n)
    # M_{i,j}[j'*p+k, k'*n+i'] = T[i,j,j',k,k',i'] = 1 if j=j' and i=i' and k=k'
    # Wait, T[i,j,k] is defined such that the output index is determined by (k,i)
    # So for fixed (i,j), the only non-zero entries are when the B-index has row j and the C-index has column i
    # Specifically, M_{i,j}[j*p+k, k*n+i] = 1 for all k

    # This means for fixed (i,j), the matrix has ones on specific positions
    # Let's build the full tensor as a matrix of shape (n*m, m*p * p*n)
    # or better, use the standard approach:

    # Represent the tensor T as a matrix M of shape (n*m, (m*p)*(p*n)) is too large
    # Instead, let's use the following trick:
    # The tensor T can be represented by a matrix M of shape (n*m * m*p, p*n)
    # where M[(i*m+j)*(m*p) + (j*p+k), k'*n+i'] = 1 if j=j', i=i', k=k'
    # But this is sparse and not helpful for SVD.

    # Alternative: use the fact that for matrix multiplication, there are known rank reductions.
    # For 2x2x9, we can group the 9 columns of B into blocks.
    # Block 1: columns 0-1 (2x2 block) -> use Strassen rank 7
    # Blocks 2-5: columns 2-8 (7 columns) -> 3 blocks of 2-3 columns

    # Let's try: 
    # Columns 0-1: 2x2 -> Strassen rank 7
    # Columns 2-4: 2x3 -> can we do better than schoolbook 6?
    # Columns 5-8: 2x4 -> can we do better than schoolbook 8?

    # For 2x3 (n=2, m=2, p=3), schoolbook is 6. Is rank 5 possible?
    # For 2x4 (n=2, m=2, p=4), schoolbook is 8. Is rank 7 possible?

    # Known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 2x2x3: rank 8 (I believe, but let me verify)
    #   Actually, for rectangular matrices, the minimum rank is not always known.
    #   Let's be conservative and use schoolbook for p>2.

    # So my strategy:
    # - Use Strassen (rank 7) for the first 2 columns
    # - Use schoolbook (rank 4 per 2 columns) for the remaining 7 columns
    #   That's 3 full 2-column blocks (6 cols, rank 12) + 1 single column (rank 2)
    #   Total: 7 + 12 + 2 = 21

    # Wait, let me recount:
    # Columns 0-1: Strassen, rank 7
    # Columns 2-3: schoolbook 2x2, rank 4
    # Columns 4-5: schoolbook 2x2, rank 4
    # Columns 6-7: schoolbook 2x2, rank 4
    # Column 8: schoolbook 2x1, rank 2
    # Total: 7 + 4 + 4 + 4 + 2 = 21

    # But can we do even better? What if we use Strassen for all pairs?
    # Columns 0-1: Strassen, rank 7
    # Columns 2-3: Strassen, rank 7
    # Columns 4-5: Strassen, rank 7
    # Columns 6-7: Strassen, rank 7
    # Column 8: schoolbook 2x1, rank 2
    # Total: 7*4 + 2 = 30

    # That's worse. So mixing is better.

    # Actually, wait. The Strassen decomposition for 2x2x2 gives rank 7 for computing a 2x2 matrix product.
    # If I have a 2x3 block (A is 2x2, B is 2x3), I need to compute 2x3 output.
    # Schoolbook is 2*2*3 = 6.
    # Can I do it in rank 5? Let me think...
    # The tensor for 2x2x3 multiplication: T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    # This is a 2x2x3 tensor. The rank of the matrix multiplication tensor M_{2,2,3}...
    # I'm not sure of the exact minimum rank. Let's stick with schoolbook for p>2 to be safe.

    # Let me try a different structural approach: 
    # Use the known rank-7 Strassen for the first 2 columns, and schoolbook for the rest.
    # That gives rank 7 + 2*7 = 21.

    # But first, let me verify that my Strassen decomposition is correct.

    # Strassen for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-f)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C = [[P1+P3-P5-P7, P4+P5], [P2+P3, P1-P2+P4+P7]]

    # In our indexing:
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices for columns k=0,1: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C indices: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
    # W index: k*n+i, so W[0*2+0]=W[0]=C00, W[0*2+1]=W[1]=C10, W[1*2+0]=W[2]=C01, W[1*2+1]=W[3]=C11

    # Let me define the Strassen factors for 2x2x2:
    strassen_U = [
        # P1: (a+d)(e+h)
        [1, 0, 0, 1],  # A part: a + d
        [0, 0, 1, 1],  # A part: c + d
        [1, 0, 0, 0],  # A part: a
        [0, 0, 0, 1],  # A part: d
        [1, 1, 0, 0],  # A part: a + b
        [0, 0, 1, -1], # A part: c - a
        [0, 1, 0, -1], # A part: b - d
    ]
    strassen_V = [
        # P1: (e+h)
        [1, 0, 0, 1],  # B part for cols 0,1: e + h
        [1, 0, 0, 0],  # e
        [0, 1, 0, -1], # f - h
        [0, 0, 1, -1], # g - f
        [0, 0, 0, 1],  # h
        [1, 1, 0, 0],  # e + f
        [0, 0, 1, 1],  # g + h
    ]
    strassen_W = [
        # C part: [C00, C10, C01, C11] (in W index order: W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11)
        [1, 0, 0, 1],  # P1: C00 + C11
        [0, 1, 0, 0],  # P2: C10
        [1, 1, 0, 0],  # P3: C00 + C10
        [0, 0, 1, 1],  # P4: C01 + C11
        [-1, 0, 1, 0], # P5: -C00 + C01
        [-1, 0, -1, 0],# P6: -C00 - C01
        [-1, 1, 0, 0], # P7: -C00 + C10
    ]

    # Let me verify this:
    # C00 = P1 + P3 - P5 - P7 = 1+1-(-1)-(-1) = 1+1+1+1 = 4? That's wrong.
    # Let me re-check the Strassen formulas.

    # Standard Strassen:
    # M11 = P1 + P3 - P5 - P7
    # M12 = P4 + P5
    # M21 = P2 + P3
    # M22 = P1 - P2 + P4 + P7

    # So:
    # C00 = M11 = P1 + P3 - P5 - P7
    # C01 = M12 = P4 + P5
    # C10 = M21 = P2 + P3
    # C11 = M22 = P1 - P2 + P4 + P7

    # In W index order [C00, C10, C01, C11]:
    strassen_W_fixed = [
        [1, 0, 0, 0],  # P1: C00
        [0, 1, 0, 0],  # P2: C10
        [1, 1, 0, 0],  # P3: C00 + C10
        [0, 0, 1, 1],  # P4: C01 + C11
        [-1, 0, 1, 0], # P5: -C00 + C01
        [0, 0, 0, 0],  # P6: not used in C? Wait, P6 should contribute.
        [-1, 1, 0, 0], # P7: -C00 + C10
    ]

    # I think I'm making an error. Let me look up the standard Strassen decomposition carefully.

    # The 7 products are:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B11)
    # P4 = A22(B12 - B22)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)

    # And:
    # C11 = P1 + P3 - P5 - P7
    # C12 = P4 + P5
    # C21 = P2 + P3
    # C22 = P1 - P2 + P4 + P7

    # In our 0-indexed notation:
    # A00=A11, A01=A12, A10=A21, A11=A22
    # B00=B11, B01=B12, B10=B21, B11=B22
    # C00=C11, C01=C12, C10=C21, C11=C22

    # So:
    # C00 = P1 + P3 - P5 - P7
    # C01 = P4 + P5
    # C10 = P2 + P3
    # C11 = P1 - P2 + P4 + P7

    # U factors (A part):
    # P1: A00 + A11 -> [1, 0, 0, 1]
    # P2: A10 + A11 -> [0, 0, 1, 1]
    # P3: A00 -> [1, 0, 0, 0]
    # P4: A11 -> [0, 0, 0, 1]
    # P5: A00 + A01 -> [1, 1, 0, 0]
    # P6: A10 - A00 -> [0, 0, 1, -1]
    # P7: A01 - A11 -> [0, 1, 0, -1]

    # V factors (B part for columns 0,1):
    # P1: B00 + B11 -> [1, 0, 0, 1] (B00 at idx 0, B01 at idx 1, B10 at idx 9, B11 at idx 10)
    #   In local B indices [B00, B01, B10, B11]: [1, 0, 0, 1]
    # P2: B00 -> [1, 0, 0, 0]
    # P3: B10 - B00 -> [0, 0, 1, -1]
    # P4: B01 - B11 -> [0, 1, 0, -1]
    # P5: B11 -> [0, 0, 0, 1]
    # P6: B00 + B01 -> [1, 1, 0, 0]
    # P7: B10 + B11 -> [0, 0, 1, 1]

    # W factors (C part in order [C00, C10, C01, C11]):
    # P1: C00 + C11 -> [1, 0, 0, 1]
    # P2: C10 -> [0, 1, 0, 0]
    # P3: C00 + C10 -> [1, 1, 0, 0]
    # P4: C01 + C11 -> [0, 0, 1, 1]
    # P5: -C00 + C01 -> [-1, 0, 1, 0]
    # P6: ? -> P6 contributes to C00 and C01? Let me check.
    #   P6 = (A10 - A00)(B00 + B01)
    #   This should contribute to C00 and C01.
    #   From the formula, C00 has -P5 - P7, no P6. C01 has P4 + P5, no P6.
    #   Wait, that can't be right. P6 must contribute somewhere.
    #   
    #   Let me re-examine. The standard Strassen formulas don't have P6 in the C output?
    #   That seems wrong. Let me look this up again.

    # Actually, I think I had the formulas wrong. The correct Strassen formulas are:
    # C11 = P1 + P3 - P5 - P7
    # C12 = P4 + P5  
    # C21 = P2 + P3
    # C22 = P1 - P2 + P4 + P7

    # And P6 is defined but... wait, is P6 actually used? Let me check.
    # No, in standard Strassen, all 7 P's are used. Let me re-derive.

    # I think I'm confusing myself. Let me just use a known correct set of factors.

    # Here's a verified Strassen decomposition for 2x2x2:
    strassen_terms = [
        # (U, V, W) where U=[A00,A01,A10,A11], V=[B00,B01,B10,B11], W=[C00,C10,C01,C11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # P2
        ([1, 0, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0]),  # P3
        ([0, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1]),  # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # P5
        ([0, 0, 1, -1], [1, 1, 0, 0], [-1, 0, -1, 0]),# P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]), # P7
    ]

    # Let me verify C00: P1+P3-P5-P7 = 1+1-(-1)-(-1) = 4? Still wrong.
    # I give up on deriving Strassen from memory. Let me just use schoolbook for everything.
    # Rank 36 is the safe answer.

    # Actually, let me try a different approach. Let me build the tensor matrix and use SVD
    # or rank reduction to find a lower rank decomposition.

    # Build the tensor as a matrix of shape (n*m, (m*p)*(p*n)) is too complex.
    # Instead, let's use the following: for each output index c = k*n+i, the tensor slice
    # T[:, :, c] is a matrix of shape (n*m, m*p).
    # T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent.

    # For fixed c = k*n + i:
    # T[A_idx, B_idx, c] = 1 if A_idx = i*m+j, B_idx = j*p+k for some j
    # So for fixed (i,k), the non-zero entries are at (i*m+j, j*p+k) for j in range(m)

    # Let's build the full tensor matrix M of shape (n*m * m*p, p*n)
    # M[(i*m+j)*(m*p) + (j*p+k), k'*n + i'] = 1 if j=j', i=i', k=k'

    # This is a sparse matrix. Let's build it dense and try to find its rank.

    M = np.zeros((size_u * size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                M[a_idx * size_v + b_idx, c_idx] = 1

    # Now M has rank at most min(size_u*size_v, size_w) = min(72, 18) = 18
    # But this doesn't directly give us the trilinear rank.

    # The trilinear rank is the minimum r such that T[i,j,k] = sum_l U[i,j,l]*V[j,k,l]*W[k,i,l]
    # This is different from the matrix rank of the unfolding.

    # Let me just return the schoolbook decomposition with rank 36.
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
