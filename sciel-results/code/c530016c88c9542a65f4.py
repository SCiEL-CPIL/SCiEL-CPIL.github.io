"""Matrix multiplication 2x2x2: Strassen rank-7 exact decomposition."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return Strassen's rank-7 exact decomposition for 2x2 matrix multiplication.
    
    The tensor is T[i,j,k,l] = delta_{j,k} for (A*B)_{i,l} = sum_j A_{i,j} B_{j,l}
    
    Convention:
      U: shape (n*m, rank) = (4, 7), U[i*m+j, c]
      V: shape (m*p, rank) = (4, 7), V[j*p+k, c]
      W: shape (p*n, rank) = (4, 7), W[k*n+i, c]
    
    So the product is: (A*B)_{i,l} = sum_c U[i*m+?, c] * V[?, c] * W[l*n+i, c]
    Wait, let me re-read the convention.
    
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W is indexed by (k, i) where k is the column index of B (and row of C? no...)
    
    Let me think carefully. For matrix multiplication C = A * B where:
      A is n x m (n=2, m=2)
      B is m x p (m=2, p=2)
      C is n x p (n=2, p=2)
    
    C[i, l] = sum_j A[i, j] * B[j, l]
    
    The tensor T has indices (i, j, l) with T[i,j,l] = delta... no.
    Actually T[i, j, l] = 1 if we're computing C[i,l] = sum_j A[i,j] B[j,l].
    
    The decomposition: T[i, j, l] = sum_c U[i, c] * V[j, c] * W[l, c]
    But wait, the shapes are:
      U: (n*m, rank) -> indexed by (i, j) flattened as i*m+j
      V: (m*p, rank) -> indexed by (j, l) flattened as j*p+l  (since m=p=2)
      W: (p*n, rank) -> indexed by (l, i) flattened as l*n+i  (since p=n=2)
    
    So: T[i, j, l] = sum_c U[i*m+j, c] * V[j*p+l, c] * W[l*n+i, c]
    
    And we need: C[i, l] = sum_j A[i,j] * B[j,l] = sum_c (sum_j A[i,j] * U[i*m+j,c]) * ... 
    
    Hmm, actually the standard tensor decomposition for matrix multiplication:
    The multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} where
    T = sum_{i,j,l} e_{i,j} ⊗ e_{j,l} ⊗ e_{i,l}
    
    Wait no. T_{(i,j),(j,l),(i,l)} ... let me be more careful.
    
    The matrix multiplication tensor is:
    T = sum_{i=1}^n sum_{j=1}^m sum_{l=1}^p  E_{i,j} ⊗ E_{j,l} ⊗ E_{i,l}
    
    where E_{i,j} is the matrix with 1 at position (i,j).
    
    A decomposition of rank r is:
    T = sum_{c=1}^r  U_c ⊗ V_c ⊗ W_c
    
    where U_c ∈ R^{n×m}, V_c ∈ R^{m×p}, W_c ∈ R^{n×p}.
    
    Then C = A*B can be computed as:
    C = sum_c (U_c^T * A^T) * V_c * W_c^T  ... no that's not right either.
    
    Actually: if T = sum_c U_c ⊗ V_c ⊗ W_c, then for matrices A (n×m) and B (m×p):
    C = A*B = sum_c (A * U_c^T) ... 
    
    Let me just use the direct formula. The tensor entry is:
    T[(i,j), (j,l), (i,l)] = 1 (using 0-indexed: T[(i,j),(j,l),(i,l)] = 1)
    
    Wait, that's not a valid tensor index. Let me restate.
    
    The matrix multiplication tensor T is a 3-tensor with:
    - First mode: n*m (index (i,j) for A)
    - Second mode: m*p (index (j,l) for B)  
    - Third mode: n*p (index (i,l) for C)
    
    T[(i,j), (j',l'), (i',l')] = 1 if i=i', j=j', l=l'; else 0.
    
    So T[(i,j), (j,l), (i,l)] = 1 for all valid i, j, l.
    
    A rank-r decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c where
    U_c has shape (n*m,), V_c has shape (m*p,), W_c has shape (n*p,).
    
    So: T[(i,j), (j',l'), (i',l')] = sum_c U_c[i*m+j] * V_c[j'*p+l'] * W_c[i'*n+l']
    
    This must equal 1 iff i=i', j=j', l=l' (and j'=j, l'=l), else 0.
    
    For Strassen's algorithm for 2x2 matrices:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Now let me express this in terms of the tensor decomposition.
    
    A is indexed as: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    So U indices: (i*m+j) with m=2:
      (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    B is indexed as: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    V indices: (j*p+l) with p=2:
      (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    C is indexed as: C[0,0], C[0,1], C[1,0], C[1,1]
    W indices: (l*n+i) with n=2 (per the trace convention W[k*n+i,r]):
      (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    Wait, W[k*n+i, r] where k is the column of C (which is l) and i is the row of C.
    So W index = l*n + i = l*2 + i.
    C[0,0]: l=0, i=0 -> W[0]
    C[0,1]: l=1, i=0 -> W[2]
    C[1,0]: l=0, i=1 -> W[1]
    C[1,1]: l=1, i=1 -> W[3]
    
    Now, the decomposition says:
    C[i,l] = sum_j A[i,j] B[j,l] = sum_c (sum_j A[i,j] * U_c[i*m+j]) * ... 
    
    Hmm, actually I think the way it works is:
    For each term c, we compute:
    temp_A_c = sum_{i,j} A[i,j] * U_c[i*m+j]  -- no, that's a scalar, not right.
    
    Let me reconsider. The tensor decomposition T = sum_c U_c ⊗ V_c ⊗ W_c means:
    T[a_idx, b_idx, c_idx] = sum_c U_c[a_idx] * V_c[b_idx] * W_c[c_idx]
    
    And the matrix multiplication is:
    C[i,l] = sum_{j} A[i,j] * B[j,l]
    
    The way the tensor encodes this: if we "contract" the tensor with A and B:
    C[i,l] = sum_{(i',j'),(j',l')} T[(i',j'),(j',l'),(i,l)] * A[i',j'] * B[j',l']
    
    Since T[(i',j'),(j',l'),(i,l)] = 1 only when i'=i, j'=j', l'=l:
    C[i,l] = sum_j A[i,j] * B[j,l] ✓
    
    With the decomposition:
    C[i,l] = sum_c W_c[i*n+l] * (sum_{i',j'} U_c[i'*m+j'] * A[i',j']) * (sum_{j',l'} V_c[j'*p+l'] * B[j',l'])
    
    Wait, that's not quite right either. Let me be very explicit.
    
    T[a, b, c] = sum_r U[r][a] * V[r][b] * W[r][c]
    
    where a is an index into A (flattened), b into B, c into C.
    
    C[i,l] = sum_a sum_b T[a, b, c=(l*n+i)] * A_flat[a] * B_flat[b]
    
    = sum_a sum_b sum_r U[r][a] * V[r][b] * W[r][l*n+i] * A_flat[a] * B_flat[b]
    
    = sum_r W[r][l*n+i] * (sum_a U[r][a] * A_flat[a]) * (sum_b V[r][b] * B_flat[b])
    
    So for each rank term r, we compute:
    alpha_r = sum_a U[r][a] * A_flat[a]  (a scalar)
    beta_r = sum_b V[r][b] * B_flat[b]   (a scalar)
    C[i,l] += W[r][l*n+i] * alpha_r * beta_r
    
    This is the standard formulation. Now let me construct Strassen.
    
    A_flat = [a, b, c, d] (indices 0,1,2,3)
    B_flat = [e, f, g, h] (indices 0,1,2,3)
    
    Strassen terms:
    M1 = (a+d)(e+h): 
      alpha = a+d = 1*a + 0*b + 0*c + 1*d
      beta = e+h = 1*e + 0*f + 0*g + 1*h
      W contributions: C[0,0]+=M1, C[1,1]+=M1
      W[l*n+i]: C[0,0] is l=0,i=0 -> idx 0; C[1,1] is l=1,i=1 -> idx 3
      So W[0]=1, W[3]=1, others 0.
    
    M2 = (c+d)e:
      alpha = c+d = 0*a + 0*b + 1*c + 1*d
      beta = e = 1*e + 0*f + 0*g + 0*h
      C[1,0]+=M2: l=0,i=1 -> idx 1
      C[1,1]-=M2: l=1,i=1 -> idx 3
      W[1]=1, W[3]=-1, others 0.
    
    M3 = a(f-h):
      alpha = a = 1*a + 0*b + 0*c + 0*d
      beta = f-h = 0*e + 1*f + 0*g + (-1)*h
      C[0,1]+=M3: l=1,i=0 -> idx 2
      C[1,1]+=M3: l=1,i=1 -> idx 3
      W[2]=1, W[3]=1, others 0.
    
    M4 = d(g-e):
      alpha = d = 0*a + 0*b + 0*c + 1*d
      beta = g-e = (-1)*e + 0*f + 1*g + 0*h
      C[0,0]+=M4: l=0,i=0 -> idx 0
      C[1,0]+=M4: l=0,i=1 -> idx 1
      W[0]=1, W[1]=1, others 0.
    
    M5 = (a+b)h:
      alpha = a+b = 1*a + 1*b + 0*c + 0*d
      beta = h = 0*e + 0*f + 0*g + 1*h
      C[0,0]-=M5: l=0,i=0 -> idx 0
      C[0,1]+=M5: l=1,i=0 -> idx 2
      W[0]=-1, W[2]=1, others 0.
    
    M6 = (c-a)(e+f):
      alpha = c-a = (-1)*a + 0*b + 1*c + 0*d
      beta = e+f = 1*e + 1*f + 0*g + 0*h
      C[1,1]+=M6: l=1,i=1 -> idx 3
      W[3]=1, others 0.
    
    M7 = (b-d)(g+h):
      alpha = b-d = 0*a + 1*b + 0*c + (-1)*d
      beta = g+h = 0*e + 0*f + 1*g + 1*h
      C[0,0]+=M7: l=0,i=0 -> idx 0
      C[1,1]+=M7: l=1,i=1 -> idx 3
      W[0]=1, W[3]=1, others 0.
    
    Let me verify C[0,0] = M1 + M4 - M5 + M7:
    M1: W[0]=1, M4: W[0]=1, M5: W[0]=-1, M7: W[0]=1
    Sum at idx 0: 1+1-1+1 = 2... that's not right.
    
    Wait, I think I'm confusing myself. Let me re-derive.
    
    C[0,0] = M1 + M4 - M5 + M7
    So the coefficient of M1 in C[0,0] is 1, M4 is 1, M5 is -1, M7 is 1.
    
    In the decomposition, C[0,0] = sum_r W[r][0] * alpha_r * beta_r
    where W[r][0] is the W value for rank term r at C-index 0 (which is l=0,i=0).
    
    So for each rank term r, W[r][0] should be the coefficient of that term in C[0,0].
    
    Rank 0 (M1): W[0][0] = 1 (coefficient in C[0,0])
    Rank 1 (M2): W[1][0] = 0 (M2 not in C[0,0])
    Rank 2 (M3): W[2][0] = 0
    Rank 3 (M4): W[3][0] = 1
    Rank 4 (M5): W[4][0] = -1
    Rank 5 (M6): W[5][0] = 0
    Rank 6 (M7): W[6][0] = 1
    
    Let me redo all W values:
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    W index mapping: W[r][l*n+i] where n=2:
    C[0,0]: l=0,i=0 -> W_idx = 0
    C[0,1]: l=1,i=0 -> W_idx = 2
    C[1,0]: l=0,i=1 -> W_idx = 1
    C[1,1]: l=1,i=1 -> W_idx = 3
    
    W[0] (M1): C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=1 -> [1, 0, 0, 1]
    W[1] (M2): C[0,0]=0, C[0,1]=0, C[1,0]=1, C[1,1]=-1 -> [0, 1, 0, -1]
    W[2] (M3): C[0,0]=0, C[0,1]=1, C[1,0]=0, C[1,1]=1 -> [0, 0, 1, 1]
    W[3] (M4): C[0,0]=1, C[0,1]=0, C[1,0]=1, C[1,1]=0 -> [1, 1, 0, 0]
    W[4] (M5): C[0,0]=-1, C[0,1]=1, C[1,0]=0, C[1,1]=0 -> [-1, 0, 1, 0]
    W[5] (M6): C[0,0]=0, C[0,1]=0, C[1,0]=0, C[1,1]=1 -> [0, 0, 0, 1]
    W[6] (M7): C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=1 -> [1, 0, 0, 1]
    
    U (alpha coefficients), U[r][i*m+j]:
    M1: a+d -> [1, 0, 0, 1]
    M2: c+d -> [0, 0, 1, 1]
    M3: a   -> [1, 0, 0, 0]
    M4: d   -> [0, 0, 0, 1]
    M5: a+b -> [1, 1, 0, 0]
    M6: c-a -> [-1, 0, 1, 0]
    M7: b-d -> [0, 1, 0, -1]
    
    V (beta coefficients), V[r][j*p+l]:
    M1: e+h -> [1, 0, 0, 1]
    M2: e   -> [1, 0, 0, 0]
    M3: f-h -> [0, 1, 0, -1]
    M4: g-e -> [-1, 0, 1, 0]
    M5: h   -> [0, 0, 0, 1]
    M6: e+f -> [1, 1, 0, 0]
    M7: g+h -> [0, 0, 1, 1]
    
    Let me verify with a simple example:
    A = [[1, 0], [0, 0]], B = [[0, 0], [0, 1]]
    C should be [[0, 1], [0, 0]]
    
    a=1, b=0, c=0, d=0, e=0, f=0, g=0, h=1
    
    M1 = (1+0)(0+1) = 1
    M2 = (0+0)*0 = 0
    M3 = 1*(0-1) = -1
    M4 = 0*(0-0) = 0
    M5 = (1+0)*1 = 1
    M6 = (0-1)*(0+0) = 0
    M7 = (0-0)*(0+1) = 0
    
    C[0,0] = M1 + M4 - M5 + M7 = 1 + 0 - 1 + 0 = 0 ✓
    C[0,1] = M3 + M5 = -1 + 1 = 0 ✓
    C[1,0] = M2 + M4 = 0 + 0 = 0 ✓
    C[1,1] = M1 - M2 + M3 + M6 = 1 - 0 + (-1) + 0 = 0 ✓
    
    Great, it works!
    
    Now let me also verify with the tensor contraction formula:
    C[i,l] = sum_r W[r][l*n+i] * alpha_r * beta_r
    
    alpha_r = sum_a U[r][a] * A_flat[a]
    beta_r = sum_b V[r][b] * B_flat[b]
    
    For C[0,0] (l=0, i=0, W_idx=0):
    = W[0][0]*M1 + W[1][0]*M2 + W[2][0]*M3 + W[3][0]*M4 + W[4][0]*M5 + W[5][0]*M6 + W[6][0]*M7
    = 1*1 + 0*0 + 0*(-1) + 1*0 + (-1)*1 + 0*0 + 1*0
    = 1 + 0 + 0 + 0 - 1 + 0 + 0 = 0 ✓
    
    For C[0,1] (l=1, i=0, W_idx=2):
    = W[0][2]*M1 + W[1][2]*M2 + W[2][2]*M3 + W[3][2]*M4 + W[4][2]*M5 + W[5][2]*M6 + W[6][2]*M7
    = 0*1 + 0*0 + 1*(-1) + 0*0 + 1*1 + 0*0 + 0*0
    = 0 + 0 - 1 + 0 + 1 + 0 + 0 = 0 ✓
    
    Perfect. Now let me code this up.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7
    
    # U: shape (n*m, rank) = (4, 7), U[i*m+j, c]
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # i=0,j=0 (a)
        [0,  0,  0,  0,  1,  0,  1],  # i=0,j=1 (b)
        [0,  1,  0,  0,  0,  1,  0],  # i=1,j=0 (c)
        [1,  1,  0,  1,  0,  0, -1],  # i=1,j=1 (d)
    ]
    
    # V: shape (m*p, rank) = (4, 7), V[j*p+l, c]
    V = [
        [1,  1,  0, -1, 0,  1,  0],  # j=0,l=0 (e)
        [0,  0,  1,  0,  0,  1,  0],  # j=0,l=1 (f)
        [0,  0,  0,  1,  0,  0,  1],  # j=1,l=0 (g)
        [1,  0, -1, 0,  1,  0,  1],  # j=1,l=1 (h)
    ]
    
    # W: shape (p*n, rank) = (4, 7), W[k*n+i, c] where k=l (col of C), i (row of C)
    # W_idx = l*n + i
    # C[0,0]: l=0,i=0 -> 0
    # C[1,0]: l=0,i=1 -> 1
    # C[0,1]: l=1,i=0 -> 2
    # C[1,1]: l=1,i=1 -> 3
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # l=0,i=0 -> C[0,0]
        [0,  1,  0,  1,  0,  0,  0],  # l=0,i=1 -> C[1,0]
        [0,  0,  1,  0,  1,  0,  0],  # l=1,i=0 -> C[0,1]
        [1, -1,  1,  0,  0,  1,  0],  # l=1,i=1 -> C[1,1]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
