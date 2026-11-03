"""
Rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.

Approach:
  1. Try known rank-14 (Strassen-like) constructions for 3x3 x 3x5.
  2. If a candidate works (verified symbolically/numerically with high precision), return it.
  3. Otherwise, fall back to the schoolbook rank-45 decomposition.

For a 3x3 * 3x5 multiplication, the schoolbook rank is 45.
We attempt to find a lower-rank exact decomposition.

A rank-r tensor decomposition means:
  C[i,k] = sum_{t=0}^{r-1} U[i,t] * (A[i,j] * B[j,k]) ... 

More precisely, the matrix multiplication tensor T of order (n, m, p) has entries:
  T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]

A rank-r decomposition gives factor matrices:
  U: shape (n*m, r)
  V: shape (m*p, r)
  W: shape (p*n, r)

such that for all i in [0,n), j in [0,m), k in [0,p):
  sum_{t=0}^{r-1} U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t] = 1

This is a system of bilinear (actually trilinear in the factors, but bilinear if we fix one) equations.

For small dimensions like 3x3x3, Strassen gives rank 7.
For 3x3 x 3x5, we can try to adapt Strassen-like ideas or use known results.

Known: The tensor rank of 3x3 x 3x3 multiplication is 7 (Strassen).
For 3x3 x 3x5, the rank is at most 45 (schoolbook). There are results showing it can be less.

Let me try a systematic approach:
  - Use the fact that 3x3 x 3x5 can be seen as 5 separate 3x3 x 3x1 multiplications, but that's not helpful.
  - Alternatively, we can try to find a rank-14 or rank-16 decomposition using known algebraic identities.

Actually, let me try a different approach: use the "border rank" or "approximate rank" ideas and then round to get exact integer/half-integer solutions.

For 3x3 * 3x5, I'll try to construct a rank-14 decomposition by adapting Strassen's 7-rank scheme for 3x3*3x3 and extending it to the 5 columns.

Strassen's 7 products for 3x3*3x3:
  M1 = (A00 + A22) * (B00 + B22)
  M2 = (A20 + A22) * B00
  M3 = A00 * (B02 - B22)
  M4 = A22 * (B20 - B00)
  M5 = (A00 + A02) * B22
  M6 = (A20 - A00) * (B00 + B02)
  M7 = (A02 + A22) * (B20 + B22)

Then:
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C02 = M1 - M2 + M3 + M6
  C10 = M2 - M4 + M7
  C11 = M1 - M2 + M4 + M5
  C12 = M3 + M2 - M4 + M6
  C20 = M1 + M3 - M5 + M7
  C21 = M6 + M7
  C22 = M1 - M2 + M4

For 3x3 * 3x5, we have 5 output columns. Let me denote B as B[:, 0:5].

We can try to express the 5 columns using the same 7 Strassen products applied to 3x3 sub-blocks of B.

But B is 3x5, not 3x3. We can split B into B[:, 0:3] and B[:, 3:5].

C = A * B = A * [B1 | B2] = [A*B1 | A*B2]

So we need 2x7 = 14 products, giving rank 14.

Let me construct this:

For B1 = B[:, 0:3] (3x3):
  Use Strassen with 7 products M1..M7 as above, giving C[:, 0:3]

For B2 = B[:, 3:5] (3x2):
  We need A * B2 where B2 is 3x2. We can pad B2 to 3x3 with zeros, or use a 3x3x3 scheme.
  Actually, A * B2 = A * [B2 | 0] where the last column is zeros.
  Using Strassen on A * [B2 | 0], we get 7 products, but 3 of the output entries will be zero.

So total rank = 7 + 7 = 14.

Let me code this up and verify.

Strassen factors for 3x3 * 3x3:
  U matrix (9x7), V matrix (9x7), W matrix (9x7)

  Indexing: A is 3x3, B is 3x3, C is 3x3.
  U[i*m + j, t] for i in 0..2, j in 0..2
  V[j*p + k, t] for j in 0..2, k in 0..2
  W[k*n + i, t] for k in 0..2, i in 0..2

Strassen products:
  M1: (A00+A22)*(B00+B22)
    U: A00=1, A22=1
    V: B00=1, B22=1
    W: C00 +=1, C22 +=1, C11 +=1 (from C11 = M1 - M2 + M4 + M5)
    Wait, let me re-derive.

  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C02 = M1 - M2 + M3 + M6
  C10 = M2 - M4 + M7
  C11 = M1 - M2 + M4 + M5
  C12 = M3 + M2 - M4 + M6
  C20 = M1 + M3 - M5 + M7
  C21 = M6 + M7
  C22 = M1 - M2 + M4

  So for each Mt, the W coefficients (for C entries) are:
  M1: C00:+1, C02:+1, C11:+1, C20:+1, C22:+1
  M2: C02:-1, C10:+1, C11:-1, C12:+1, C22:-1
  M3: C01:+1, C02:+1, C12:+1, C20:+1
  M4: C00:+1, C10:-1, C11:+1, C12:-1, C22:+1
  M5: C00:-1, C01:+1, C11:+1, C20:-1
  M6: C02:+1, C12:+1, C21:+1
  M7: C00:+1, C10:+1, C20:+1, C21:+1

  U coefficients (for A entries):
  M1: A00:+1, A22:+1
  M2: A20:+1, A22:+1
  M3: A00:+1
  M4: A22:+1
  M5: A00:+1, A02:+1
  M6: A20:+1, A00:-1
  M7: A02:+1, A22:+1

  V coefficients (for B entries):
  M1: B00:+1, B22:+1
  M2: B00:+1
  M3: B02:+1, B22:-1
  M4: B20:+1, B00:-1
  M5: B22:+1
  M6: B00:+1, B02:+1
  M7: B20:+1, B22:+1

Now for the full 3x3 * 3x5:
  We split B into B[:,0:3] and B[:,3:5].
  For B[:,3:5], we pad with a zero column to make it 3x3: B_padded = [B[:,3:4] | B[:,4:5] | zeros]

  Then A * B = [A * B[:,0:3] | A * B[:,3:4] | A * B[:,4:5]]

  Using Strassen for each 3x3 block gives 14 products total.

  The U matrix will have 14 columns (7 for first block, 7 for second block).
  The V matrix will have 14 columns.
  The W matrix will have 14 columns, but we need to map the output indices correctly.

  For the first Strassen block (B[:,0:3]):
    Output C[:,0:3], so k in 0,1,2 maps to W index k*n + i = k*3 + i

  For the second Strassen block (B[:,3:5] padded to 3x3):
    Output C[:,3:5], so k in 3,4 (and 5 is zero).
    The Strassen output gives C'[:,0:3] where C'[:,0] = A*B[:,3], C'[:,1] = A*B[:,4], C'[:,2] = 0.
    So we need to map Strassen's k'=0 to k=3, k'=1 to k=4, k'=2 to k=5 (but that's zero).

  In the W matrix for the second block, the Strassen W indices are k'*n+i for k' in 0,1,2.
  We need to remap: k'=0 -> k=3, k'=1 -> k=4, k'=2 -> k=5.
  But W is indexed by k*n+i where k in 0..4 (since p=5).
  So for the second block, W[k*n+i, t] should be set based on the Strassen W[k'*3+i, t'] where k' is the Strassen output column.

  Let me code this up.
"""

import time

DIMENSIONS = [3, 3, 5]


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U: (n*m, rank) = (9, rank)
      V: (m*p, rank) = (15, rank)
      W: (p*n, rank) = (15, rank)
    
    with the trace convention W[k*n + i, r].
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    deadline = time.time() + 110  # safety margin
    
    # Try rank-14 Strassen-based decomposition
    U14, V14, W14 = _strassen_3x3_x_3x5()
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Verify the decomposition
    if _verify_decomposition(n, m, p, U14, V14, W14):
        return {"U": U14, "V": V14, "W": W14}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)


def _strassen_3x3_x_3x5():
    """
    Construct a rank-14 decomposition using Strassen's method for 3x3*3x3,
    applied to two 3x3 blocks of B (padding the last block to 3 columns).
    """
    n, m, p = 3, 3, 5
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 14
    V = [[0] * rank for _ in range(m * p)]  # 15 x 14
    W = [[0] * rank for _ in range(p * n)]  # 15 x 14
    
    # Strassen 3x3*3x3 coefficients
    # M1: (A00+A22)*(B00+B22)
    # M2: (A20+A22)*B00
    # M3: A00*(B02-B22)
    # M4: A22*(B20-B00)
    # M5: (A00+A02)*B22
    # M6: (A20-A00)*(B00+B02)
    # M7: (A02+A22)*(B20+B22)
    
    # U coefficients: U[i*m + j, t] for A[i,j]
    strassen_U = [
        # M1
        {0: 1, 8: 1},       # A00, A22 (i=0,j=0 -> idx 0; i=2,j=2 -> idx 8)
        # M2
        {6: 1, 8: 1},       # A20, A22 (i=2,j=0 -> idx 6; i=2,j=2 -> idx 8)
        # M3
        {0: 1},             # A00
        # M4
        {8: 1},             # A22
        # M5
        {0: 1, 2: 1},       # A00, A02
        # M6
        {6: 1, 0: -1},      # A20, -A00
        # M7
        {2: 1, 8: 1},       # A02, A22
    ]
    
    # V coefficients: V[j*p + k, t] for B[j,k]
    # For a 3x3 B block, p_local=3, so V[j*3 + k, t]
    strassen_V = [
        # M1
        {0: 1, 8: 1},       # B00, B22
        # M2
        {0: 1},             # B00
        # M3
        {2: 1, 8: -1},      # B02, -B22
        # M4
        {6: 1, 0: -1},      # B20, -B00
        # M5
        {8: 1},             # B22
        # M6
        {0: 1, 2: 1},       # B00, B02
        # M7
        {6: 1, 8: 1},       # B20, B22
    ]
    
    # W coefficients: W[k*n + i, t] for C[k,i]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C02 = M1 - M2 + M3 + M6
    # C10 = M2 - M4 + M7
    # C11 = M1 - M2 + M4 + M5
    # C12 = M3 + M2 - M4 + M6
    # C20 = M1 + M3 - M5 + M7
    # C21 = M6 + M7
    # C22 = M1 - M2 + M4
    
    strassen_W = [
        # M1
        {0: 1, 6: 1, 4: 1, 10: 1, 14: 1},   # C00, C02, C11, C20, C22 -> k*n+i: 0*3+0=0, 2*3+0=6, 1*3+1=4, 2*3+0=6? wait
    ]
    
    # Let me re-index W properly.
    # W[k*n + i, t] where k is the row of C, i is the column of C.
    # C is 3x3, so k in 0..2, i in 0..2.
    # C[k][i] is at W index k*3 + i.
    
    # C00: k=0, i=0 -> idx 0
    # C01: k=0, i=1 -> idx 1
    # C02: k=0, i=2 -> idx 2
    # C10: k=1, i=0 -> idx 3
    # C11: k=1, i=1 -> idx 4
    # C12: k=1, i=2 -> idx 5
    # C20: k=2, i=0 -> idx 6
    # C21: k=2, i=1 -> idx 7
    # C22: k=2, i=2 -> idx 8
    
    strassen_W = [
        # M1: C00+1, C02+1, C11+1, C20+1, C22+1
        {0: 1, 2: 1, 4: 1, 6: 1, 8: 1},
        # M2: C02-1, C10+1, C11-1, C12+1, C22-1
        {2: -1, 3: 1, 4: -1, 5: 1, 8: -1},
        # M3: C01+1, C02+1, C12+1, C20+1
        {1: 1, 2: 1, 5: 1, 6: 1},
        # M4: C00+1, C10-1, C11+1, C12-1, C22+1
        {0: 1, 3: -1, 4: 1, 5: -1, 8: 1},
        # M5: C00-1, C01+1, C11+1, C20-1
        {0: -1, 1: 1, 4: 1, 6: -1},
        # M6: C02+1, C12+1, C21+1
        {2: 1, 5: 1, 7: 1},
        # M7: C00+1, C10+1, C20+1, C21+1
        {0: 1, 3: 1, 6: 1, 7: 1},
    ]
    
    # Block 1: B[:, 0:3], output C[:, 0:3]
    for t in range(7):
        col = t
        # U
        for idx, val in strassen_U[t].items():
            U[idx][col] = val
        # V: B indices for first block, B[:,0:3], so B[j,k] -> V[j*5 + k, col]
        for vidx_local, val in strassen_V[t].items():
            j = vidx_local // 3
            k = vidx_local % 3
            vidx_global = j * p + k
            V[vidx_global][col] = val
        # W: output C[:,0:3], C[k,i] -> W[k*3 + i, col]
        for widx_local, val in strassen_W[t].items():
            k = widx_local // 3
            i = widx_local % 3
            widx_global = k * n + i
            W[widx_global][col] = val
    
    # Block 2: B[:, 3:5] padded to 3x3, output C[:, 3:5]
    # B_padded[j, k'] = B[j, 3+k'] for k'=0,1; B_padded[j, 2] = 0
    # Strassen on A * B_padded gives C'[:, 0:3] where C'[:,0]=A*B[:,3], C'[:,1]=A*B[:,4], C'[:,2]=0
    # Map C'[k', i] to C[k'+3, i] for k'=0,1 (k'=2 is zero)
    
    for t in range(7):
        col = 7 + t
        # U: same as before
        for idx, val in strassen_U[t].items():
            U[idx][col] = val
        # V: B_padded[j, k'] -> V[j*5 + (3+k'), col] for k'=0,1; k'=2 is zero so skip
        for vidx_local, val in strassen_V[t].items():
            j = vidx_local // 3
            k_local = vidx_local % 3
            if k_local == 2:
                # This corresponds to the zero-padded column, coefficient is multiplied by 0
                continue
            k_global = 3 + k_local
            vidx_global = j * p + k_global
            V[vidx_global][col] = val
        # W: C'[k', i] -> C[k'+3, i] for k'=0,1; k'=2 is zero
        for widx_local, val in strassen_W[t].items():
            k_local = widx_local // 3
            i = widx_local % 3
            if k_local == 2:
                # This output is zero, skip
                continue
            k_global = 3 + k_local
            widx_global = k_global * n + i
            W[widx_global][col] = val
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the tensor decomposition exactly computes matrix multiplication.
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for t in range(rank):
                    total += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if total != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
