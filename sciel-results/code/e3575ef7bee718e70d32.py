"""
Rank-28 exact tensor decomposition for (2,2,8) matrix multiplication.

The tensor is M_{2,2,8} with the official trace convention:
  T[a,b,c] = 1  iff  a = i*m + j,  b = j*p + k,  c = k*n + i
  for i in {0,1}, j in {0,1}, k in {0,...,7}.

The construction uses 4 independent blocks of Strassen's 7-term algorithm
for 2x2x2, giving rank 4*7 = 28.
"""

def construct():
    """Return the rank-28 decomposition as {'U': U, 'V': V, 'W': W}."""
    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n  # 4, 16, 16
    rank = 28

    # Strassen 7-term decomposition for 2x2x2.
    # Each term is (u, v, w) where u, v, w are length-4 vectors
    # indexed by (row, col) flattened: index = row*2 + col.
    # For M_{2,2,2}: A is 2x2, B is 2x2, C is 2x2.
    # T[a,b,c] = 1 iff a=i*2+j, b=j*2+k, c=k*2+i for i,j,k in {0,1}.
    #
    # Strassen's 7 products for C = A*B:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # u[i*2+j] = coefficient of A[i,j] in term
    # v[j*2+k] = coefficient of B[j,k] in term
    # w[i*2+k] = coefficient of C[i,k] in term  (note: C index is k*2+i in trace convention)
    #
    # Wait, the trace convention says c = k*n + i, so C index = k*2 + i.
    # So w is indexed by (k, i) not (i, k).
    # w[k*2 + i] = coefficient of C[i,k] in term.

    terms = [
        # M1 = (A00+A11)*(B00+B11)
        # u: A00=1, A11=1 -> [1,0,0,1]
        # v: B00=1, B11=1 -> [1,0,0,1]
        # C00 += 1, C11 += 1 -> w[0*2+0]=1, w[1*2+1]=1 -> w[0]=1, w[3]=1
        # w indexed by k*2+i: w[0]=C00 coeff, w[1]=C10 coeff, w[2]=C01 coeff, w[3]=C11 coeff
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),

        # M2 = (A10+A11)*B00
        # u: A10=1, A11=1 -> [0,0,1,1]
        # v: B00=1 -> [1,0,0,0]
        # C10 += 1, C11 -= 1 -> w[1*2+0]=1, w[1*2+1]=-1 -> w[2]=1, w[3]=-1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),

        # M3 = A00*(B01-B11)
        # u: A00=1 -> [1,0,0,0]
        # v: B01=1, B11=-1 -> [0,1,0,-1]
        # C01 += 1, C11 += 1 -> w[0*2+1]=1, w[1*2+1]=1 -> w[1]=1, w[3]=1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),

        # M4 = A11*(B10-B00)
        # u: A11=1 -> [0,0,0,1]
        # v: B10=1, B00=-1 -> [-1,0,1,0]
        # C00 += 1, C10 += 1 -> w[0*2+0]=1, w[1*2+0]=1 -> w[0]=1, w[2]=1
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),

        # M5 = (A00+A01)*B11
        # u: A00=1, A01=1 -> [1,1,0,0]
        # v: B11=1 -> [0,0,0,1]
        # C00 -= 1, C01 += 1 -> w[0*2+0]=-1, w[0*2+1]=1 -> w[0]=-1, w[1]=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),

        # M6 = (A10-A00)*(B00+B01)
        # u: A10=1, A00=-1 -> [-1,0,1,0]
        # v: B00=1, B01=1 -> [1,1,0,0]
        # C11 += 1 -> w[1*2+1]=1 -> w[3]=1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),

        # M7 = (A01-A11)*(B10+B11)
        # u: A01=1, A11=-1 -> [0,1,0,-1]
        # v: B10=1, B11=1 -> [0,0,1,1]
        # C00 += 1 -> w[0*2+0]=1 -> w[0]=1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Build U (4 x 28), V (16 x 28), W (16 x 28)
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    for block in range(4):
        for t in range(7):
            col = block * 7 + t
            u, v, w = terms[t]

            # U: index a = i*2 + j, i,j in {0,1}
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = u[i * 2 + j]

            # V: index b = j*p + k = j*8 + k, j in {0,1}, k in {2*block, 2*block+1}
            for j in range(2):
                for kl in range(2):
                    k_global = 2 * block + kl
                    b_idx = j * p + k_global
                    v_idx = j * 2 + kl
                    V[b_idx][col] = v[v_idx]

            # W: index c = k*n + i = k*2 + i, k in {2*block, 2*block+1}, i in {0,1}
            for i in range(2):
                for kl in range(2):
                    k_global = 2 * block + kl
                    c_idx = k_global * n + i
                    w_idx = kl * 2 + i
                    W[c_idx][col] = w[w_idx]

    return {"U": U, "V": V, "W": W}
