# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 matrix multiplication.
    These terms correspond to the Strassen algorithm, which can be viewed
    as exploiting the algebraic structure of the group ring (cyclic group Z_2 x Z_2).
    """
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00 +=, C11 +=
    terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
    # M2 = (A10+A11)B00 -> C10 +=
    terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}))
    # M3 = A00(B01-B11) -> C01 +=
    terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}))
    # M4 = A11(B10-B00) -> C10 +=
    terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(1,0):1}))
    # M5 = (A00+A01)B11 -> C01 +=
    terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,1):1}))
    # M6 = (A10-A00)(B00+B01) -> C11 +=
    terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
    # M7 = (A01-A11)(B10+B11) -> C00 +=
    terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
    return terms

def _build_4x4x4_strassen_terms():
    """
    Constructs the 49 rank-1 terms for 4x4x4 matrix multiplication by composing
    two levels of 2x2x2 Strassen decompositions. This leverages the recursive
    algebraic structure similar to FFT/Winograd schemes.
    """
    outer = _strassen_2x2_terms()
    inner = _strassen_2x2_terms()
    terms = []
    for oA, oB, oC in outer:
        for iA, iB, iC in inner:
            A = {}
            B = {}
            C = {}
            for (bI, bJ), oa in oA.items():
                for (li, lj), ia in iA.items():
                    gi, gj = bI*2+li, bJ*2+lj
                    A[(gi,gj)] = A.get((gi,gj),0) + oa*ia
            for (bJ, bK), ob in oB.items():
                for (lj, lk), ib in iB.items():
                    gj, gk = bJ*2+lj, bK*2+lk
                    B[(gj,gk)] = B.get((gj,gk),0) + ob*ib
            for (bK, bI), oc in oC.items():
                for (lk, li), ic in iC.items():
                    gk, gi = bK*2+lk, bI*2+li
                    C[(gk,gi)] = C.get((gk,gi),0) + oc*ic
            terms.append((A,B,C))
    return terms

def construct():
    n, m, p = 4, 4, 5
    # 1. Build the 4x4x4 part (k=0..3) using Strassen (Rank 49)
    terms_4x4 = _build_4x4x4_strassen_terms()
    R1 = len(terms_4x4)
    
    # 2. Build the 4x4x1 part (k=4) using Schoolbook (Rank 16)
    # C[:, 4] = A @ B[:, 4]
    # This is a direct sum, so we append 16 terms.
    R2 = 16
    rank = R1 + R2
    
    # Initialize matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Fill 4x4x4 part
    for c, (A, B, C) in enumerate(terms_4x4):
        for (i, j), val in A.items():
            U[i*m+j][c] += val
        for (j, k), val in B.items():
            V[j*p+k][c] += val
        for (k, i), val in C.items():
            W[k*n+i][c] += val
            
    # Fill 5th column part (k=4)
    t = 0
    for i in range(n):
        for j in range(m):
            col = R1 + t
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            t += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
