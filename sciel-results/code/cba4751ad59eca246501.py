"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U index: i*m + j (A[i,j])
    # V index: j*p + k (B[j,k])
    # W index: k*n + i (C[i,k])
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(U_coeff, V_coeff, W_coeff):
        U_terms.append(U_coeff)
        V_terms.append(V_coeff)
        W_terms.append(W_coeff)
    
    def schoolbook_block(oi, oj, ok, ni, nj, nk):
        """Schoolbook for block A[oi:oi+ni, oj:oj+nj] * B[oj:oj+nj, ok:ok+nk] -> C[oi:oi+ni, ok:ok+nk]"""
        for di in range(ni):
            for dj in range(nj):
                for dk in range(nk):
                    i = oi + di
                    j = oj + dj
                    k = ok + dk
                    add_term({i*m + j: 1}, {j*p + k: 1}, {k*n + i: 1})
    
    def strassen_2x2x2(oi, oj, ok):
        """Strassen 2x2x2 at offsets (oi,oj,ok). A block: rows oi,oi+1; cols oj,oj+1. B block: rows oj,oj+1; cols ok,ok+1."""
        # M1 = (a00+a11)(b00+b11)
        add_term({oi*m+oj:1, (oi+1)*m+(oj+1):1},
                 {oj*p+ok:1, (oj+1)*p+(ok+1):1},
                 {ok*n+oi:1, (ok+1)*n+(oi+1):1})
        # M2 = (a10+a11)b00
        add_term({(oi+1)*m+oj:1, (oi+1)*m+(oj+1):1},
                 {oj*p+ok:1},
                 {(ok+1)*n+(oi+1):1})
        # M3 = a00(b01-b11)
        add_term({oi*m+oj:1},
                 {oj*p+(ok+1):1, (oj+1)*p+(ok+1):-1},
                 {(ok+1)*n+oi:1})
        # M4 = a11(b10-b00)
        add_term({(oi+1)*m+(oj+1):1},
                 {(oj+1)*p+ok:1, oj*p+ok:-1},
                 {ok*n+(oi+1):1})
        # M5 = (a00+a01)b11
        add_term({oi*m+oj:1, oi*m+(oj+1):1},
                 {(oj+1)*p+(ok+1):1},
                 {ok*n+oi:-1, (ok+1)*n+oi:1})
        # M6 = (a10-a00)(b00+b01)
        add_term({(oi+1)*m+oj:1, oi*m+oj:-1},
                 {oj*p+ok:1, oj*p+(ok+1):1},
                 {(ok+1)*n+(oi+1):1})
        # M7 = (a01-a11)(b10+b11)
        add_term({oi*m+(oj+1):1, (oi+1)*m+(oj+1):-1},
                 {(oj+1)*p+ok:1, (oj+1)*p+(ok+1):1},
                 {ok*n+oi:1})
    
    def strassen_4x4x4(oi, oj, ok):
        """Strassen 4x4x4 using recursive 2x2x2 Strassen. Rank 49."""
        # Split 4x4 into 2x2 blocks of 2x2 matrices
        # Block offsets: (0,0), (0,2), (2,0), (2,2)
        # A blocks: A00=(oi,oj), A01=(oi,oj+2), A10=(oi+2,oj), A11=(oi+2,oj+2)
        # B blocks: B00=(oj,ok), B01=(oj,ok+2), B10=(oj+2,ok), B11=(oj+2,ok+2)
        # C blocks: C00=(oi,ok), C01=(oi,ok+2), C10=(oi+2,ok), C11=(oi+2,ok+2)
        
        # Each Strassen term is a 2x2x2 matrix multiplication at the block level.
        # We decompose each using strassen_2x2x2.
        
        # M1 = (A00+A11)(B00+B11)
        # This is a 2x2x2 multiplication where:
        # U entries: A00 at (0,0), A11 at (1,1) in the 2x2 U space
        # V entries: B00 at (0,0), B11 at (1,1)
        # W entries: C00 at (0,0), C11 at (1,1)
        # We need to map this to our indexing.
        # The 2x2x2 Strassen works on consecutive indices.
        # For M1: U has A[oi,oj], A[oi+1,oj+1], A[oi+2,oj+2], A[oi+3,oj+3]
        #         V has B[oj,ok], B[oj+1,ok+1], B[oj+2,ok+2], B[oj+3,ok+3]
        #         W has C[oi,ok], C[oi+1,ok+1], C[oi+2,ok+2], C[oi+3,ok+3]
        # We can use two 2x2x2 Strassen calls:
        #   One for top-left 2x2: strassen_2x2x2(oi, oj, ok)
        #   One for bottom-right 2x2: strassen_2x2x2(oi+2, oj+2, ok+2)
        # But that's not quite right because M1 = (A00+A11)(B00+B11) means
        # the U form is A00 + A11 (diagonal blocks), V form is B00 + B11.
        # The product (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # But Strassen's M1 is a single multiplication of (A00+A11) by (B00+B11).
        # (A00+A11) is a 4x4 matrix with A00 in top-left, A11 in bottom-right.
        # (B00+B11) is a 4x4 matrix with B00 in top-left, B11 in bottom-right.
        # Their product is a 4x4 matrix. This is NOT the same as two separate 2x2x2 multiplications.
        
        # Actually, I realize the issue. The Strassen decomposition for 4x4x4 with rank 49
        # works by viewing the 4x4 matrices as 2x2 block matrices and applying the 7-term
        # Strassen formula at the block level. Each of the 7 terms is a 2x2x2 matrix
        # multiplication (of 2x2 blocks). Each 2x2x2 multiplication has rank 7.
        # Total: 7*7 = 49.
        
        # To implement: for each of the 7 Strassen terms, I need to decompose a 2x2x2
        # matrix multiplication where each "scalar" is a 2x2 matrix.
        # A 2x2x2 matrix multiplication of 2x2 matrices can be decomposed into
        # 7 rank-1 terms (Strassen), where each rank-1 term involves 2x2 matrices.
        # Each 2x2 matrix rank-1 term expands to 4 scalar rank-1 terms.
        # So each Strassen term: 7 * 4 = 28 scalar rank-1 terms.
        # Total: 7 * 28 = 196. That's way more than 49.
        
        # I'm clearly confused. Let me just use the direct approach:
        # The 4x4x4 tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
        # contributes to (i,k) entry of C.
        # Strassen says this has rank 49.
        
        # Let me just use schoolbook for the 4x4x4 block (rank 64) for now and
        # focus on getting the overall structure right.
        schoolbook_block(oi, oj, ok, 4, 4, 4)
    
    # Block decomposition:
    # Split j into {0,1,2,3} and {4}, k into {0,1,2,3} and {4}
    # Block 1: A[:,0:4] (4x4) * B[0:4,0:4] (4x4) -> C[:,0:4] (4x4): rank 64 (schoolbook)
    schoolbook_block(0, 0, 0, 4, 4, 4)
    # Block 2: A[:,0:4] (4x4) * B[0:4,4] (4x1) -> C[:,4] (4x1): rank 16
    schoolbook_block(0, 0, 4, 4, 4, 1)
    # Block 3: A[:,4] (4x1) * B[4,0:4] (1x4) -> C[:,0:4] (4x4): rank 16
    schoolbook_block(0, 4, 0, 4, 1, 4)
    # Block 4: A[:,4] (4x1) * B[4,4] (1x1) -> C[:,4] (4x1): rank 4
    schoolbook_block(0, 4, 4, 4, 1, 1)
    
    # Convert terms to matrices
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        for idx, val in U_terms[r].items():
            U[idx][r] = val
        for idx, val in V_terms[r].items():
            V[idx][r] = val
        for idx, val in W_terms[r].items():
            W[idx][r] = val
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
