"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x5 x 5x6 matrix multiplication tensor using Strassen for 2x2x2 blocks.
    Rank = 6*7 + 3*4 = 54.
    """
    n, m, p = DIMENSIONS
    
    def strassen_decompose():
        """
        Return the 7 Strassen terms for 2x2x2.
        Each term is (U_contribs, V_contribs, W_contribs) where each is a list of (local_idx, coeff).
        Local U idx: (a, b) -> a*2+b for the 2x2 A sub-block
        Local V idx: (b, c) -> b*2+c for the 2x2 B sub-block  
        Local W idx: (c, a) -> c*2+a for the 2x2 C sub-block (W[k*n+i] convention)
        """
        # Verify Strassen formulas numerically
        import random
        for _ in range(100):
            A = [[random.randint(-5,5) for _ in range(2)] for _ in range(2)]
            B = [[random.randint(-5,5) for _ in range(2)] for _ in range(2)]
            C_true = [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
                      [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]
            
            # Strassen multipliers
            P1 = (A[0][0]+A[1][1])*(B[0][0]+B[1][1])
            P2 = (A[1][0]+A[1][1])*B[0][0]
            P3 = A[0][0]*(B[1][0]-B[1][1])
            P4 = A[1][1]*(B[0][1]-B[0][0])
            P5 = (A[0][0]+A[0][1])*B[1][1]
            P6 = (A[1][0]-A[0][0])*(B[0][0]+B[0][1])
            P7 = (A[0][1]-A[1][1])*(B[1][0]+B[1][1])
            
            C11 = P1 + P4 - P5 + P7
            C12 = P3 + P5
            C21 = P2 + P4
            C22 = P1 - P2 + P3 + P6
            
            if C11 != C_true[0][0] or C12 != C_true[0][1] or C21 != C_true[1][0] or C22 != C_true[1][1]:
                # Try alternate formula
                C11b = P1 + P4 - P5 + P7
                C12b = P3 + P5
                C21b = P2 + P4
                C22b = P1 - P2 + P3 + P6
                raise ValueError(f"Strassen failed: {A}, {B}, got [{C11},{C12},{C21},{C22}], want {C_true}")
        
        # The formula is verified. Now express each Pk as a rank-1 tensor.
        # P1 = (A00+A11)(B00+B11): U has A00,A11; V has B00,B11; W has C00,C11
        # P2 = (A10+A11)B00: U has A10,A11; V has B00; W has C10
        # P3 = A00(B10-B11): U has A00; V has B10,-B11; W has C01
        # P4 = A11(B01-B00): U has A11; V has B01,-B00; W has C10
        # P5 = (A00+A01)B11: U has A00,A01; V has B11; W has C00,C01
        # P6 = (A10-A00)(B00+B01): U has A10,-A00; V has B00,B01; W has C11
        # P7 = (A01-A11)(B10+B11): U has A01,-A11; V has B10,B11; W has C00,C11
        
        # W convention: W[k*n+i], so C[i][k] is at W[k*2+i]
        # C00 -> W[0*2+0] = W[0]
        # C01 -> W[1*2+0] = W[2]
        # C10 -> W[0*2+1] = W[1]
        # C11 -> W[1*2+1] = W[3]
        
        terms = [
            # P1: contributes to C00 (+1) and C11 (+1)
            ([(0,1),(1,1)], [(0,0),(1,1)], [(0,1),(3,1)]),
            # P2: contributes to C10 (+1)
            ([(1,0),(1,1)], [(0,0)], [(1,1)]),
            # P3: contributes to C01 (+1)
            ([(0,0)], [(1,0),(1,1)], [(2,1)]),  # Wait, B10-B11 means V[1,0]=1, V[1,1]=-1
            # Let me redo with explicit coefficients
        ]
        
        # Redo with explicit (index, coeff) pairs
        # U local: (a,b) where a=i, b=j
        # V local: (b,c) where b=j, c=k
        # W local: (c,a) where c=k, a=i -> index = c*2+a
        
        terms = [
            # P1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(3,1)]),
            # P2 = (A10+A11)B00 -> C10 += 1
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,1)]),
            # P3 = A00(B10-B11) -> C01 += 1
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(2,1)]),
            # P4 = A11(B01-B00) -> C10 += 1
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(1,1)]),
            # P5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,-1),(2,1)]),
            # P6 = (A10-A00)(B00+B01) -> C11 += 1
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(3,1)]),
            # P7 = (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1),(3,-1)]),
        ]
        
        return terms
    
    strassen_terms = strassen_decompose()
    
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4, 5)]
    
    rank = 0
    for jg in j_groups:
        for kg in k_groups:
            if len(jg) == 2 and len(kg) == 2:
                rank += 7
            else:
                rank += n * len(jg) * len(kg)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for jg in j_groups:
        for kg in k_groups:
            if len(jg) == 2 and len(kg) == 2:
                j0, j1 = jg
                k0, k1 = kg
                for (u_list, v_list, w_list) in strassen_terms:
                    for (a, b, coeff) in u_list:
                        U[(j0 + b) + (a) * m][col] += coeff  # U[i*m + j]
                    for (b, c, coeff) in v_list:
                        V[(j0 + b) * p + (k0 + c)][col] += coeff  # V[j*p + k]
                    for (c, a, coeff) in w_list:
                        W[(k0 + c) * n + a][col] += coeff  # W[k*n + i]
                    col += 1
            else:
                j_list = list(jg)
                k_list = list(kg)
                for i in range(n):
                    for j in j_list:
                        for k in k_list:
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
