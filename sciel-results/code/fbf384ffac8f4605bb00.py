def construct():
    n, m, p = 2, 4, 6
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0
    
    # Block 1: C_L (cols 0-2 of C)
    # C_L = A1 B1L + A2 B2L
    # A1 is A[:, 0:2], A2 is A[:, 2:4]
    # B1L is B[0:2, 0:3], B2L is B[2:4, 0:3]
    
    # Sub-block 1.1: A1 B1L (2x2 x 2x3)
    # A1 indices: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B1L indices: (0,0)->0, (0,1)->1, (0,2)->2, (1,0)->6, (1,1)->7, (1,2)->8
    # CL indices: (0,0)->0, (0,1)->2, (0,2)->4, (1,0)->1, (1,1)->3, (1,2)->5
    # Wait, W index is k*n + i.
    # C_L is 2x3.
    # C_L[0,0] is global C[0,0] -> W[0*2+0]=0
    # C_L[0,1] is global C[1,0] -> W[1*2+0]=2
    # C_L[0,2] is global C[2,0] -> W[2*2+0]=4
    # C_L[1,0] is global C[0,1] -> W[0*2+1]=1
    # C_L[1,1] is global C[1,1] -> W[1*2+1]=3
    # C_L[1,2] is global C[2,1] -> W[2*2+1]=5

    # Strassen for A1 B1L[:, 0:2] (2x2 x 2x2)
    # A1: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B1L_left: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # m1 = (a+d)(e+h)
    add_term(0, 1, 0, 1, 0, 1, col); add_term(0, 1, 1, 1, 0, 1, col) # a*e, a*f ? No, Strassen is specific.
    # Let's just hardcode the Strassen coefficients for the 7 terms.
    
    # Term 1: (a+d)(e+h)
    # U: a(0,0)=1, d(1,1)=1 -> U[0]=1, U[5]=1
    # V: e(0,0)=1, h(1,1)=1 -> V[0]=1, V[7]=1
    # W: C00 += 1, C11 += 1 -> W[0]=1, W[3]=1
    add_term(0, 1, 0, 1, 0, 1, col); add_term(0, 1, 7, 1, 0, 1, col)
    add_term(5, 1, 0, 1, 0, 1, col); add_term(5, 1, 7, 1, 0, 1, col)
    # Wait, add_term adds to a single column. I need to add 4 entries to U/V/W for this term?
    # No, U, V, W are vectors.
    # U for term 1: U[0]=1, U[5]=1. Others 0.
    # V for term 1: V[0]=1, V[7]=1. Others 0.
    # W for term 1: W[0]=1, W[3]=1. Others 0.
    
    # My add_term function is for single non-zero entries.
    # I need a way to add a full vector.
    
    pass
