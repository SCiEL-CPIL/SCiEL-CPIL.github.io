"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx: i*m + j
    # v_idx: j*p + k
    # w_idx: k*n + i
    # coeff: value for this term
    def add_term(c, u_idx, v_idx, w_idx, coeff):
        U[u_idx][c] = coeff
        V[v_idx][c] = coeff
        W[w_idx][c] = coeff

    # Block 1: C[:, 0:2]
    # Part 1a: A[:, 0] * B[0, 0:2]
    # 2x1x2 schoolbook
    # i=0,1; j=0; k=0,1
    c = 0
    # i=0, j=0, k=0
    add_term(c, 0*m+0, 0*p+0, 0*n+0, 1); c+=1
    # i=0, j=0, k=1
    add_term(c, 0*m+0, 0*p+1, 1*n+0, 1); c+=1
    # i=1, j=0, k=0
    add_term(c, 1*m+0, 0*p+0, 0*n+1, 1); c+=1
    # i=1, j=0, k=1
    add_term(c, 1*m+0, 0*p+1, 1*n+1, 1); c+=1
    
    # Part 1b: A[:, 1:3] * B[1:3, 0:2]
    # 2x2x2 Strassen
    # A indices: i=0,1; j=1,2
    # B indices: j=1,2; k=0,1
    # C indices: i=0,1; k=0,1
    
    # Strassen M1: (A01 + A12) * (B20 + B21)
    # U: i=0,j=1 (1), i=1,j=2 (1)
    # V: j=2,k=0 (1), j=2,k=1 (1)
    # W: k=0,i=0 (1), k=1,i=1 (1)? No, W must match the output.
    # Strassen formula:
    # M1 = (a+d)(e+h)
    # a=A01, d=A12, e=B10, h=B21?
    # Let's map carefully.
    # A_sub = [[A01, A02], [A11, A12]]
    # B_sub = [[B10, B11], [B20, B21]]
    # C_sub = A_sub * B_sub
    
    # M1 = (A01 + A12) * (B10 + B21)
    # U: (0,1)=1, (1,2)=1
    # V: (1,0)=1, (2,1)=1
    # W: contributes to C00, C01, C10, C11
    # C00 += M1
    # C01 += M1
    # C10 += M1
    # C11 -= M1
    # So W: (0,0)=1, (1,0)=1, (0,1)=1, (1,1)=-1
    # Note: W index is k*n + i.
    # (k=0, i=0) -> 0
    # (k=1, i=0) -> 1
    # (k=0, i=1) -> 2
    # (k=1, i=1) -> 3
    
    # This manual construction is error-prone.
    # Better to define the Strassen terms generically and map indices.
    
    pass
# EVOLVE-BLOCK-END
