DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    # A is n x m (2x3), B is m x p (3x7), C is n x p (2x7)
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (14, rank), W[k*n+i, r] for C[i,k]
    
    U_entries = []  # list of (row, value) per term
    V_entries = []
    W_entries = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a bilinear term. u_dict, v_dict, w_dict map index->coefficient."""
        U_entries.append(dict(u_dict))
        V_entries.append(dict(v_dict))
        W_entries.append(dict(w_dict))
    
    # A indices: A[i,j] -> U row i*m+j = i*3+j
    # B indices: B[j,k] -> V row j*p+k = j*7+k
    # C indices: C[i,k] -> W row k*n+i = k*2+i
    
    # A1 = A[0:2, 0:2] -> indices (0,0),(0,1),(1,0),(1,1)
    # a2 = A[0:2, 2] -> indices (0,2),(1,2)
    # B1 = B[0:2, 0:7] -> rows 0,1
    # b2 = B[2, 0:7] -> row 2
    
    A00 = 0*3+0  # 0
    A01 = 0*3+1  # 1
    A10 = 1*3+0  # 3
    A11 = 1*3+1  # 4
    A02 = 0*3+2  # 2
    A12 = 1*3+2  # 5
    
    # B1 blocks: cols 0-1, 2-3, 4-5, col 6
    # Block s: B[0,s: s+2], B[1, s:s+2]
    
    # Strassen for 2x2 x 2x2, for each of 3 full blocks (s=0,2,4)
    for s in [0, 2, 4]:
        B00 = 0*7+s    # B[0,s]
        B01 = 0*7+s+1  # B[0,s+1]
        B10 = 1*7+s    # B[1,s]
        B11 = 1*7+s+1  # B[1,s+1]
        
        # Output positions in C:
        # C[0,s], C[0,s+1], C[1,s], C[1,s+1]
        C00 = s*2+0    # C[0,s]
        C01 = (s+1)*2+0  # C[0,s+1]
        C10 = s*2+1    # C[1,s]
        C11 = (s+1)*2+1  # C[1,s+1]
        
        # M1 = (A00+A11)(B00+B11)
        add_term({A00:1, A11:1}, {B00:1, B11:1}, {C00:1, C11:1})
        
        # M2 = (A10+A11)B00
        add_term({A10:1, A11:1}, {B00:1}, {C10:1})
        
        # M3 = A00(B10-B11)
        add_term({A00:1}, {B10:1, B11:-1}, {C01:1})
        
        # M4 = A11(B01-B00)
        add_term({A11:1}, {B01:1, B00:-1}, {C00:-1, C10:1})
        
        # M5 = (A00+A01)B11
        add_term({A00:1, A01:1}, {B11:1}, {C00:1, C11:1})
        
        # M6 = (A10-A00)(B00+B01)
        add_term({A10:1, A00:-1}, {B00:1, B01:1}, {C11:1})
        
        # M7 = (A01-A11)(B10+B11)
        add_term({A01:1, A11:-1}, {B10:1, B11:1}, {C01:1, C11:1})
    
    # 2x2 x 2x1 block (col 6)
    # A1 (2x2) * v (2x1) where v = [B[0,6], B[1,6]]
    # Result: C[0,6] = A00*v0 + A01*v1, C[1,6] = A10*v0 + A11*v1
    # Rank 4: four scalar mults
    
    B0_6 = 0*7+6  # B[0,6]
    B1_6 = 1*7+6  # B[1,6]
    C0_6 = 6*2+0  # C[0,6]
    C1_6 = 6*2+1  # C[1,6]
    
    # C[0,6] += A00*B[0,6]
    add_term({A00:1}, {B0_6:1}, {C0_6:1})
    # C[0,6] += A01*B[1,6]
    add_term({A01:1}, {B1_6:1}, {C0_6:1})
    # C[1,6] += A10*B[0,6]
    add_term({A10:1}, {B0_6:1}, {C1_6:1})
    # C[1,6] += A11*B[1,6]
    add_term({A11:1}, {B1_6:1}, {C1_6:1})
    
    # Part 2: a2 * b2, outer product A[*,2] * B[2,*]
    # C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in 0..6
    # 14 terms
    for i in range(2):
        for k in range(7):
            Ai2 = i*3+2
            B2k = 2*7+k
            Cik = k*2+i
            add_term({Ai2:1}, {B2k:1}, {Cik:1})
    
    rank = len(U_entries)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for idx, val in U_entries[r].items():
            U[idx][r] = val
        for idx, val in V_entries[r].items():
            V[idx][r] = val
        for idx, val in W_entries[r].items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
