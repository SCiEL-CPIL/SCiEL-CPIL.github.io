"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    all_U = []
    all_V = []
    all_W = []
    
    def add_term(u_dict, v_dict, w_dict):
        all_U.append(u_dict)
        all_V.append(v_dict)
        all_W.append(w_dict)
    
    # Strassen 2x2: verified correct
    # C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    strassen_terms = [
        # M1=(A00+A11)(B00+B11)
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # M2=(A10+A11)B00
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        # M3=A00(B10-B11)
        ({(0,0):1}, {(1,0):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        # M4=A11(B01-B10)
        ({(1,1):1}, {(0,1):1,(1,0):-1}, {(0,0):1,(1,0):1}),
        # M5=(A00+A10)B11
        ({(0,0):1,(1,0):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        # M6=(A11-A10)(B00+B10)
        ({(1,1):1,(1,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1}),
        # M7=(A01-A11)(B01+B11)
        ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1}),
    ]
    
    # Schoolbook rect
    def schoolbook(n_r, m_mid, p_c):
        terms = []
        for i in range(n_r):
            for j in range(m_mid):
                for k in range(p_c):
                    terms.append(({(i,j):1}, {(j,k):1}, {(i,k):1}))
        return terms
    
    # Subproblem 1: C11 += A11*B11 (2x2 by 2x2), rank 7
    for u,v,w in strassen_terms:
        u_g = {i*5+j: val for (i,j),val in u.items()}
        v_g = {j*5+k: val for (j,k),val in v.items()}
        w_g = {k*5+i: val for (i,k),val in w.items()}
        add_term(u_g, v_g, w_g)
    
    # Subproblem 2: C11 += A12*B21 (2x3 by 3x2), rank 12
    for u,v,w in schoolbook(2,3,2):
        u_g = {i*5+(2+j): val for (i,j),val in u.items()}
        v_g = {(2+j)*5+k: val for (j,k),val in v.items()}
        w_g = {k*5+i: val for (i,k),val in w.items()}
        add_term(u_g, v_g, w_g)
    
    # Subproblem 3: C12 += A11*B12 (2x2 by 2x3), rank 12
    for u,v,w in schoolbook(2,2,3):
        u_g = {i*5+j: val for (i,j),val in u.items()}
        v_g = {j*5+(2+k): val for (j,k),val in v.items()}
        w_g = {(2+k)*5+i: val for (i,k),val in w.items()}
        add_term(u_g, v_g, w_g)
    
    # Subproblem 4: C12 += A12*B22 (2x3 by 3x3), rank 18
    for u,v,w in schoolbook(2,3,3):
        u_g = {i*5+(2+j): val for (i,j),val in u.items()}
        v_g = {(2+j)*5+(2+k): val for (j,k),val in v.items()}
        w_g = {(2+k)*5+i: val for (i,k),val in w.items()}
        add_term(u_g, v_g, w_g)
    
    # Subproblem 5: C21 += A21*B11 (3x2 by 2x2), rank 12
    for u,v,w in schoolbook(3,2,2):
        u_g = {(2+i)*5+j: val for (i,j),val in u.items()}
        v_g = {j*5+k: val for (j,k),val in v.items()}
        w_g = {k*5+(2+i): val for (i,k),val in w.items()}
        add_term(u_g, v_g, w_g)
    
    # Subproblem 6: C21 += A22*B21 (3x3 by 3x2), rank 18
    for u,v,w in schoolbook(3,3,2):
        u_g = {(2+i)*5+(2+j): val for (i,j),val in u.items()}
        v_g = {(2+j)*5+k: val for (j,k),val in v.items()}
        w_g = {k*5+(2+i): val for (i,k),val in w.items()}
        add_term(u_g, v_g, w_g)
    
    # Subproblem 7: C22 += A21*B12 (3x2 by 2x3), rank 18
    for u,v,w in schoolbook(3,2,3):
        u_g = {(2+i)*5+j: val for (i,j),val in u.items()}
        v_g = {j*5+(2+k): val for (j,k),val in v.items()}
        w_g = {(2+k)*5+(2+i): val for (i,k),val in w.items()}
        add_term(u_g, v_g, w_g)
    
    # Subproblem 8: C22 += A22*B22 (3x3 by 3x3), rank 27 (schoolbook)
    for u,v,w in schoolbook(3,3,3):
        u_g = {(2+i)*5+(2+j): val for (i,j),val in u.items()}
        v_g = {(2+j)*5+(2+k): val for (j,k),val in v.items()}
        w_g = {(2+k)*5+(2+i): val for (i,k),val in w.items()}
        add_term(u_g, v_g, w_g)
    
    rank = len(all_U)
    
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for c in range(rank):
        for row, val in all_U[c].items():
            U[row][c] = val
        for row, val in all_V[c].items():
            V[row][c] = val
        for row, val in all_W[c].items():
            W[row][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
