def construct():
    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a Strassen 2x2x2 block
    # u_indices: list of (i, j) for A
    # v_indices: list of (j, k) for B
    # w_indices: list of (i, k) for C
    # The Strassen steps define linear combinations.
    # We will just hardcode the 7 steps for 2x2x2.
    
    def add_strassen_2x2x2(base_u_i, base_u_j, base_v_j, base_v_k, base_w_i, base_w_k, current_col):
        # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # U indices:
        # A00: (base_u_i+0, base_u_j+0)
        # A01: (base_u_i+0, base_u_j+1)
        # A10: (base_u_i+1, base_u_j+0)
        # A11: (base_u_i+1, base_u_j+1)
        
        # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
        # V indices:
        # B00: (base_v_j+0, base_v_k+0)
        # B01: (base_v_j+0, base_v_k+1)
        # B10: (base_v_j+1, base_v_k+0)
        # B11: (base_v_j+1, base_v_k+1)
        
        # C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1]
        # W indices:
        # C00: (base_w_i+0, base_w_k+0)
        # C01: (base_w_i+0, base_w_k+1)
        # C10: (base_w_i+1, base_w_k+0)
        # C11: (base_w_i+1, base_w_k+1)
        
        # Note: W index in global is k*n + i.
        # So W[k, i] corresponds to row k*n + i.
        
        def get_u(i, j):
            return (base_u_i + i) * m + (base_u_j + j)
            
        def get_v(j, k):
            return (base_v_j + j) * p + (base_v_k + k)
            
        def get_w(i, k):
            return (base_w_k + k) * n + (base_w_i + i)
            
        # Strassen M1..M7
        # M1 = (A00 + A11)(B00 + B11)
        U[get_u(0,0)][current_col] += 1
        U[get_u(1,1)][current_col] += 1
        V[get_v(0,0)][current_col] += 1
        V[get_v(1,1)][current_col] += 1
        W[get_w(0,0)][current_col] += 1
        W[get_w(1,1)][current_col] += 1
        current_col += 1
        
        # M2 = (A10 + A11)B00
        U[get_u(1,0)][current_col] += 1
        U[get_u(1,1)][current_col] += 1
        V[get_v(0,0)][current_col] += 1
        W[get_w(1,0)][current_col] += 1
        W[get_w(1,1)][current_col] -= 1 # Wait, Strassen formulas vary.
        
        # Let's use the standard one:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # I already added M1. Let's reset and do it cleanly in the loop.
        return current_col

    # It's easier to just write the columns explicitly for each block.
    
    # We will create a list of tuples (u_indices, v_indices, w_indices, coeffs)
    # Actually, since each column is one multiplication, we just need to know which U, V, W entries are 1 (or -1).
    
    # Let's refactor to just build the 54 columns.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(3): # 3 blocks of columns
        k_base = b * 2
        
        # Part A: j in {0, 1}, k in {k_base, k_base+1}
        # 2x2x2 Strassen
        # U indices: i in {0,1}, j in {0,1} -> i*5 + j
        # V indices: j in {0,1}, k in {k_base, k_base+1} -> j*6 + k
        # W indices: i in {0,1}, k in {k_base, k_base+1} -> k*2 + i
        
        # M1
        U[0*5+0][col] = 1; U[1*5+1][col] = 1
        V[0*6+k_base][col] = 1; V[1*6+k_base+1][col] = 1
        W[k_base*2+0][col] = 1; W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # M2
        U[1*5+0][col] = 1; U[1*5+1][col] = 1
        V[0*6+k_base][col] = 1
        W[k_base*2+1][col] = 1; W[(k_base+1)*2+1][col] = -1
        col += 1
        
        # M3
        U[0*5+0][col] = 1
        V[1*6+k_base][col] = 1; V[1*6+k_base+1][col] = -1
        W[k_base*2+0][col] = 1; W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M4
        U[1*5+1][col] = 1
        V[0*6+k_base+1][col] = 1; V[0*6+k_base][col] = -1
        W[k_base*2+0][col] = 1; W[k_base*2+1][col] = 1
        col += 1
        
        # M5
        U[0*5+0][col] = 1; U[0*5+1][col] = 1
        V[1*6+k_base+1][col] = 1
        W[k_base*2+0][col] = -1; W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M6
        U[1*5+0][col] = 1; U[0*5+0][col] = -1
        V[0*6+k_base][col] = 1; V[0*6+k_base+1][col] = 1
        W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # M7
        U[0*5+1][col] = 1; U[1*5+1][col] = -1
        V[1*6+k_base][col] = 1; V[1*6+k_base+1][col] = 1
        W[k_base*2+0][col] = 1
        col += 1
        
        # Part B1: j in {2}, k in {k_base, k_base+1}
        # 2x1x2 Rank 4
        # A: i in {0,1}, j=2
        # B: j=2, k in {k_base, k_base+1}
        # C: i in {0,1}, k in {k_base, k_base+1}
        
        # M0: A0,2 B2,k0
        U[0*5+2][col] = 1
        V[2*6+k_base][col] = 1
        W[k_base*2+0][col] = 1
        col += 1
        
        # M1: A0,2 B2,k1
        U[0*5+2][col] = 1
        V[2*6+k_base+1][col] = 1
        W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M2: A1,2 B2,k0
        U[1*5+2][col] = 1
        V[2*6+k_base][col] = 1
        W[k_base*2+1][col] = 1
        col += 1
        
        # M3: A1,2 B2,k1
        U[1*5+2][col] = 1
        V[2*6+k_base+1][col] = 1
        W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # Part B2: j in {3, 4}, k in {k_base, k_base+1}
        # 2x2x2 Strassen
        # U indices: i in {0,1}, j in {3,4} -> i*5 + j
        # V indices: j in {3,4}, k in {k_base, k_base+1} -> j*6 + k
        # W indices: i in {0,1}, k in {k_base, k_base+1} -> k*2 + i
        
        # M1
        U[0*5+3][col] = 1; U[1*5+4][col] = 1
        V[3*6+k_base][col] = 1; V[4*6+k_base+1][col] = 1
        W[k_base*2+0][col] = 1; W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # M2
        U[1*5+3][col] = 1; U[1*5+4][col] = 1
        V[3*6+k_base][col] = 1
        W[k_base*2+1][col] = 1; W[(k_base+1)*2+1][col] = -1
        col += 1
        
        # M3
        U[0*5+3][col] = 1
        V[4*6+k_base][col] = 1; V[4*6+k_base+1][col] = -1
        W[k_base*2+0][col] = 1; W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M4
        U[1*5+4][col] = 1
        V[3*6+k_base+1][col] = 1; V[3*6+k_base][col] = -1
        W[k_base*2+0][col] = 1; W[k_base*2+1][col] = 1
        col += 1
        
        # M5
        U[0*5+3][col] = 1; U[0*5+4][col] = 1
        V[4*6+k_base+1][col] = 1
        W[k_base*2+0][col] = -1; W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M6
        U[1*5+3][col] = 1; U[0*5+3][col] = -1
        V[3*6+k_base][col] = 1; V[3*6+k_base+1][col] = 1
        W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # M7
        U[0*5+4][col] = 1; U[1*5+4][col] = -1
        V[4*6+k_base][col] = 1; V[4*6+k_base+1][col] = 1
        W[k_base*2+0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
