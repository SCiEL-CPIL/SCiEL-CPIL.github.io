"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-23 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy: Split A (2x3) = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Split B (3x5) = [B1; B2] where B1 is 2x5, B2 is 1x5.
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x5): split B1 = [B1a | B1b | B1c]
      B1a is 2x2, B1b is 2x2, B1c is 2x1
      A1*B1a: Strassen (rank 7)
      A1*B1b: Strassen (rank 7)  
      A1*B1c: schoolbook (rank 4)
    For A2*B2 (2x1 * 1x5): schoolbook (rank 5)
    Total: 7 + 7 + 4 + 5 = 23
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    U_list = []  # will hold rows for U (shape n*m x rank)
    V_list = []  # will hold rows for V (shape m*p x rank)
    W_list = []  # will hold rows for W (shape p*n x rank)
    
    rank = 0
    
    def add_term(u_vals, v_vals, w_vals):
        """Add one rank-1 term. u_vals is dict for U, v_vals for V, w_vals for W."""
        nonlocal rank
        u_row = [0] * 1  # placeholder, we'll build full matrix later
        # We'll collect terms as tuples
        U_list.append(u_vals)
        V_list.append(v_vals)
        W_list.append(w_vals)
        rank += 1
    
    # Helper to create sparse vector as dict
    def sparse(d, idx, val=1):
        d[idx] = val
        return d
    def sparse2(d, idx1, val1, idx2, val2):
        d[idx1] = val1
        d[idx2] = val2
        return d
    def sparse3(d, idx1, val1, idx2, val2, idx3, val3):
        d[idx1] = val1
        d[idx2] = val2
        d[idx3] = val3
        return d
    
    # Indices:
    # U: i*m + j, i in [0,1], j in [0,1,2] -> indices 0,1,2,3,4,5
    # V: j*p + k, j in [0,1,2], k in [0,1,2,3,4] -> indices 0..14
    # W: k*n + i, k in [0,1,2,3,4], i in [0,1] -> indices 0..9
    
    # Part 1: A1 * B1a (2x2 * 2x2) using Strassen
    # A1 = [[a00, a01], [a10, a11]] -> U indices: i*m+j, i=0,1; j=0,1
    #   A1[0,0] = U[0*3+0] = U[0]
    #   A1[0,1] = U[0*3+1] = U[1]
    #   A1[1,0] = U[1*3+0] = U[3]
    #   A1[1,1] = U[1*3+1] = U[4]
    # B1a = [[b00, b01], [b10, b11]] -> V indices: j*p+k, j=0,1; k=0,1
    #   B1a[0,0] = V[0*5+0] = V[0]
    #   B1a[0,1] = V[0*5+1] = V[1]
    #   B1a[1,0] = V[1*5+0] = V[5]
    #   B1a[1,1] = V[1*5+1] = V[6]
    # Result C1a = A1*B1a is 2x2, W indices: k*n+i, k=0,1; i=0,1
    #   C1a[0,0] = W[0*2+0] = W[0]
    #   C1a[0,1] = W[0*2+1] = W[1]
    #   C1a[1,0] = W[1*2+0] = W[2]
    #   C1a[1,1] = W[1*2+1] = W[3]
    
    # Strassen for 2x2 * 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # M1: (a00+a11)(b00+b11)
    u = sparse2({}, 0, 1, 4, 1)
    v = sparse2({}, 0, 1, 6, 1)
    w = sparse3({}, 0, 1, 3, -1, 2, 1)  # C00 += 1, C11 += -1, C10 += 1
    add_term(u, v, w)
    
    # M2: (a10+a11)b00
    u = sparse2({}, 3, 1, 4, 1)
    v = sparse({}, 0, 1)
    w = sparse({}, 2, 1)  # C10 += 1
    add_term(u, v, w)
    
    # M3: a00(b01-b11)
    u = sparse({}, 0, 1)
    v = sparse2({}, 1, 1, 6, -1)
    w = sparse3({}, 1, 1, 3, -1)  # C01 += 1, C11 += -1
    add_term(u, v, w)
    
    # M4: a11(b10-b00)
    u = sparse({}, 4, 1)
    v = sparse2({}, 5, 1, 0, -1)
    w = sparse2({}, 0, -1, 2, 1)  # C00 += -1, C10 += 1
    add_term(u, v, w)
    
    # M5: (a00+a01)b11
    u = sparse2({}, 0, 1, 1, 1)
    v = sparse({}, 6, 1)
    w = sparse({}, 1, 1)  # C01 += 1
    add_term(u, v, w)
    
    # M6: (a10-a00)(b00+b01)
    u = sparse2({}, 3, 1, 0, -1)
    v = sparse2({}, 0, 1, 1, 1)
    w = sparse({}, 0, 1)  # C00 += 1
    add_term(u, v, w)
    
    # M7: (a01-a11)(b10+b11)
    u = sparse2({}, 1, 1, 4, -1)
    v = sparse2({}, 5, 1, 6, 1)
    w = sparse({}, 3, 1)  # C11 += 1
    add_term(u, v, w)
    
    # Part 2: A1 * B1b (2x2 * 2x2) using Strassen
    # B1b is columns 2,3 of B1, so B1b[j,k] = B[j, 2+k] for j=0,1; k=0,1
    # V indices: j*p + (2+k), j=0,1; k=0,1
    #   B1b[0,0] = V[0*5+2] = V[2]
    #   B1b[0,1] = V[0*5+3] = V[3]
    #   B1b[1,0] = V[1*5+2] = V[7]
    #   B1b[1,1] = V[1*5+3] = V[8]
    # Result C1b is 2x2, contributes to columns 2,3 of C
    # W indices: (2+k)*n + i, k=0,1; i=0,1
    #   C1b[0,0] = W[2*2+0] = W[4]
    #   C1b[0,1] = W[2*2+1] = W[5]
    #   C1b[1,0] = W[3*2+0] = W[6]
    #   C1b[1,1] = W[3*2+1] = W[7]
    
    # M1: (a00+a11)(b00+b11) where b00=V[2], b11=V[8]
    u = sparse2({}, 0, 1, 4, 1)
    v = sparse2({}, 2, 1, 8, 1)
    w = sparse3({}, 4, 1, 7, -1, 6, 1)
    add_term(u, v, w)
    
    # M2: (a10+a11)b00
    u = sparse2({}, 3, 1, 4, 1)
    v = sparse({}, 2, 1)
    w = sparse({}, 6, 1)
    add_term(u, v, w)
    
    # M3: a00(b01-b11)
    u = sparse({}, 0, 1)
    v = sparse2({}, 3, 1, 8, -1)
    w = sparse2({}, 5, 1, 7, -1)
    add_term(u, v, w)
    
    # M4: a11(b10-b00)
    u = sparse({}, 4, 1)
    v = sparse2({}, 7, 1, 2, -1)
    w = sparse2({}, 4, -1, 6, 1)
    add_term(u, v, w)
    
    # M5: (a00+a01)b11
    u = sparse2({}, 0, 1, 1, 1)
    v = sparse({}, 8, 1)
    w = sparse({}, 5, 1)
    add_term(u, v, w)
    
    # M6: (a10-a00)(b00+b01)
    u = sparse2({}, 3, 1, 0, -1)
    v = sparse2({}, 2, 1, 3, 1)
    w = sparse({}, 4, 1)
    add_term(u, v, w)
    
    # M7: (a01-a11)(b10+b11)
    u = sparse2({}, 1, 1, 4, -1)
    v = sparse2({}, 7, 1, 8, 1)
    w = sparse({}, 7, 1)
    add_term(u, v, w)
    
    # Part 3: A1 * B1c (2x2 * 2x1) schoolbook, rank 4
    # B1c is column 4 of B1, so B1c[j,0] = B[j, 4] for j=0,1
    # V indices: j*p + 4, j=0,1
    #   B1c[0,0] = V[0*5+4] = V[4]
    #   B1c[1,0] = V[1*5+4] = V[9]
    # Result C1c is 2x1, contributes to column 4 of C
    # W indices: 4*n + i, i=0,1
    #   C1c[0,0] = W[4*2+0] = W[8]
    #   C1c[1,0] = W[4*2+1] = W[9]
    
    # Schoolbook for 2x2 * 2x1: C[i,0] = sum_j A[i,j] * B[j,0]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    # Term 1: A[0,0] * B[0,0]
    u = sparse({}, 0, 1)
    v = sparse({}, 4, 1)
    w = sparse({}, 8, 1)
    add_term(u, v, w)
    
    # Term 2: A[0,1] * B[1,0]
    u = sparse({}, 1, 1)
    v = sparse({}, 9, 1)
    w = sparse({}, 8, 1)
    add_term(u, v, w)
    
    # Term 3: A[1,0] * B[0,0]
    u = sparse({}, 3, 1)
    v = sparse({}, 4, 1)
    w = sparse({}, 9, 1)
    add_term(u, v, w)
    
    # Term 4: A[1,1] * B[1,0]
    u = sparse({}, 4, 1)
    v = sparse({}, 9, 1)
    w = sparse({}, 9, 1)
    add_term(u, v, w)
    
    # Part 4: A2 * B2 (2x1 * 1x5) schoolbook, rank 5
    # A2 is column 2 of A, so A2[i,0] = A[i,2] for i=0,1
    # U indices: i*m + 2, i=0,1
    #   A2[0,0] = U[0*3+2] = U[2]
    #   A2[1,0] = U[1*3+2] = U[5]
    # B2 is row 2 of B, so B2[0,k] = B[2,k] for k=0,1,2,3,4
    # V indices: 2*p + k, k=0,1,2,3,4
    #   B2[0,0] = V[2*5+0] = V[10]
    #   B2[0,1] = V[2*5+1] = V[11]
    #   B2[0,2] = V[2*5+2] = V[12]
    #   B2[0,3] = V[2*5+3] = V[13]
    #   B2[0,4] = V[2*5+4] = V[14]
    # Result C2 is 2x5, W indices: k*n + i, k=0..4; i=0,1
    #   C2[i,k] = A2[i,0] * B2[0,k]
    
    # For each i in [0,1] and k in [0,4]:
    for i in range(2):
        for k in range(5):
            u = sparse({}, i * 3 + 2, 1)  # A2[i,0]
            v = sparse({}, 2 * 5 + k, 1)  # B2[0,k]
            w = sparse({}, k * 2 + i, 1)  # C2[i,k]
            add_term(u, v, w)
    
    # Now build the matrices
    total_rank = rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for c in range(total_rank):
        u_dict = U_list[c]
        v_dict = V_list[c]
        w_dict = W_list[c]
        for idx, val in u_dict.items():
            U[idx][c] = val
        for idx, val in v_dict.items():
            V[idx][c] = val
        for idx, val in w_dict.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
