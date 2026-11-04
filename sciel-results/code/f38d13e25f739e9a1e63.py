"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 20 decomposition
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_indices: list of (i, j)
    # v_indices: list of (j, k)
    # w_indices: list of (k, i)
    # coeffs: dict mapping index to coeff
    def add_term(col, u_coeffs, v_coeffs, w_coeffs):
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val

    # Block 1: j=3 (2x1x5) Rank 1
    # U[i, 3] = 1
    # V[3, k] = 1
    # W[k, i] = 1
    u1 = {0*m+3: 1, 1*m+3: 1}
    v1 = {3*p+k: 1 for k in range(p)}
    w1 = {k*n+i: 1 for k in range(p) for i in range(n)}
    add_term(0, u1, v1, w1)

    # Block 2: j=2 (2x1x5) Rank 1
    u2 = {0*m+2: 1, 1*m+2: 1}
    v2 = {2*p+k: 1 for k in range(p)}
    w2 = {k*n+i: 1 for k in range(p) for i in range(n)}
    add_term(1, u2, v2, w2)

    # Block 3: j in {0,1}, k in {0,1} (2x2x2) Rank 7 (Strassen)
    # Map local indices (i,j) -> global (i, j)
    # Map local indices (j,k) -> global (j, k)
    # Map local indices (k,i) -> global (k, i)
    
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Local A indices: 00, 01, 10, 11
    # Local B indices: 00, 01, 10, 11
    # Local C indices: 00, 01, 10, 11
    
    # We need to express M_c as sum of products of linear forms.
    # For each multiplication M_c, we define U, V, W entries.
    
    # Col 2: M1
    # U: A11=1, A22=1 -> (0,0)=1, (1,1)=1
    # V: B11=1, B22=1 -> (0,0)=1, (1,1)=1
    # W: C11 += M1, C22 += M1? No.
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # Let's just add the terms directly.
    # M1 contributes to C11 (+1), C22 (+1)
    u_m1 = {0*m+0: 1, 1*m+1: 1}
    v_m1 = {0*p+0: 1, 1*p+1: 1}
    w_m1 = {0*n+0: 1, 1*n+1: 1}
    add_term(2, u_m1, v_m1, w_m1)
    
    # M2 contributes to C21 (+1)
    u_m2 = {1*m+0: 1, 1*m+1: 1}
    v_m2 = {0*p+0: 1}
    w_m2 = {0*n+1: 1}
    add_term(3, u_m2, v_m2, w_m2)
    
    # M3 contributes to C12 (+1)
    u_m3 = {0*m+0: 1}
    v_m3 = {0*p+1: 1, 1*p+1: -1}
    w_m3 = {1*n+0: 1}
    add_term(4, u_m3, v_m3, w_m3)
    
    # M4 contributes to C11 (-1), C21 (+1)
    u_m4 = {1*m+1: 1}
    v_m4 = {1*p+0: 1, 0*p+0: -1}
    w_m4 = {0*n+0: -1, 0*n+1: 1}
    add_term(5, u_m4, v_m4, w_m4)
    
    # M5 contributes to C12 (+1), C22 (-1)
    u_m5 = {0*m+0: 1, 0*m+1: 1}
    v_m5 = {1*p+1: 1}
    w_m5 = {1*n+0: 1, 1*n+1: -1}
    add_term(6, u_m5, v_m5, w_m5)
    
    # M6 contributes to C11 (-1), C22 (+1)
    u_m6 = {1*m+0: 1, 0*m+0: -1}
    v_m6 = {0*p+0: 1, 0*p+1: 1}
    w_m6 = {0*n+0: -1, 1*n+1: 1}
    add_term(7, u_m6, v_m6, w_m6)
    
    # M7 contributes to C11 (+1), C12 (-1)
    u_m7 = {0*m+1: 1, 1*m+1: -1}
    v_m7 = {1*p+0: 1, 1*p+1: 1}
    w_m7 = {0*n+0: 1, 1*n+0: -1}
    add_term(8, u_m7, v_m7, w_m7)

    # Block 4: j in {0,1}, k in {2,3} (2x2x2) Rank 7
    # Same as Block 3, but k offset by 2.
    # k_local 0 -> k_global 2, k_local 1 -> k_global 3.
    
    # M1
    u_m1_b4 = {0*m+0: 1, 1*m+1: 1}
    v_m1_b4 = {0*p+2: 1, 1*p+3: 1}
    w_m1_b4 = {2*n+0: 1, 3*n+1: 1}
    add_term(9, u_m1_b4, v_m1_b4, w_m1_b4)
    
    # M2
    u_m2_b4 = {1*m+0: 1, 1*m+1: 1}
    v_m2_b4 = {0*p+2: 1}
    w_m2_b4 = {2*n+1: 1}
    add_term(10, u_m2_b4, v_m2_b4, w_m2_b4)
    
    # M3
    u_m3_b4 = {0*m+0: 1}
    v_m3_b4 = {0*p+3: 1, 1*p+3: -1}
    w_m3_b4 = {3*n+0: 1}
    add_term(11, u_m3_b4, v_m3_b4, w_m3_b4)
    
    # M4
    u_m4_b4 = {1*m+1: 1}
    v_m4_b4 = {1*p+2: 1, 0*p+2: -1}
    w_m4_b4 = {2*n+0: -1, 2*n+1: 1}
    add_term(12, u_m4_b4, v_m4_b4, w_m4_b4)
    
    # M5
    u_m5_b4 = {0*m+0: 1, 0*m+1: 1}
    v_m5_b4 = {1*p+3: 1}
    w_m5_b4 = {3*n+0: 1, 3*n+1: -1}
    add_term(13, u_m5_b4, v_m5_b4, w_m5_b4)
    
    # M6
    u_m6_b4 = {1*m+0: 1, 0*m+0: -1}
    v_m6_b4 = {0*p+2: 1, 0*p+3: 1}
    w_m6_b4 = {2*n+0: -1, 3*n+1: 1}
    add_term(14, u_m6_b4, v_m6_b4, w_m6_b4)
    
    # M7
    u_m7_b4 = {0*m+1: 1, 1*m+1: -1}
    v_m7_b4 = {1*p+2: 1, 1*p+3: 1}
    w_m7_b4 = {2*n+0: 1, 3*n+0: -1}
    add_term(15, u_m7_b4, v_m7_b4, w_m7_b4)

    # Block 5: j in {0,1}, k in {4} (2x2x1) Rank 4
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # Term 1: A00 B04
    u_t1 = {0*m+0: 1}
    v_t1 = {0*p+4: 1}
    w_t1 = {4*n+0: 1}
    add_term(16, u_t1, v_t1, w_t1)
    
    # Term 2: A01 B14
    u_t2 = {0*m+1: 1}
    v_t2 = {1*p+4: 1}
    w_t2 = {4*n+0: 1}
    add_term(17, u_t2, v_t2, w_t2)
    
    # Term 3: A10 B04
    u_t3 = {1*m+0: 1}
    v_t3 = {0*p+4: 1}
    w_t3 = {4*n+1: 1}
    add_term(18, u_t3, v_t3, w_t3)
    
    # Term 4: A11 B14
    u_t4 = {1*m+1: 1}
    v_t4 = {1*p+4: 1}
    w_t4 = {4*n+1: 1}
    add_term(19, u_t4, v_t4, w_t4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
