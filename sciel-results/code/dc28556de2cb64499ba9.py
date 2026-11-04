"""Matrix multiplication 2,4,6 - Strassen block decomposition."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Split m=4 into 2 groups of 2: j_groups = [0,1], [2,3]
    # Split p=6 into 3 groups of 2: k_groups = [0,1], [2,3], [4,5]
    # n=2, no split on i
    # 1 * 2 * 3 = 6 blocks of 2x2x2
    
    # Strassen's 7 terms for 2x2x2:
    # For matrices A(2x2), B(2x2), C(2x2) = A*B
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B10)
    # m5 = (A00+A10)B11
    # m6 = (A11-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    # C00 = m1+m4-m5+m7
    # C01 = m3+m5
    # C10 = m2+m4
    # C11 = m1-m2+m3+m6
    
    # Each Strassen term is a bilinear form.
    # For block (i_group, j_group, k_group), the 2x2x2 tensor has:
    # U indices: (i_local, j_local) where i_local in {0,1}, j_local in {0,1}
    # V indices: (j_local, k_local) where j_local in {0,1}, k_local in {0,1}
    # W indices: (k_local, i_local) where k_local in {0,1}, i_local in {0,1}
    
    # Strassen coefficients for each of 7 terms:
    # term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: 4 values for (i0j0, i0j1, i1j0, i1j1)
    # V_coeffs: 4 values for (j0k0, j0k1, j1k0, j1k1)
    # W_coeffs: 4 values for (k0i0, k0i1, k1i0, k1i1)
    
    strassen_terms = [
        # m1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        {
            'U': [1, 0, 0, 1],      # A00 + A11
            'V': [1, 0, 0, 1],      # B00 + B11
            'W': [1, 0, 0, 1],      # C00 + C11 (but we need C00 and C11 separately)
        },
    ]
    
    # Actually, let me be more careful. The tensor decomposition needs:
    # For each term t, U_t, V_t, W_t such that:
    # T[a,b,c] = sum_t U_t[a] * V_t[b] * W_t[c]
    
    # For 2x2x2 with indices a=(i,j), b=(j',k), c=(k',i')
    # T[(i,j),(j',k),(k',i')] = 1 if j==j' and k==k' and i==i', else 0
    
    # Strassen terms (standard):
    # Let me define them properly.
    
    # The 7 rank-1 tensors for Strassen:
    strassen_7 = [
        # Each is (U_vec, V_vec, W_vec) of length 4 each
        # Indexing: U[(i,j)], V[(j',k)], W[(k',i')]
        # m1: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # This gives C00 and C11 both get +1
        # But we need to be careful with signs for other terms
    ]
    
    # Let me use the standard Strassen formulation more carefully.
    # The 7 bilinear forms and their contributions:
    
    # I'll define each term as (u, v, w) where:
    # u[i*2+j] for i,j in {0,1}
    # v[j'*2+k] for j',k in {0,1}  
    # w[k'*2+i] for k',i in {0,1}
    
    # Term 1: (A00+A11)(B00+B11) -> adds to C00 and C11
    t1_u = [1, 0, 0, 1]  # A00 + A11
    t1_v = [1, 0, 0, 1]  # B00 + B11
    t1_w = [1, 0, 0, 1]  # C00 + C11
    
    # Term 2: (A10+A11)*B00 -> adds to C10
    t2_u = [0, 0, 1, 1]  # A10 + A11
    t2_v = [1, 0, 0, 0]  # B00
    t2_w = [0, 0, 1, 0]  # C10
    
    # Term 3: A00*(B10-B11) -> adds to C01
    t3_u = [1, 0, 0, 0]  # A00
    t3_v = [0, -1, 1, 0] # B10 - B11
    t3_w = [0, 1, 0, 0]  # C01
    
    # Term 4: A11*(B01-B10) -> adds to C00 and C10
    t4_u = [0, 0, 0, 1]  # A11
    t4_v = [0, 1, -1, 0] # B01 - B10
    t4_w = [1, 0, 1, 0]  # C00 + C10
    
    # Term 5: (A00+A10)*B11 -> subtracts from C00, adds to C01
    t5_u = [1, 0, 1, 0]  # A00 + A10
    t5_v = [0, 0, 0, 1]  # B11
    t5_w = [-1, 0, 0, 1] # -C00 + C01
    
    # Term 6: (A11-A00)*(B00+B10) -> adds to C11
    t6_u = [1, 0, 0, -1] # A11 - A00
    t6_v = [1, 0, 1, 0]  # B00 + B10
    t6_w = [0, 0, 0, 1]  # C11
    
    # Term 7: (A01-A11)*(B01+B11) -> adds to C00
    t7_u = [0, 1, 0, -1] # A01 - A11
    t7_v = [0, 1, 0, 1]  # B01 + B11
    t7_w = [1, 0, 0, 0]  # C00
    
    # Verify: C00 = t1 + t4 - t5 + t7
    # C00 from t1: 1, t4: 1, t5: -1, t7: 1 -> total 2? That's wrong.
    
    # I'm making errors. Let me look up the correct Strassen formulation.
    
    # Standard Strassen for C = A*B:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b12-b21)
    # m5 = (a11+a12)b22
    # m6 = (a22-a11)(b11+b21)
    # m7 = (a12-a22)(b12+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    # So the W coefficients should reflect the sum of contributions.
    # Let me redo:
    
    # For each term, W should have the coefficient in each C position.
    
    # m1: contributes +1 to c11, +1 to c22
    # m2: contributes +1 to c21, -1 to c22
    # m3: contributes +1 to c12, +1 to c22
    # m4: contributes +1 to c11, +1 to c21
    # m5: contributes -1 to c11, +1 to c12
    # m6: contributes +1 to c22
    # m7: contributes +1 to c11
    
    # Now let me define properly:
    
    strassen_decomp = [
        # (U, V, W) for each of 7 terms
        # U: [A00, A01, A10, A11] coefficients
        # V: [B00, B01, B10, B11] coefficients  
        # W: [C00, C01, C10, C11] coefficients
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # m2
        ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 1]),   # m3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),   # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # m5
        ([1, 0, 0, -1], [1, 0, 1, 0], [0, 0, 0, 1]),   # m6
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),   # m7
    ]
    
    # Now build the full decomposition for 2x4x6
    # 6 blocks: (j_group, k_group) where j_group in {0,1}, k_group in {0,1,2}
    # j_group 0: j_local 0->j=0, j_local 1->j=1
    # j_group 1: j_local 0->j=2, j_local 1->j=3
    # k_group 0: k_local 0->k=0, k_local 1->k=1
    # k_group 1: k_local 0->k=2, k_local 1->k=3
    # k_group 2: k_local 0->k=4, k_local 1->k=5
    
    rank = 6 * 7  # 42
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    for jg in range(2):  # 2 j-groups
        for kg in range(3):  # 3 k-groups
            for (su, sv, sw) in strassen_decomp:
                # U: for i in {0,1}, j_local in {0,1}
                # U index: i*m + j = i*4 + (jg*2 + j_local)
                for i in range(2):
                    for jl in range(2):
                        u_idx = i * m + (jg * 2 + jl)
                        u_coeff = su[jl * 2 + i]  # su indexed as [A00,A01,A10,A11] = [i0j0,i0j1,i1j0,i1j1]
                        # Wait, su is [A00, A01, A10, A11], so index = i*2 + j_local
                        u_coeff = su[i * 2 + jl]
                        U[u_idx][col] += u_coeff
                
                # V: for j_local in {0,1}, k_local in {0,1}
                # V index: j*p + k = (jg*2 + j_local)*6 + (kg*2 + k_local)
                for jl in range(2):
                    for kl in range(2):
                        v_idx = (jg * 2 + jl) * p + (kg * 2 + kl)
                        v_coeff = sv[jl * 2 + kl]  # sv is [B00,B01,B10,B11] = [j0k0,j0k1,j1k0,j1k1]
                        V[v_idx][col] += v_coeff
                
                # W: for k_local in {0,1}, i in {0,1}
                # W index: k*n + i = (kg*2 + k_local)*2 + i
                for kl in range(2):
                    for i in range(2):
                        w_idx = (kg * 2 + kl) * n + i
                        w_coeff = sw[kl * 2 + i]  # sw is [C00,C01,C10,C11] = [k0i0,k0i1,k1i0,k1i1]
                        W[w_idx][col] += w_coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
