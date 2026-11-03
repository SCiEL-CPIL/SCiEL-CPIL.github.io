"""Matrix multiplication 3x3x3 rank-26 decomposition (Strassen core + border)."""
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # We build a rank-26 decomposition.
    # The 3x3 matrices are partitioned into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1.
    # C = [[C11, C12], [C21, C22]] where C11 is 2x2, C12 is 2x1, C21 is 1x2, C22 is 1x1.
    #
    # The 2x2x2 core uses Strassen's 7 multiplications.
    # The border terms (involving indices 2) require additional multiplications.
    #
    # Total rank = 7 (Strassen) + 19 (border) = 26.
    #
    # Let's label the indices:
    # A indices: (0,0), (0,1), (1,0), (1,1), (0,2), (1,2), (2,0), (2,1), (2,2)
    # B indices: (0,0), (0,1), (1,0), (1,1), (0,2), (1,2), (2,0), (2,1), (2,2)
    # C indices: (0,0), (0,1), (1,0), (1,1), (0,2), (1,2), (2,0), (2,1), (2,2)
    #
    # Strassen's 7 products for the 2x2 core (indices 0,1):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Wait, the standard Strassen formulas are:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Let me verify:
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # M4 = A22(B21-B11) = A22B21 - A22B11
    # M5 = (A11+A12)B22 = A11B22 + A12B22
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
    #
    # C11 = M1 + M4 - M5 + M7
    #      = (A11B11 + A11B22 + A22B11 + A22B22) + (A22B21 - A22B11) - (A11B22 + A12B22) + (A12B21 + A12B22 - A22B21 - A22B22)
    #      = A11B11 + A11B22 + A22B11 + A22B22 + A22B21 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
    #      = A11B11 + A12B21
    #
    # That's not right. C11 should be A11B11 + A12B21.
    # Yes, it is! Because C11 = A11*B11 + A12*B21 (for the 2x2 block product).
    # Wait, C is 2x2, A is 2x2, B is 2x2.
    # C11 = A11*B11 + A12*B21. Yes.
    #
    # C12 = M3 + M5
    # M3 = A11(B12-B22) = A11B12 - A11B22
    # M5 = (A11+A12)B22 = A11B22 + A12B22
    # C12 = A11B12 - A11B22 + A11B22 + A12B22 = A11B12 + A12B22. Correct.
    #
    # C21 = M2 + M4
    # M2 = (A21+A22)B11 = A21B11 + A22B11
    # M4 = A22(B21-B11) = A22B21 - A22B11
    # C21 = A21B11 + A22B11 + A22B21 - A22B11 = A21B11 + A22B21. Correct.
    #
    # C22 = M1 - M2 + M3 + M6
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
    # C22 = (A11B11 + A11B22 + A22B11 + A22B22) - (A21B11 + A22B11) + (A11B12 - A11B22) + (A21B11 + A21B12 - A11B11 - A11B12)
    #     = A11B11 + A11B22 + A22B11 + A22B22 - A21B11 - A22B11 + A11B12 - A11B22 + A21B11 + A21B12 - A11B11 - A11B12
    #     = A22B22 + A21B12. Correct.
    #
    # So the Strassen 7 products give us the 2x2 block product.
    #
    # Now for the border terms (involving index 2):
    # c00 = a00*b00 + a01*b10 + a02*b20  -> a02*b20 is new (1 mult)
    # c01 = a00*b01 + a01*b11 + a02*b21  -> a02*b21 is new (1 mult)
    # c02 = a00*b02 + a01*b12 + a02*b22  -> 3 new mults
    # c10 = a10*b00 + a11*b10 + a12*b20  -> a12*b20 is new (1 mult)
    # c11 = a10*b01 + a11*b11 + a12*b21  -> a12*b21 is new (1 mult)
    # c12 = a10*b02 + a11*b12 + a12*b22  -> 3 new mults
    # c20 = a20*b00 + a21*b10 + a22*b20  -> 3 new mults
    # c21 = a20*b01 + a21*b11 + a22*b21  -> 3 new mults
    # c22 = a20*b02 + a21*b12 + a22*b22  -> 3 new mults
    #
    # Total new mults: 1+1+3+1+1+3+3+3+3 = 19.
    # Total rank = 7 + 19 = 26.
    #
    # Now I need to build U, V, W matrices of shape (9, 26), (9, 26), (9, 26).
    # U[i, r] = coefficient of A index i in multiplication r.
    # V[j, r] = coefficient of B index j in multiplication r.
    # W[k, r] = coefficient of C index k in multiplication r.
    #
    # The decomposition is: C_k = sum_r U_k,r * A_r * V_r * B_r ... 
    # Actually, the standard form is:
    # sum_{i,j} A_{i,j} B_{j,k} = sum_r (sum_i U_{i,r} A_i) (sum_j V_{j,r} B_j) W_{k,r}
    # Wait, no. The tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where T_{i,j,k} = 1 if j == i' and k == j' ... 
    # Actually, for matrix multiplication, the tensor is:
    # T_{a,b,c} = 1 if c = a and b = b (where a is row of A, b is col of A = row of B, c is col of B).
    # No, T_{i,j,k} = delta_{j,k} ... no.
    # 
    # Let me think again. The matrix multiplication tensor is:
    # sum_{j} A_{i,j} B_{j,k}
    # So the tensor T has T_{i,j,k} = 1 if the indices match appropriately.
    # Specifically, T_{i,j,k} = delta_{j, j} ... 
    # 
    # The decomposition is:
    # A_{i,j} B_{j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where for each r, we multiply (sum_i U_{i,r} A_i) * (sum_j V_{j,r} B_j) and add W_{k,r} times that to C_k.
    #
    # Wait, that's not quite right either. Let me use the standard formulation:
    # The bilinear map (A, B) -> C where C = A*B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    #
    # A rank-1 term is: (sum_i u_i A_i) * (sum_j v_j B_j) * w_{i,k}
    # where u_i, v_j are scalars, and w_{i,k} is the coefficient for C_{i,k}.
    #
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where T_{i,j,k} = 1 if j == k (for the summation index) ... 
    #
    # No, let me look at this more carefully.
    # The tensor for matrix multiplication M_{n,m,p} has entries T_{i,j,k} where:
    # i indexes rows of A (0..n-1)
    # j indexes columns of A / rows of B (0..m-1)  
    # k indexes columns of B (0..p-1)
    #
    # T_{i,j,k} = 1 if the product A_{i,j} * B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all valid i,j,k.
    #
    # The decomposition is:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    #
    # Wait, that's not right either. Let me think about it as:
    # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r (sum_i' U_{i',r} A_{i',j'}) ... 
    #
    # Actually, I think the correct formulation is:
    # For each rank-r term r:
    #   A_r = sum_i U_{i,r} * e_i (a vector in A-space)
    #   B_r = sum_j V_{j,r} * e_j (a vector in B-space)
    #   W_r = sum_k W_{k,r} * e_k (a vector in C-space)
    #   The term contributes: W_r * (A_r^T * B_r)
    #
    # So: C = sum_r W_r * (A_r^T * B_r)
    # where A_r^T * B_r is the matrix product of the rank-1 matrices A_r B_r^T ... 
    #
    # I'm getting confused. Let me just use the direct approach:
    # For each multiplication term (which is a product of a linear combination of A entries and a linear combination of B entries), 
    # I add 1 to U at the A indices involved, 1 to V at the B indices involved, and the appropriate coefficient to W at the C indices.
    #
    # For Strassen's M1 = (A11 + A22)(B11 + B22):
    # U has 1 at (0,0) and (1,1) [for A11 and A22]
    # V has 1 at (0,0) and (1,1) [for B11 and B22]
    # W has contributions to C11, C12, C21, C22 based on the Strassen formulas.
    #
    # From C11 = M1 + M4 - M5 + M7: W[0,0] += 1 (for M1)
    # From C12 = M3 + M5: W[0,1] += 0 (M1 doesn't contribute to C12)
    # From C21 = M2 + M4: W[1,0] += 0
    # From C22 = M1 - M2 + M3 + M6: W[1,1] += 1
    #
    # Wait, I need to be more careful. Let me just code it up directly.
    #
    # I'll create a list of 26 terms. Each term is a dict with:
    #   'A': dict mapping (i,j) -> coeff
    #   'B': dict mapping (j,k) -> coeff
    #   'C': dict mapping (i,k) -> coeff
    #
    # Then I'll build U, V, W from these.
    
    terms = []
    
    # Helper to add a term
    def add_term(A_dict, B_dict, C_dict):
        terms.append({'A': A_dict, 'B': B_dict, 'C': C_dict})
    
    # Strassen's 7 products for the 2x2 core (indices 0,1):
    # M1 = (A11 + A22)(B11 + B22)
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {})
    # M2 = (A21 + A22)(B11)
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {})
    # M3 = (A11)(B12 - B22)
    add_term({(0,0):1}, {(0,1):1, (1,1):-1}, {})
    # M4 = (A22)(B21 - B11)
    add_term({(1,1):1}, {(1,0):1, (0,0):-1}, {})
    # M5 = (A11 + A12)(B22)
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {})
    # M6 = (A21 - A11)(B11 + B12)
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {})
    # M7 = (A12 - A22)(B21 + B22)
    add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {})
    
    # Now assign C coefficients based on Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    terms[0]['C'][(0,0)] = 1   # M1 -> C11
    terms[0]['C'][(1,1)] = 1   # M1 -> C22
    terms[1]['C'][(1,0)] = 1   # M2 -> C21
    terms[1]['C'][(1,1)] = -1  # M2 -> C22
    terms[2]['C'][(0,1)] = 1   # M3 -> C12
    terms[2]['C'][(1,1)] = 1   # M3 -> C22
    terms[3]['C'][(0,0)] = 1   # M4 -> C11
    terms[3]['C'][(1,0)] = 1   # M4 -> C21
    terms[4]['C'][(0,0)] = -1  # M5 -> C11
    terms[4]['C'][(0,1)] = 1   # M5 -> C12
    terms[5]['C'][(1,1)] = 1   # M6 -> C22
    terms[6]['C'][(0,0)] = 1   # M7 -> C11
    
    # Border terms (19 additional multiplications):
    # c00 += a02*b20
    add_term({(0,2):1}, {(2,0):1}, {(0,0):1})
    # c01 += a02*b21
    add_term({(0,2):1}, {(2,1):1}, {(0,1):1})
    # c02 += a00*b02 + a01*b12 + a02*b22
    add_term({(0,0):1}, {(0,2):1}, {(0,2):1})
    add_term({(0,1):1}, {(1,2):1}, {(0,2):1})
    add_term({(0,2):1}, {(2,2):1}, {(0,2):1})
    # c10 += a12*b20
    add_term({(1,2):1}, {(2,0):1}, {(1,0):1})
    # c11 += a12*b21
    add_term({(1,2):1}, {(2,1):1}, {(1,1):1})
    # c12 += a10*b02 + a11*b12 + a12*b22
    add_term({(1,0):1}, {(0,2):1}, {(1,2):1})
    add_term({(1,1):1}, {(1,2):1}, {(1,2):1})
    add_term({(1,2):1}, {(2,2):1}, {(1,2):1})
    # c20 += a20*b00 + a21*b10 + a22*b20
    add_term({(2,0):1}, {(0,0):1}, {(2,0):1})
    add_term({(2,1):1}, {(1,0):1}, {(2,0):1})
    add_term({(2,2):1}, {(2,0):1}, {(2,0):1})
    # c21 += a20*b01 + a21*b11 + a22*b21
    add_term({(2,0):1}, {(0,1):1}, {(2,1):1})
    add_term({(2,1):1}, {(1,1):1}, {(2,1):1})
    add_term({(2,2):1}, {(2,1):1}, {(2,1):1})
    # c22 += a20*b02 + a21*b12 + a22*b22
    add_term({(2,0):1}, {(0,2):1}, {(2,2):1})
    add_term({(2,1):1}, {(1,2):1}, {(2,2):1})
    add_term({(2,2):1}, {(2,2):1}, {(2,2):1})
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        for (i, j), coeff in term['A'].items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in term['B'].items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in term['C'].items():
            W[k * n + i][r] = coeff
    
    return {"U": U, "V": V, "W": W}
