"""
Optimized 6x6x6 matrix multiplication via tensor product of 3x3 schoolbook
and 2x2 Strassen decompositions.

Strategy:
  - 6x6 = (3x3) ⊗ (2x2)
  - 3x3 schoolbook: rank 27, U/V/W are 9x27 identity-like
  - 2x2 Strassen: rank 7, U/V/W are 4x7 with ±1 entries
  - Tensor product rank = 27 * 7 = 189
  - This is strictly better than schoolbook rank 216.

The output follows the trace convention W[k*n + i, r].
"""

import numpy as np

DIMENSIONS = [6, 6, 6]


def _strassen_2x2():
    """
    Return the 2x2 Strassen decomposition as (U, V, W) arrays.
    U: (4, 7), V: (4, 7), W: (4, 7)
    
    A is indexed as A[row, col] where row = 0,1 (i), col = 0,1 (j)
    B is indexed as B[row, col] where row = 0,1 (j), col = 0,1 (k)
    C is indexed as C[row, col] where row = 0,1 (k), col = 0,1 (i)
    
    So U[i, r] picks A[i, j] entries, V[j, r] picks B[j, k] entries,
    and W[k, r] picks C[k, i] entries.
    
    Strassen formulas (for C = A * B where A, B, C are 2x2):
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M5 + M6
    C[1,0] = M2 + M3
    C[1,1] = M1 + M3 - M4 + M6
    
    With:
    M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    M2 = (A[1,0] + A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1] - B[1,1])
    M4 = A[1,1] * (B[1,0] - B[0,0])
    M5 = (A[0,0] + A[0,1]) * B[1,1]
    M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    U[row, r] where row = i (0 or 1), r = 0..6:
    M1: A[0,0] + A[1,1] -> U[0,0]=1, U[1,0]=1
    M2: A[1,0] + A[1,1] -> U[1,1]=1, U[1,1]=1 (wait, U[1,1] for A[1,0] and U[1,1] for A[1,1]?)
    
    Let me be more careful. U is a 2x7 matrix where U[i, r] is the coefficient
    of A[i, j] in the r-th Strassen product. But A has indices (i, j) where i is row,
    j is col. So U should be indexed by (i, j) flattened to i*2+j, but the problem
    says U has shape (n*m, rank) = (2*2, 7) = (4, 7).
    
    So U is 4x7, where row index = i*2 + j (for n=2, m=2).
    V is 4x7, where row index = j*2 + k (for m=2, p=2).
    W is 4x7, where row index = k*2 + i (for p=2, n=2).
    
    Let me redefine:
    A indices: (i, j) with i in {0,1}, j in {0,1}. U[i*2+j, r].
    B indices: (j, k) with j in {0,1}, k in {0,1}. V[j*2+k, r].
    C indices: (k, i) with k in {0,1}, i in {0,1}. W[k*2+i, r].
    
    Strassen products:
    M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
      U: A[0,0] -> row 0*2+0=0, coeff 1; A[1,1] -> row 1*2+1=3, coeff 1
      V: B[0,0] -> row 0*2+0=0, coeff 1; B[1,1] -> row 1*2+1=3, coeff 1
      W: contributes to C[0,0] and C[1,1]
        C[0,0]: k=0, i=0 -> row 0*2+0=0, coeff +1
        C[1,1]: k=1, i=1 -> row 1*2+1=3, coeff +1
    
    M2 = (A[1,0] + A[1,1]) * B[0,0]
      U: A[1,0] -> row 1*2+0=2, coeff 1; A[1,1] -> row 3, coeff 1
      V: B[0,0] -> row 0, coeff 1
      W: contributes to C[1,0] and C[1,1]
        C[1,0]: k=1, i=0 -> row 1*2+0=2, coeff +1
        C[1,1]: k=1, i=1 -> row 3, coeff -1  (because C[1,1] = M1 + M3 - M4 + M6, no M2)
        Wait, let me recheck: C[1,0] = M2 + M3, so M2 contributes +1 to C[1,0].
        C[1,1] does not include M2. So W for M2: row 2 (k=1,i=0) coeff 1, others 0.
    
    M3 = A[0,0] * (B[0,1] - B[1,1])
      U: A[0,0] -> row 0, coeff 1
      V: B[0,1] -> row 0*2+1=1, coeff 1; B[1,1] -> row 3, coeff -1
      W: contributes to C[0,1] and C[1,0]
        C[0,1] = M5 + M6? No, let me recheck the Strassen formulas.
        
        Actually, let me use the standard Strassen:
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M5 + M6
        C[1,0] = M2 + M3
        C[1,1] = M1 + M3 - M4 + M6
        
        So M3 contributes to C[1,0] (coeff +1) and C[1,1] (coeff +1).
        C[1,0]: k=1, i=0 -> row 2, coeff +1
        C[1,1]: k=1, i=1 -> row 3, coeff +1
    
    M4 = A[1,1] * (B[1,0] - B[0,0])
      U: A[1,1] -> row 3, coeff 1
      V: B[1,0] -> row 1*2+0=2, coeff 1; B[0,0] -> row 0, coeff -1
      W: contributes to C[0,0] (coeff +1) and C[1,1] (coeff -1)
        C[0,0]: k=0, i=0 -> row 0, coeff +1
        C[1,1]: k=1, i=1 -> row 3, coeff -1
    
    M5 = (A[0,0] + A[0,1]) * B[1,1]
      U: A[0,0] -> row 0, coeff 1; A[0,1] -> row 1, coeff 1
      V: B[1,1] -> row 3, coeff 1
      W: contributes to C[0,0] (coeff -1) and C[0,1] (coeff +1)
        C[0,0]: k=0, i=0 -> row 0, coeff -1
        C[0,1]: k=0, i=1 -> row 1, coeff +1
    
    M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
      U: A[1,0] -> row 2, coeff 1; A[0,0] -> row 0, coeff -1
      V: B[0,0] -> row 0, coeff 1; B[0,1] -> row 1, coeff 1
      W: contributes to C[0,1] (coeff +1) and C[1,1] (coeff +1)
        C[0,1]: k=0, i=1 -> row 1, coeff +1
        C[1,1]: k=1, i=1 -> row 3, coeff +1
    
    M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
      U: A[0,1] -> row 1, coeff 1; A[1,1] -> row 3, coeff -1
      V: B[1,0] -> row 2, coeff 1; B[1,1] -> row 3, coeff 1
      W: contributes to C[0,0] (coeff +1)
        C[0,0]: k=0, i=0 -> row 0, coeff +1
    
    Now let's build the matrices:
    
    U (4x7), rows = [i*2+j] = [0,1,2,3], cols = [M1..M7] = [0..6]
    M1: row0=1, row3=1
    M2: row2=1, row3=1
    M3: row0=1
    M4: row3=1
    M5: row0=1, row1=1
    M6: row2=1, row0=-1
    M7: row1=1, row3=-1
    
    U = [
      [1, 0, 1, 0, 1, -1, 0],  # row 0 (A[0,0])
      [0, 0, 0, 0, 1,  0, 1],  # row 1 (A[0,1])
      [0, 1, 0, 0, 0,  1, 0],  # row 2 (A[1,0])
      [1, 1, 0, 1, 0,  0,-1],  # row 3 (A[1,1])
    ]
    
    V (4x7), rows = [j*2+k] = [0,1,2,3], cols = [M1..M7]
    M1: row0=1, row3=1
    M2: row0=1
    M3: row1=1, row3=-1
    M4: row2=1, row0=-1
    M5: row3=1
    M6: row0=1, row1=1
    M7: row2=1, row3=1
    
    V = [
      [1, 1, 0, -1, 0, 1, 0],  # row 0 (B[0,0])
      [0, 0, 1,  0, 0, 1, 0],  # row 1 (B[0,1])
      [0, 0, 0,  1, 0, 0, 1],  # row 2 (B[1,0])
      [1, 0,-1,  0, 1, 0, 1],  # row 3 (B[1,1])
    ]
    
    W (4x7), rows = [k*2+i] = [0,1,2,3], cols = [M1..M7]
    M1: row0=1, row3=1
    M2: row2=1
    M3: row2=1, row3=1
    M4: row0=1, row3=-1
    M5: row0=-1, row1=1
    M6: row1=1, row3=1
    M7: row0=1
    
    W = [
      [1, 0, 0, 1, -1, 0, 1],  # row 0 (C[0,0]: k=0,i=0)
      [0, 0, 0, 0,  1, 1, 0],  # row 1 (C[0,1]: k=0,i=1)
      [0, 1, 1, 0,  0, 0, 0],  # row 2 (C[1,0]: k=1,i=0)
      [1, 0, 1,-1,  0, 1, 0],  # row 3 (C[1,1]: k=1,i=1)
    ]
    
    Let me verify C[0,0] = sum_r W[0,r] * (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k])
    
    For r=0 (M1): W[0,0]=1. U col 0: [1,0,0,1] -> A[0,0]+A[1,1]. V col 0: [1,0,0,1] -> B[0,0]+B[1,1]. Product = (A00+A11)(B00+B11).
    For r=1 (M2): W[0,1]=0. Skip.
    For r=2 (M3): W[0,2]=0. Skip.
    For r=3 (M4): W[0,3]=1. U col 3: [0,0,0,1] -> A[1,1]. V col 3: [-1,0,1,0] -> -B[0,0]+B[1,0]. Product = A11(B10-B00).
    For r=4 (M5): W[0,4]=-1. U col 4: [1,1,0,0] -> A[0,0]+A[0,1]. V col 4: [0,0,0,1] -> B[1,1]. Product = (A00+A01)B11. With coeff -1: -(A00+A01)B11.
    For r=5 (M6): W[0,5]=0. Skip.
    For r=6 (M7): W[0,6]=1. U col 6: [0,1,0,-1] -> A[0,1]-A[1,1]. V col 6: [0,0,1,1] -> B[1,0]+B[1,1]. Product = (A01-A11)(B10+B11).
    
    Sum: (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    = A00B00 + A01B10
    = C[0,0]. Correct!
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1],
    ], dtype=int)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1],
    ], dtype=int)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 0, 0,  1, 1, 0],
        [0, 1, 1, 0,  0, 0, 0],
        [1, 0, 1,-1,  0, 1, 0],
    ], dtype=int)
    return U, V, W


def _schoolbook_3x3():
    """
    Return the 3x3 schoolbook decomposition as (U, V, W) arrays.
    U: (9, 27), V: (9, 27), W: (9, 27)
    Row indices: U[i*3+j], V[j*3+k], W[k*3+i]
    Each of the 27 columns corresponds to a unique (i,j,k) triple with U, V, W having 1.
    """
    n = m = p = 3
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    """
    Construct the 6x6x6 matrix multiplication tensor decomposition using
    the tensor product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7).
    Total rank = 27 * 7 = 189.
    """
    # Get the component decompositions
    U3, V3, W3 = _schoolbook_3x3()  # shapes (9, 27)
    U2, V2, W2 = _strassen_2x2()    # shapes (4, 7)
    
    n3, m3, p3 = 3, 3, 3
    n2, m2, p2 = 2, 2, 2
    r3 = 27
    r2 = 7
    
    # For the 6x6x6 case:
    # n = n3 * n2 = 6, m = m3 * m2 = 6, p = p3 * p2 = 6
    # A has indices (i, j) where i = i3 * n2 + i2, j = j3 * m2 + j2
    # B has indices (j, k) where j = j3 * m2 + j2, k = k3 * p2 + k2
    # C has indices (k, i) where k = k3 * p2 + k2, i = i3 * n2 + i2
    
    # The tensor product decomposition:
    # U[i3*n2+i2 * m + (j3*m2+j2), r3*r2 + c2] = U3[i3*m3+j3, r3] * U2[i2*m2+j2, c2]
    # But wait, the row index of U is i*m + j = (i3*n2+i2)*6 + (j3*m2+j2)
    # = i3*n2*6 + i2*6 + j3*2 + j2
    # And U3 row is i3*m3 + j3 = i3*3 + j3
    # U2 row is i2*m2 + j2 = i2*2 + j2
    
    # So U[row_6, c] = U3[i3*3+j3, c3] * U2[i2*2+j2, c2] where row_6 = (i3*2+i2)*6 + (j3*2+j2)
    # and c = c3*7 + c2
    
    rank = r3 * r2  # 189
    n, m, p = 6, 6, 6
    U6 = np.zeros((n * m, rank), dtype=int)
    V6 = np.zeros((m * p, rank), dtype=int)
    W6 = np.zeros((p * n, rank), dtype=int)
    
    for c3 in range(r3):
        for c2 in range(r2):
            c = c3 * r2 + c2
            for i3 in range(n3):
                for i2 in range(n2):
                    for j3 in range(m3):
                        for j2 in range(m2):
                            row_U3 = i3 * m3 + j3
                            row_U2 = i2 * m2 + j2
                            u3 = U3[row_U3, c3]
                            u2 = U2[row_U2, c2]
                            if u3 == 0 or u2 == 0:
                                continue
                            row_U6 = (i3 * n2 + i2) * m + (j3 * m2 + j2)
                            U6[row_U6, c] = u3 * u2
            
            for j3 in range(m3):
                for j2 in range(m2):
                    for k3 in range(p3):
                        for k2 in range(p2):
                            row_V3 = j3 * p3 + k3
                            row_V2 = j2 * p2 + k2
                            v3 = V3[row_V3, c3]
                            v2 = V2[row_V2, c2]
                            if v3 == 0 or v2 == 0:
                                continue
                            row_V6 = (j3 * p2 + j2) * p + (k3 * p2 + k2)
                            V6[row_V6, c] = v3 * v2
            
            for k3 in range(p3):
                for k2 in range(p2):
                    for i3 in range(n3):
                        for i2 in range(n2):
                            row_W3 = k3 * n3 + i3
                            row_W2 = k2 * n2 + i2
                            w3 = W3[row_W3, c3]
                            w2 = W2[row_W2, c2]
                            if w3 == 0 or w2 == 0:
                                continue
                            row_W6 = (k3 * n2 + k2) * n + (i3 * n2 + i2)
                            W6[row_W6, c] = w3 * w2
    
    # Convert to lists for the return format
    U_list = U6.tolist()
    V_list = V6.tolist()
    W_list = W6.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    print(f"U shape: ({len(result['U'])}, {rank})")
    print(f"V shape: ({len(result['V'])}, {rank})")
    print(f"W shape: ({len(result['W'])}, {rank})")
    
    # Quick verification with random matrices
    n, m, p = 6, 6, 6
    A = np.random.randint(-5, 6, size=(n, m))
    B = np.random.randint(-5, 6, size=(m, p))
    C_expected = A @ B
    
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    C_computed = np.zeros((p, n))
    for r in range(rank):
        a_vec = U @ A.reshape(n * m)
        b_vec = V @ B.reshape(m * p)
        c_outer = W[:, r] * (a_vec * b_vec)
        # Reshape c_outer to (p, n)
        C_computed += c_outer.reshape(p, n)
    
    # But wait, the convention is W[k*n+i, r], so C[k, i] = sum_r W[k*n+i, r] * U_col * V_col
    # Let me redo the verification properly
    C_computed = np.zeros((n, p))  # C[i, k]
    for r in range(rank):
        # U column r gives coefficients for A[i,j]
        # V column r gives coefficients for B[j,k]
        # W column r gives coefficients for C[k,i]
        a_vals = U[:, r]  # length n*m
        b_vals = V[:, r]  # length m*p
        w_vals = W[:, r]  # length p*n
        
        a_mat = a_vals.reshape(n, m)
        b_mat = b_vals.reshape(m, p)
        w_mat = w_vals.reshape(p, n)
        
        product = a_mat @ A @ b_mat  # This is wrong, let me think again
        
    # Actually, the tensor decomposition means:
    # C[k, i] = sum_r W[k*n+i, r] * (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j,k} V[j*p+k, r] * B[j,k])
    # But this doesn't look right either. The standard form is:
    # C = sum_r W_r * (U_r^T A) * (V_r^T B) ... no.
    
    # The correct interpretation: the multiplication tensor T has T[i,j,k] = 1 if... 
    # Actually, for matrix multiplication, the tensor is:
    # T[A_idx, B_idx, C_idx] where A_idx = (i,j), B_idx = (j,k), C_idx = (k,i)
    # T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # So C[k,i] = sum_{i,j} sum_{j,k} T[(i,j),(j,k),(k,i)] A[i,j] B[j,k]
    #           = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) (sum_{j,k} V[j*p+k,r] B[j,k]) W[k*n+i,r]
    
    # Let me verify numerically
    C_check = np.zeros((p, n))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        a_sum = np.sum(u_col.reshape(n, m) * A)
        b_sum = np.sum(v_col.reshape(m, p) * B)
        C_check += w_col * (a_sum * b_sum)
    
    C_check = C_check.reshape(p, n)
    # C_check[k, i] should equal C_expected[i, k]
    C_final = C_check.T  # C_final[i, k] = C_check[k, i]
    
    if np.array_equal(C_final, C_expected):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print("Max diff:", np.max(np.abs(C_final - C_expected)))

