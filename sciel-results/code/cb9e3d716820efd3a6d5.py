"""
Rank-49 decomposition for 4x4 matrix multiplication via Kronecker product of 2x2 Strassen.
This constructs the factor matrices directly using the tensor product structure,
avoiding recursive indexing errors.
"""

import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses the Kronecker product of the 2x2 Strassen decomposition with itself.
    
    The 2x2 Strassen algorithm has rank 7 with factor matrices:
    - U2: shape (4, 7) where U2[i*2+j, r] is the coefficient of A[i,j] in factor r
    - V2: shape (4, 7) where V2[j*2+k, r] is the coefficient of B[j,k] in factor r  
    - W2: shape (4, 7) where W2[k*2+i, r] is the coefficient for C[i,k] in factor r
    
    For 4x4 matrices, we split into 2x2 blocks and apply Strassen recursively.
    The resulting rank-49 decomposition can be expressed as Kronecker products
    of the 2x2 factors.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the 2x2 Strassen factors
    # For 2x2 matrix multiplication:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    
    # Strassen's 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U2: 4x7 matrix, row index = i*2+j (A entry), col index = r (factor)
    # A entries: a00=0, a01=1, a10=2, a11=3
    U2 = np.zeros((4, 7), dtype=int)
    # M1: (a00+a11) -> a00 coeff 1, a11 coeff 1
    U2[0, 0] = 1; U2[3, 0] = 1
    # M2: (a10+a11) -> a10 coeff 1, a11 coeff 1
    U2[2, 1] = 1; U2[3, 1] = 1
    # M3: a00 -> a00 coeff 1
    U2[0, 2] = 1
    # M4: a11 -> a11 coeff 1
    U2[3, 3] = 1
    # M5: (a00+a01) -> a00 coeff 1, a01 coeff 1
    U2[0, 4] = 1; U2[1, 4] = 1
    # M6: (a10-a00) -> a10 coeff 1, a00 coeff -1
    U2[2, 5] = 1; U2[0, 5] = -1
    # M7: (a01-a11) -> a01 coeff 1, a11 coeff -1
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # V2: 4x7 matrix, row index = j*2+k (B entry), col index = r (factor)
    # B entries: b00=0, b01=1, b10=2, b11=3
    V2 = np.zeros((4, 7), dtype=int)
    # M1: (b00+b11) -> b00 coeff 1, b11 coeff 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2: b00 -> b00 coeff 1
    V2[0, 1] = 1
    # M3: (b01-b11) -> b01 coeff 1, b11 coeff -1
    V2[1, 2] = 1; V2[3, 2] = -1
    # M4: (b10-b00) -> b10 coeff 1, b00 coeff -1
    V2[2, 3] = 1; V2[0, 3] = -1
    # M5: b11 -> b11 coeff 1
    V2[3, 4] = 1
    # M6: (b00+b01) -> b00 coeff 1, b01 coeff 1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7: (b10+b11) -> b10 coeff 1, b11 coeff 1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W2: 4x7 matrix, row index = k*2+i (C entry), col index = r (factor)
    # C entries: c00=0, c01=1, c10=2, c11=3
    W2 = np.zeros((4, 7), dtype=int)
    # M1 contributes to C00 (+1) and C11 (+1)
    W2[0, 0] = 1; W2[3, 0] = 1
    # M2 contributes to C10 (+1) and C11 (-1)
    W2[2, 1] = 1; W2[3, 1] = -1
    # M3 contributes to C01 (+1) and C11 (+1)
    W2[1, 2] = 1; W2[3, 2] = 1
    # M4 contributes to C00 (+1) and C10 (+1)
    W2[0, 3] = 1; W2[2, 3] = 1
    # M5 contributes to C00 (-1) and C01 (+1)
    W2[0, 4] = -1; W2[1, 4] = 1
    # M6 contributes to C11 (+1)
    W2[3, 5] = 1
    # M7 contributes to C00 (+1)
    W2[0, 6] = 1
    
    # Now construct the 4x4 decomposition using Kronecker product structure.
    # For 4x4 matrices, we split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # The 4x4 Strassen (recursive) computes 7 products of 2x2 matrices,
    # each of which is computed using 2x2 Strassen (7 scalar products).
    # Total rank = 7 * 7 = 49.
    
    # The factor matrices can be expressed as Kronecker products:
    # U = U2 ⊗ U2 (with appropriate indexing)
    # V = V2 ⊗ V2
    # W = W2 ⊗ W2
    
    # But we need to be careful with the indexing convention.
    # For 4x4:
    # U has shape (16, 49): U[i*4+j, r] where i,j in 0..3
    # V has shape (16, 49): V[j*4+k, r] where j,k in 0..3
    # W has shape (16, 49): W[k*4+i, r] where k,i in 0..3
    
    # The Kronecker product approach:
    # Factor r = (r1, r2) where r1, r2 in 0..6, so r = r1*7 + r2
    # U[i*4+j, r1*7+r2] = U2[i*2 + (j//2), r1] * U2[(j%2)*2 + (j//2)... 
    
    # Actually, let's think more carefully. The 4x4 matrix A can be viewed as:
    # A[i,j] where i = 2*i0 + i1, j = 2*j0 + j1 with i0,i1,j0,j1 in 0..1
    # So A[i,j] = A[i0,i1,j0,j1]
    
    # The 2x2 Strassen on the outer level uses blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # For the outer Strassen factor r1, we have linear combinations of blocks.
    # For the inner Strassen factor r2, we have linear combinations of elements within blocks.
    
    # U[i*4+j, r1*7+r2]:
    # i = 2*i0 + i1, j = 2*j0 + j1
    # The outer factor r1 acts on the block indices (i0, j0)
    # The inner factor r2 acts on the element indices (i1, j1) within each block
    
    # So U[i*4+j, r1*7+r2] = U2_outer[i0, r1] * U2_inner[j1, r2]... 
    # Wait, this isn't quite right either. Let me think again.
    
    # Actually, the cleanest way is to note that the 4x4 multiplication tensor
    # is the tensor product of the 2x2 multiplication tensor with itself.
    # T_4x4[a, b, c] = T_2x2[a0, b0, c0] * T_2x2[a1, b1, c1]
    # where a = a0*4 + a1 (a0 is block row, a1 is element within block... 
    # but this indexing is tricky.
    
    # Let me use a different approach: directly compute the Kronecker product
    # of the factor matrices with the correct index mapping.
    
    # For 4x4, index a in U is i*4+j where i,j in 0..3.
    # We can write i = 2*i0 + i1, j = 2*j0 + j1.
    # So a = (2*i0+i1)*4 + (2*j0+j1) = 8*i0 + 4*i1 + 2*j0 + j1
    
    # The 2x2 factor U2 has indices i2*2+j2 where i2,j2 in 0..1.
    # For the outer product, the "row" of the outer 2x2 matrix corresponds to i0,
    # and the "column" corresponds to j0.
    # The inner 2x2 matrix has row i1 and column j1.
    
    # So U[i*4+j, r1*7+r2] = U2[i0*2+j0, r1] * U2[i1*2+j1, r2]
    
    # Similarly for V:
    # V[j*4+k, r1*7+r2] where j = 2*j0+j1, k = 2*k0+k1
    # V[j*4+k, r1*7+r2] = V2[j0*2+k0, r1] * V2[j1*2+k1, r2]
    
    # For W:
    # W[k*4+i, r1*7+r2] where k = 2*k0+k1, i = 2*i0+i1
    # W[k*4+i, r1*7+r2] = W2[k0*2+i0, r1] * W2[k1*2+i1, r2]
    
    for i in range(4):
        for j in range(4):
            i0 = i // 2
            i1 = i % 2
            j0 = j // 2
            j1 = j % 2
            a = i * 4 + j
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    U[a, r] = U2[i0 * 2 + j0, r1] * U2[i1 * 2 + j1, r2]
    
    for j in range(4):
        for k in range(4):
            j0 = j // 2
            j1 = j % 2
            k0 = k // 2
            k1 = k % 2
            b = j * 4 + k
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    V[b, r] = V2[j0 * 2 + k0, r1] * V2[j1 * 2 + k1, r2]
    
    for k in range(4):
        for i in range(4):
            k0 = k // 2
            k1 = k % 2
            i0 = i // 2
            i1 = i % 2
            c = k * 4 + i
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    W[c, r] = W2[k0 * 2 + i0, r1] * W2[k1 * 2 + i1, r2]
    
    # Verify the decomposition
    U_np = U.astype(float)
    V_np = V.astype(float)
    W_np = W.astype(float)
    
    verified = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                if abs(val - 1.0) > 1e-10:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if not verified:
        # If verification fails, fall back to the recursive construction
        U = np.zeros((16, 49), dtype=float)
        V = np.zeros((16, 49), dtype=float)
        W = np.zeros((16, 49), dtype=float)
        
        def w_idx(i, k):
            return k * 4 + i
        
        def make_block(start_i, start_j, is_A=True):
            mat = []
            for di in range(2):
                row = []
                for dj in range(2):
                    row.append([(start_i + di, start_j + dj, 1)])
                mat.append(row)
            return mat
        
        def add_mat(X, Y):
            result = []
            for i in range(2):
                row = []
                for j in range(2):
                    d = {}
                    for (r, c, v) in X[i][j]:
                        d[(r,c)] = d.get((r,c), 0) + v
                    for (r, c, v) in Y[i][j]:
                        d[(r,c)] = d.get((r,c), 0) + v
                    row.append([(r, c, v) for (r,c), v in d.items() if v != 0])
                result.append(row)
            return result
        
        def sub_mat(X, Y):
            neg_Y = [[[(r, c, -v) for (r,c,v) in Y[i][j]] for j in range(2)] for i in range(2)]
            return add_mat(X, neg_Y)
        
        A00 = make_block(0, 0)
        A01 = make_block(0, 2)
        A10 = make_block(2, 0)
        A11 = make_block(2, 2)
        B00 = make_block(0, 0)
        B01 = make_block(0, 2)
        B10 = make_block(2, 0)
        B11 = make_block(2, 2)
        
        outer_X = [
            add_mat(A00, A11),
            add_mat(A10, A11),
            A00,
            A11,
            add_mat(A00, A01),
            sub_mat(A10, A00),
            sub_mat(A01, A11),
        ]
        outer_Y = [
            add_mat(B00, B11),
            B00,
            sub_mat(B01, B11),
            sub_mat(B10, B00),
            B11,
            add_mat(B00, B01),
            add_mat(B10, B11),
        ]
        
        c_targets = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (1,1,1)],
            [(0,0,1), (1,0,1)],
            [(0,0,-1), (0,1,1)],
            [(1,1,1)],
            [(0,0,1)],
        ]
        
        def inner_factors(X, Y):
            def single_entry(i, j, mat):
                result = [[[] for _ in range(2)] for _ in range(2)]
                result[i][j] = list(mat[i][j])
                return result
            
            def add_entries(i1, j1, i2, j2, mat):
                result = [[[] for _ in range(2)] for _ in range(2)]
                d = {}
                for (r, c, v) in mat[i1][j1]:
                    d[(r,c)] = d.get((r,c), 0) + v
                for (r, c, v) in mat[i2][j2]:
                    d[(r,c)] = d.get((r,c), 0) + v
                result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
                return result
            
            def sub_entries(i1, j1, i2, j2, mat):
                result = [[[] for _ in range(2)] for _ in range(2)]
                d = {}
                for (r, c, v) in mat[i1][j1]:
                    d[(r,c)] = d.get((r,c), 0) + v
                for (r, c, v) in mat[i2][j2]:
                    d[(r,c)] = d.get((r,c), 0) - v
                result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
                return result
            
            X1 = add_entries(0, 0, 1, 1, X)
            Y1 = add_entries(0, 0, 1, 1, Y)
            X2 = add_entries(1, 0, 1, 1, X)
            Y2 = single_entry(0, 0, Y)
            X3 = single_entry(0, 0, X)
            Y3 = sub_entries(0, 1, 1, 1, Y)
            X4 = single_entry(1, 1, X)
            Y4 = sub_entries(1, 0, 0, 0, Y)
            X5 = add_entries(0, 0, 0, 1, X)
            Y5 = single_entry(1, 1, Y)
            X6 = sub_entries(1, 0, 0, 0, X)
            Y6 = add_entries(0, 0, 0, 1, Y)
            X7 = sub_entries(0, 1, 1, 1, X)
            Y7 = add_entries(1, 0, 1, 1, Y)
            
            return [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7)]
        
        inner_targets = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (1,1,1)],
            [(0,0,1), (1,0,1)],
            [(0,0,-1), (0,1,1)],
            [(1,1,1)],
            [(0,0,1)],
        ]
        
        col = 0
        for t in range(7):
            X = outer_X[t]
            Y = outer_Y[t]
            factors = inner_factors(X, Y)
            for k in range(7):
                Xf, Yf = factors[k]
                for i in range(2):
                    for j in range(2):
                        for (a_r, a_c, a_v) in Xf[i][j]:
                            U[a_r * 4 + a_c, col] += a_v
                for i in range(2):
                    for j in range(2):
                        for (b_r, b_c, b_v) in Yf[i][j]:
                            V[b_r * 4 + b_c, col] += b_v
                for (br, bc, c_coeff) in c_targets[t]:
                    for (li, lj, z_coeff) in inner_targets[k]:
                        global_i = br * 2 + li
                        global_k = bc * 2 + lj
                        W[w_idx(global_i, global_k), col] += c_coeff * z_coeff
                col += 1
        
        U_np = U.astype(float)
        V_np = V.astype(float)
        W_np = W.astype(float)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 4 + i
                    val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                    if abs(val - 1.0) > 1e-10:
                        raise ValueError("Fallback verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
