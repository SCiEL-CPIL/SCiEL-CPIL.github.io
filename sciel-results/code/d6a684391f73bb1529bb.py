"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_MARGIN = 10.0  # seconds before the 150s kill to stop

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7) with W[k*2+i, c] convention."""
    # Strassen's 7 products for 2x2 matrices.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # Result: [[p1+p4+p5-p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    # U: rows indexed by (i,j) for A[i,j], i in 0..1, j in 0..1
    # V: rows indexed by (j,k) for B[j,k], j in 0..1, k in 0..1
    # W: rows indexed by (k,i) for C[k,i], k in 0..1, i in 0..1
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # A indices: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    # B indices: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    # C indices: C[0,0]=k0i0, C[0,1]=k0i1, C[1,0]=k1i0, C[1,1]=k1i1
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # p2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C[1,0]
    
    # p3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C[0,1]
    
    # p4 = d(g-e)
    U[3, 3] = 1  # d
    V[0, 3] = -1 # e
    V[2, 3] = 1  # g
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # p5 = (a+b)h
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    V[3, 4] = 1  # h
    W[0, 4] = 1  # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # p6 = (c-a)(e+f)
    U[0, 5] = -1 # a
    U[2, 5] = 1  # c
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[2, 5] = 1  # C[1,0]
    
    # p7 = (b-d)(g+h)
    U[1, 6] = 1  # b
    U[3, 6] = -1 # d
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = -1 # C[0,0]
    W[3, 6] = 1  # C[1,1]
    
    return U, V, W

def _strassen_2x2x3():
    """Return (U, V, W) for 2x2 * 2x3 -> 2x3 with rank 7 (using Strassen trick).
    U shape (4, 7), V shape (6, 7), W shape (6, 7) with W[k*2+i, c] convention.
    Here A is 2x2, B is 2x3, C is 2x3.
    W index: k in 0..2, i in 0..1, so W[k*2+i, c]"""
    # We can use the 2x2x2 Strassen for the first two columns of B,
    # and a simpler decomposition for the third column.
    # Actually, let's think differently.
    # 2x2 * 2x3: A is 2x2, B is 2x3.
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[0,2] = a*i + b*j
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    # C[1,2] = c*i + d*j
    
    # This is equivalent to 3 independent 2x1 * 2x1 products (column-wise).
    # Each 2x1 * 2x1 -> 2x1 has rank 2 (schoolbook) or can we do better?
    # Actually 2x1 * 2x1: C[0] = a*e + b*g, C[1] = c*e + d*g. Rank is 2.
    # So 3 columns * rank 2 = rank 6? No, that's if they're independent.
    # But we can share some terms.
    
    # Alternative: treat as 2x2 * 2x2 (for cols 0,1) + 2x2 * 2x1 (for col 2).
    # 2x2*2x2 uses rank 7, 2x2*2x1 uses rank 2. Total rank 9? That's worse than schoolbook (2*2*3=12).
    
    # Let me just use schoolbook for 2x2x3: rank = 2*2*3 = 12.
    # Or better: use the fact that it's 3 copies of 2x2x1.
    # 2x2 * 2x1 -> 2x1: rank 2. Three of them: rank 6.
    # Wait, 2x2 * 2x1: A is 2x2, B is 2x1. C is 2x1.
    # C[0] = a*e + b*f, C[1] = c*e + d*f.
    # This is a bilinear form in 4+2=6 variables with 2 outputs. Rank is at most 2.
    # Indeed: C[0] = a*e + b*f, C[1] = c*e + d*f.
    # We can write: C[0] = 1*e*a + 1*f*b, C[1] = 1*e*c + 1*f*d.
    # So rank 2: two rank-1 terms.
    # Three columns: rank 6.
    
    # Actually, can we do better than 6 for 2x2*2x3?
    # The tensor is 2x2x3. Bilinear complexity... 
    # Let's just use rank 6 (schoolbook for the 2x2x1 parts).
    
    U = np.zeros((4, 6), dtype=np.int64)
    V = np.zeros((6, 6), dtype=np.int64)
    W = np.zeros((6, 6), dtype=np.int64)
    
    # Column 0 of B: e=(0,0), g=(1,0)
    # C[0,0] = a*e + b*g -> W[0*2+0, c] = W[0, c]
    # C[1,0] = c*e + d*g -> W[1*2+0, c] = W[2, c]
    U[0, 0] = 1  # a
    V[0, 0] = 1  # e
    W[0, 0] = 1  # C[0,0]
    
    U[1, 1] = 1  # b
    V[2, 1] = 1  # g
    W[0, 1] = 1  # C[0,0]
    
    U[2, 2] = 1  # c
    V[0, 2] = 1  # e
    W[2, 2] = 1  # C[1,0]
    
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    W[2, 3] = 1  # C[1,0]
    
    # Column 1 of B: f=(0,1), h=(1,1)
    # C[0,1] = a*f + b*h -> W[0*2+1, c] = W[1, c]
    # C[1,1] = c*f + d*h -> W[1*2+1, c] = W[3, c]
    U[0, 4] = 1  # a
    V[1, 4] = 1  # f
    W[1, 4] = 1  # C[0,1]
    
    U[1, 5] = 1  # b
    V[3, 5] = 1  # h
    W[1, 5] = 1  # C[0,1]
    
    U[2, 0] = 1  # c
    V[1, 0] = 1  # f
    W[3, 0] = 1  # C[1,1]
    
    U[3, 1] = 1  # d
    V[3, 1] = 1  # h
    W[3, 1] = 1  # C[1,1]
    
    # Column 2 of B: i=(0,2), j=(1,2)
    # C[0,2] = a*i + b*j -> W[2*2+0, c] = W[4, c]
    # C[1,2] = c*i + d*j -> W[2*2+1, c] = W[5, c]
    U[0, 2] = 1  # a
    V[4, 2] = 1  # i
    W[4, 2] = 1  # C[0,2]
    
    U[1, 3] = 1  # b
    V[5, 3] = 1  # j
    W[4, 3] = 1  # C[0,2]
    
    U[2, 4] = 1  # c
    V[4, 4] = 1  # i
    W[5, 4] = 1  # C[1,2]
    
    U[3, 5] = 1  # d
    V[5, 5] = 1  # j
    W[5, 5] = 1  # C[1,2]
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Split the 4x4 * 4x6 into blocks.
    # A is 4x4, B is 4x6, C is 4x6.
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B: B = [[B00, B01, B02], [B10, B11, B12]] where each Bi is 2x3
    # C = [[C00, C01, C02], [C10, C11, C12]] where each Cij is 2x3
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Each product is 2x2 * 2x3 -> 2x3.
    # We have 12 such products.
    # If each 2x2*2x3 has rank 6, total rank = 12*6 = 72.
    # But we can potentially share terms between the products.
    
    # Alternative: use Strassen for the 2x2*2x2 parts.
    # Split B differently: B = [B_left (4x2), B_right (4x4)]
    # Actually, let's try: split B into 3 blocks of 2 columns each.
    # B = [B0 (4x2), B1 (4x2), B2 (4x2)]
    # C = [C0 (4x2), C1 (4x2), C2 (4x2)]
    # C0 = A*B0, C1 = A*B1, C2 = A*B2
    
    # For each Ci = A*Bi, where A is 4x4 and Bi is 4x2.
    # This is 4x4 * 4x2 -> 4x2.
    # Split A into 2x2 blocks, Bi into 2x2 blocks.
    # Ci = [[A00*Bi00 + A01*Bi10], [A10*Bi00 + A11*Bi10]]
    # Each 2x2 * 2x2 -> 2x2 product can use Strassen rank 7.
    # For each Ci, we have 4 such products, each rank 7, total rank 28 per Ci.
    # 3 Ci's: total rank 84.
    
    # This matches the "84" from the knowledge. Let's implement this.
    
    # For each column block b (0,1,2):
    #   For each row block i (0,1) and col block j (0,1) of the 4x2 result:
    #     Product: A[i*2:(i+1)*2, :] * B[:, b*2:(b+1)*2]
    #     This is 4x4 * 4x2.
    #     Split into 2x2 blocks:
    #       A = [[A00, A01], [A10, A11]] (each 2x2)
    #       B_block = [[B00, B10]] (each 2x2, since B_block is 4x2)
    #     C_block = [[A00*B00 + A01*B10], [A10*B00 + A11*B10]]
    #     4 Strassen products per column block.
    
    # Total: 3 column blocks * 4 Strassen products * rank 7 = 84.
    
    all_U = []
    all_V = []
    all_W = []
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for b in range(3):  # column block of B
        for i in range(2):  # row block of A and C
            for j in range(2):  # col block of A (and row block of B_block)
                # This product: A[i*2:(i+1)*2, j*2:(j+1)*2] * B[j*2:(j+1)*2, b*2:(b+1)*2]
                # contributes to C[i*2:(i+1)*2, b*2:(b+1)*2]
                
                # U: 2x2 block of A, indices (i,j)
                # V: 2x2 block of B, indices (j, b)
                # W: 2x2 block of C, indices (b, i)
                
                # U rows: A has 4 rows, 4 cols. Block (i,j) is rows i*2..i*2+1, cols j*2..j*2+1.
                # U row index: row_in_block * 2 + col_in_block, where row_in_block is 0..1, col_in_block is 0..1.
                # But we need to map to the global U which has 4*4=16 rows.
                # Actually, U has shape (n*m, rank) = (16, rank).
                # U row index = i_A * m + j_A = i_A * 4 + j_A.
                # For block (i,j), i_A is in {i*2, i*2+1}, j_A is in {j*2, j*2+1}.
                
                for c in range(7):
                    u_row = U_s[:, c]
                    v_row = V_s[:, c]
                    w_row = W_s[:, c]
                    
                    # Build sparse U, V, W contributions
                    for r in range(4):
                        if u_row[r] != 0:
                            i_A = i * 2 + r // 2
                            j_A = j * 2 + r % 2
                            all_U.append((i_A * 4 + j_A, c, u_row[r]))
                        
                        if v_row[r] != 0:
                            j_B = j * 2 + r // 2
                            k_B = b * 2 + r % 2
                            all_V.append((j_B * 6 + k_B, c, v_row[r]))
                        
                        if w_row[r] != 0:
                            k_C = b * 2 + r // 2
                            i_C = i * 2 + r % 2
                            all_W.append((k_C * 4 + i_C, c, w_row[r]))
    
    # Now we have 84 rank-1 terms. Let's assemble them.
    rank = 84
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((24, rank), dtype=np.int64)
    W = np.zeros((24, rank), dtype=np.int64)
    
    col = 0
    for b in range(3):
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    u_row = U_s[:, c]
                    v_row = V_s[:, c]
                    w_row = W_s[:, c]
                    
                    for r in range(4):
                        if u_row[r] != 0:
                            i_A = i * 2 + r // 2
                            j_A = j * 2 + r % 2
                            U[i_A * 4 + j_A, col] = u_row[r]
                        
                        if v_row[r] != 0:
                            j_B = j * 2 + r // 2
                            k_B = b * 2 + r % 2
                            V[j_B * 6 + k_B, col] = v_row[r]
                        
                        if w_row[r] != 0:
                            k_C = b * 2 + r // 2
                            i_C = i * 2 + r % 2
                            W[k_C * 4 + i_C, col] = w_row[r]
                    
                    col += 1
    
    # Verify correctness with a random test
    A = np.random.randint(-5, 5, (4, 4))
    B = np.random.randint(-5, 5, (4, 6))
    C_expected = A @ B
    
    C_computed = np.zeros((4, 6), dtype=np.int64)
    for c in range(rank):
        u_vec = U[:, c]
        v_vec = V[:, c]
        w_vec = W[:, c]
        prod = np.dot(u_vec, v_vec)
        C_computed += prod * w_vec.reshape(4, 6).T  # W is (p*n, rank), so w_vec is length 24
        # Wait, W has shape (p*n, rank) = (24, rank). w_vec is length 24.
        # C[k, i] = sum_c U[:,c] * V[:,c] * W[k*n+i, c]
        # So C_computed[k, i] += prod * W[k*4+i, c]
    
    # Let me redo the computation properly
    C_computed = np.zeros((4, 6), dtype=np.int64)
    for c in range(rank):
        prod = np.dot(U[:, c], V[:, c])
        for k in range(6):
            for i in range(4):
                C_computed[k, i] += prod * W[k * 4 + i, c]
    
    # Check if C_computed matches C_expected
    if not np.allclose(C_computed, C_expected):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
