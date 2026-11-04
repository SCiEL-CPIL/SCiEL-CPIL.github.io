import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    t0 = time.time()
    deadline = t0 + 110.0

    # Top-level Strassen: partition A(4x4) into 2x2 blocks of 2x2
    # Partition B(4x6) into 2x3 blocks of 2x2
    # C(4x6) = sum over J of A_IJ * B_JK
    # With Strassen, we have 7 combined products
    # Each product: (2x2 matrix) * (2x3 matrix)
    # For 2x2 * 2x3: use Strassen for first 2 cols (7) + schoolbook for 3rd col (4) = 11
    # Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((n * m, R))  # (16, 77)
    V = np.zeros((m * p, R))  # (24, 77)
    W = np.zeros((p * n, R))  # (24, 77)
    
    # Strassen coefficients for 2x2x2
    # A blocks indexed 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks indexed 0=B00, 1=B01, 2=B10, 3=B11
    alpha = [
        [1, 0, 0, 1],  # s=0: A00 + A11
        [0, 0, 1, 1],  # s=1: A10 + A11
        [1, 0, 0, 0],  # s=2: A00
        [0, 0, 0, 1],  # s=3: A11
        [1, 1, 0, 0],  # s=4: A00 + A01
        [-1, 0, 1, 0], # s=5: A10 - A00
        [0, 1, 0, -1]  # s=6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # s=0: B00 + B11
        [1, 0, 0, 0],  # s=1: B00
        [0, 1, 0, -1], # s=2: B01 - B11
        [-1, 0, 1, 0], # s=3: B10 - B00
        [0, 0, 0, 1],  # s=4: B11
        [1, 1, 0, 0],  # s=5: B00 + B01
        [0, 0, 1, 1]   # s=6: B10 + B11
    ]
    # Reconstruction: C = sum_s recon[s] * M_s
    # recon[s] is list of (i, k, coeff) for C[i,k]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0: C00 += M0, C11 += M0
        [(1, 0, 1), (1, 1, -1)],    # s=1: C10 += M1, C11 -= M1
        [(0, 1, 1), (1, 1, 1)],     # s=2: C01 += M2, C11 += M2
        [(0, 0, 1), (1, 0, 1)],     # s=3: C00 += M3, C10 += M3
        [(0, 0, -1), (0, 1, 1)],    # s=4: C00 -= M4, C01 += M4
        [(1, 1, 1)],                # s=5: C11 += M5
        [(0, 0, 1)]                 # s=6: C00 += M6
    ]
    
    col = 0
    for s in range(7):
        # Top-level Strassen term s
        # X_s = sum_r alpha[s][r] * A_r  (2x2 matrix)
        # Y_s = sum_c beta[s][c] * B_c  (2x3 matrix)
        # Need to compute X_s * Y_s and add to C with recon[s] coefficients
        
        # Part 1: Inner Strassen for X_s * Y_s[:,0:2] (2x2 * 2x2)
        for si in range(7):
            # Inner Strassen term si
            # inner_X = sum_a alpha[si][a] * X_s[a//2, a%2]
            # inner_Y = sum_b beta[si][b] * Y_s[b//2, b%2]
            # inner_C = sum_{(ii,kk,coeff)} coeff * inner_C[ii,kk]
            
            # U part: contributes to U[i*m+j, col]
            # X_s[ar, br] = sum_r alpha[s][r] * A_r[ar, br]
            # inner_X[x_row, x_col] = sum_a alpha[si][a] * X_s[a//2, a%2]
            # So the coefficient for A_r[x_row, x_col] is alpha[si][a] * alpha[s][r]
            # where a indexes position in X_s: a = x_row*2 + x_col
            # and r indexes A block: r = ar*2 + br where ar=x_row, br=x_col
            
            for a in range(4):
                a_coeff = alpha[si][a]
                if a_coeff == 0:
                    continue
                x_row = a // 2
                x_col = a % 2
                for r in range(4):
                    r_coeff = alpha[s][r]
                    if r_coeff == 0:
                        continue
                    # A_r is block at row r//2, col r%2
                    # A_r[x_row, x_col] corresponds to global A[(r//2)*2+x_row, (r%2)*2+x_col]
                    global_i = (r // 2) * 2 + x_row
                    global_j = (r % 2) * 2 + x_col
                    u_idx = global_i * m + global_j
                    U[u_idx, col] += a_coeff * r_coeff
            
            # V part: contributes to V[j*p+k, col]
            # Y_s[y_row, y_col] = sum_c beta[s][c] * B_c[y_row, y_col]
            # For inner Strassen, we use Y_s[:, 0:2], so y_col in {0, 1}
            # inner_Y[y_row, y_inner_col] = sum_b beta[si][b] * Y_s[b//2, b%2]
            # where b indexes position in Y_s[:,0:2]: b = y_row*2 + y_inner_col
            # So coefficient for B_c[y_row, y_inner_col] is beta[si][b] * beta[s][c]
            # where c indexes B block: c = ac*2 + bc where ac=y_row, bc=y_inner_col
            
            for b in range(4):
                b_coeff = beta[si][b]
                if b_coeff == 0:
                    continue
                y_row = b // 2
                y_inner_col = b % 2
                for c in range(4):
                    c_coeff = beta[s][c]
                    if c_coeff == 0:
                        continue
                    # B_c is block at row c//2, col c%2
                    # B_c[y_row, y_inner_col] corresponds to global B[(c//2)*2+y_row, (c%2)*2+y_inner_col]
                    # But wait: B is 4x6, partitioned into 2x3 blocks of 2x2
                    # Block c has row block c//2 (0 or 1), col block c%2 (0,1,2)
                    # B_c[y_row, y_inner_col] -> global B[(c//2)*2+y_row, (c%2)*2+y_inner_col]
                    global_j = (c // 2) * 2 + y_row
                    global_k = (c % 2) * 2 + y_inner_col
                    v_idx = global_j * p + global_k
                    V[v_idx, col] += b_coeff * c_coeff
            
            # W part: contributes to W[k*n+i, col]
            # inner_C[ii, kk] = inner_X * inner_Y result
            # Top-level recon[s] says C[si_row, si_col] += coeff * M_s[si_row, si_col]
            # Inner recon[si] says inner_C[inner_ii, inner_kk] += inner_coeff * M_si
            # So C[si_row + inner_ii, si_col + inner_kk] += coeff * inner_coeff * M_si
            
            for (si_row, si_col, si_coeff) in recon[s]:
                for (inner_ii, inner_kk, inner_coeff) in recon[si]:
                    global_i = si_row * 2 + inner_ii
                    global_k = si_col * 2 + inner_kk
                    w_idx = global_k * n + global_i
                    W[w_idx, col] += si_coeff * inner_coeff
            
            col += 1
        
        # Part 2: Schoolbook for X_s * Y_s[:,2] (2x2 * 2x1)
        # X_s[i,j] * Y_s[j, 2] for i=0,1; j=0,1
        # 4 scalar multiplications
        for i in range(2):
            for j in range(2):
                # U part: X_s[i,j] = sum_r alpha[s][r] * A_r[i,j]
                # A_r[i,j] -> global A[(r//2)*2+i, (r%2)*2+j]
                for r in range(4):
                    r_coeff = alpha[s][r]
                    if r_coeff == 0:
                        continue
                    global_i = (r // 2) * 2 + i
                    global_j = (r % 2) * 2 + j
                    u_idx = global_i * m + global_j
                    U[u_idx, col] += r_coeff
                
                # V part: Y_s[j, 2] = sum_c beta[s][c] * B_c[j, 2]
                # But B_c is 2x2, so B_c[j, 2] doesn't exist!
                # Wait: Y_s is 2x3. Y_s[j, 2] is the 3rd column.
                # B is partitioned into 2x3 blocks. Block c has col index c%2 (0,1,2).
                # Y_s[j, 2] = sum_c beta[s][c] * B_c[j, 0] if c%2==2? No...
                
                # Let me reconsider. B is 4x6. Partition into 2 row-blocks and 3 col-blocks.
                # Block (r_b, c_b) is 2x2, located at rows [2*r_b, 2*r_b+2), cols [2*c_b, 2*c_b+2).
                # B has 2*3=6 blocks. Index c in beta: c = r_b*3 + c_b? Or c = r_b*2 + c_b?
                
                # In the Strassen setup for 2x2 * 2x3:
                # X is 2x2, Y is 2x3.
                # Y's columns: col 0, col 1, col 2.
                # For inner Strassen on cols 0,1: we treat Y[:,0:2] as 2x2.
                # For schoolbook on col 2: we compute X * Y[:,2].
                
                # How does Y_s relate to B blocks?
                # Top-level: B is 4x6, partitioned into 2x3 blocks of 2x2.
                # B block (I, K) is at rows [2*I, 2*I+2), cols [2*K, 2*K+2).
                # There are 2*3 = 6 blocks.
                # In beta, we index 4 B blocks for the 2x2 Strassen.
                # But B has 6 blocks! So beta should have 6 entries?
                
                # I think I need to reconsider the top-level partition.
                # Actually, for top-level Strassen on 4x4 * 4x6:
                # A is 4x4 -> 2x2 blocks of 2x2 (4 blocks)
                # B is 4x6 -> 2x3 blocks of 2x2 (6 blocks)
                # But Strassen requires square blocks. So we can't directly apply Strassen.
                
                # Alternative: partition B into 2x2 blocks of 2x3? No, that's not standard.
                
                # Let me reconsider. The standard approach for rectangular:
                # A(4x4) * B(4x6) = C(4x6)
                # Partition A into 2x2 blocks of 2x2: A_00, A_01, A_10, A_11
                # Partition B into 2x2 blocks of 2x3: B_0, B_1 (each 2x3)
                # Or partition B into 2x3 blocks of 2x2: B_00, B_01, B_02, B_10, B_11, B_12
                # C = A_00*B_0 + A_01*B_1 + A_10*B_0 + A_11*B_1 where B_0 = [B_00 B_01 B_02], B_1 = [B_10 B_11 B_12]
                # Each term is 2x2 * 2x6? No, 2x2 * 2x3? Wait...
                
                # If B_0 is 2x3 (rows 0-1, cols 0-5) and B_1 is 2x3 (rows 2-3, cols 0-5),
                # then A_00*B_0 is 2x3, etc.
                # So we have 4 products of 2x2 * 2x3.
                # Can we use Strassen on these 4 products? Strassen is for 2x2 * 2x2.
                # For 2x2 * 2x3, we'd need a generalized Strassen.
                
                # The inspiration's approach: treat B as having 4 blocks for Strassen purposes?
                # That doesn't work since B has 6 blocks.
                
                # Let me re-examine the inspiration code more carefully.
                
                # Looking at inspiration 1:
                # beta has 4 entries, indexed 0-3.
                # v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                # So ac = c//2, bc = c%2, and the global col is bc*3 + y_inner_col
                # This means B blocks are indexed as: block (ac, bc) is at rows [2*ac, 2*ac+2), cols [3*bc, 3*bc+3)
                # So B is partitioned into 2 row-blocks and 2 col-blocks, each col-block being 3 wide!
                # B_00: rows 0-1, cols 0-2 (2x3)
                # B_01: rows 0-1, cols 3-5 (2x3)
                # B_10: rows 2-3, cols 0-2 (2x3)
                # B_11: rows 2-3, cols 3-5 (2x3)
                # That's 4 blocks of 2x3. Now Strassen works on 2x2 * 2x3!
                
                # So the partition is:
                # A: 2x2 blocks of 2x2
                # B: 2x2 blocks of 2x3
                # C: 2x2 blocks of 2x3
                
                # For each Strassen term s:
                # X_s = sum_r alpha[s][r] * A_r (2x2)
                # Y_s = sum_c beta[s][c] * B_c (2x3)
                # C_s = X_s * Y_s (2x3)
                # C += sum_s recon[s] * C_s
                
                # For computing X_s * Y_s (2x2 * 2x3):
                # Partition Y_s into Y_s[:,0:2] (2x2) and Y_s[:,2] (2x1)
                # Use Strassen for X_s * Y_s[:,0:2] (7 terms)
                # Use schoolbook for X_s * Y_s[:,2] (4 terms)
                
                # This makes sense! Let me redo the V part.
                
                # Y_s[j, k] where j in {0,1}, k in {0,1,2}
                # Y_s = sum_c beta[s][c] * B_c
                # B_c is 2x3, located at rows [2*(c//2), 2*(c//2)+2), cols [3*(c%2), 3*(c%2)+3)
                # B_c[local_j, local_k] -> global B[2*(c//2)+local_j, 3*(c%2)+local_k]
                
                # For schoolbook on Y_s[:,2]:
                # Y_s[j, 2] = sum_c beta[s][c] * B_c[j, 2]
                # B_c[j, 2] -> global B[2*(c//2)+j, 3*(c%2)+2]
                
                # U part for schoolbook:
                # X_s[i, j] = sum_r alpha[s][r] * A_r[i, j]
                # A_r is 2x2 at rows [2*(r//2), 2*(r//2)+2), cols [2*(r%2), 2*(r%2)+2)
                # A_r[i, j] -> global A[2*(r//2)+i, 2*(r%2)+j]
                
                # W part for schoolbook:
                # C_s[i, 2] = X_s[i, 0]*Y_s[0, 2] + X_s[i, 1]*Y_s[1, 2]
                # But in tensor form, for each (i, j) pair in schoolbook:
                # The contribution to C[si_row+i, si_col+2] from X_s[i,j]*Y_s[j,2]
                # recon[s] gives (si_row, si_col, coeff)
                # So W[(si_col+2)*n + (si_row+i), col] += coeff * 1 (for this schoolbook term)
                
                # Wait, I need to be more careful. The schoolbook term for (i,j) computes:
                # partial = X_s[i, j] * Y_s[j, 2]
                # This contributes to C_s[i, 2]
                # Then recon[s] maps C_s to C.
                
                # For the tensor decomposition, each rank-1 term corresponds to one scalar multiplication.
                # For schoolbook (i,j): U has the X_s[i,j] part, V has the Y_s[j,2] part, W has the recon part.
                
                # U[i*m+j, col] += sum_r alpha[s][r] * (coefficient for A_r[i,j])
                # But X_s[i,j] = sum_r alpha[s][r] * A_r[i,j], so the coefficient is just alpha[s][r].
                
                # Actually, I realize I need to restructure. Let me start fresh with the correct partition.
                
                pass  # This is getting complicated, let me rewrite the whole thing
    
    # Let me restart with a cleaner approach
    if time.time() > deadline - 10:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Correct approach:
    # A(4x4) partitioned into 2x2 blocks of 2x2: A_00, A_01, A_10, A_11
    # B(4x6) partitioned into 2x2 blocks of 2x3: B_00, B_01, B_10, B_11
    #   B_00: rows 0-1, cols 0-2
    #   B_01: rows 0-1, cols 3-5
    #   B_10: rows 2-3, cols 0-2
    #   B_11: rows 2-3, cols 3-5
    # C(4x6) partitioned into 2x2 blocks of 2x3: C_00, C_01, C_10, C_11
    #   C_IK = sum_J A_IJ * B_JK where A_IJ is 2x2, B_JK is 2x3, C_IK is 2x3
    
    # Strassen on the 2x2 block structure:
    # 7 products, each is (2x2) * (2x3)
    # For each (2x2)*(2x3): use Strassen on first 2 cols of the 2x3 (7 terms) + schoolbook on 3rd col (4 terms) = 11
    # Total: 7 * 11 = 77
    
    R = 77
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for X_s * Y_s[:,0:2]
        for si in range(7):
            # U: X_s[i,j] combined with inner Strassen
            for a in range(4):
                a_coeff = alpha[si][a]
                if a_coeff == 0:
                    continue
                x_row = a // 2
                x_col = a % 2
                for r in range(4):
                    r_coeff = alpha[s][r]
                    if r_coeff == 0:
                        continue
                    global_i = (r // 2) * 2 + x_row
                    global_j = (r % 2) * 2 + x_col
                    u_idx = global_i * m + global_j
                    U[u_idx, col] += a_coeff * r_coeff
            
            # V: Y_s[j,k] for k in {0,1} combined with inner Strassen
            for b in range(4):
                b_coeff = beta[si][b]
                if b_coeff == 0:
                    continue
                y_row = b // 2
                y_inner_col = b % 2
                for c in range(4):
                    c_coeff = beta[s][c]
                    if c_coeff == 0:
                        continue
                    global_j = (c // 2) * 2 + y_row
                    global_k = (c % 2) * 3 + y_inner_col
                    v_idx = global_j * p + global_k
                    V[v_idx, col] += b_coeff * c_coeff
            
            # W: reconstruction
            for (si_row, si_col, si_coeff) in recon[s]:
                for (inner_ii, inner_kk, inner_coeff) in recon[si]:
                    global_i = si_row * 2 + inner_ii
                    global_k = si_col * 3 + inner_kk
                    w_idx = global_k * n + global_i
                    W[w_idx, col] += si_coeff * inner_coeff
            
            col += 1
        
        # Part 2: Schoolbook for X_s * Y_s[:,2]
        for i in range(2):
            for j in range(2):
                # U: X_s[i,j]
                for r in range(4):
                    r_coeff = alpha[s][r]
                    if r_coeff == 0:
                        continue
                    global_i = (r // 2) * 2 + i
                    global_j = (r % 2) * 2 + j
                    u_idx = global_i * m + global_j
                    U[u_idx, col] += r_coeff
                
                # V: Y_s[j,2]
                for c in range(4):
                    c_coeff = beta[s][c]
                    if c_coeff == 0:
                        continue
                    global_j = (c // 2) * 2 + j
                    global_k = (c % 2) * 3 + 2
                    v_idx = global_j * p + global_k
                    V[v_idx, col] += c_coeff
                
                # W: reconstruction for C_s[i,2]
                for (si_row, si_col, si_coeff) in recon[s]:
                    global_i = si_row * 2 + i
                    global_k = si_col * 3 + 2
                    w_idx = global_k * n + global_i
                    W[w_idx, col] += si_coeff
                
                col += 1
    
    # Verify
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, R)
    Vr = Va.reshape(m, p, R)
    Wr = Wa.reshape(p, n, R)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    
    if err > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists with integer values where possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                rounded = round(x)
                if abs(x - rounded) < 1e-9:
                    new_row.append(int(rounded))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
