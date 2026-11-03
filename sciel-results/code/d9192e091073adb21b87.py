"""Matrix multiplication tensor decomposition search for 4x4 * 4x5 using Strassen block method."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a00+a11)*(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00*(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11*(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5 = (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _build_strassen_4x4_4x4():
    """Build Strassen-based decomposition for 4x4 * 4x4 -> 4x4 with rank 49.
    U (16x49), V (16x49), W (16x49)
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen at the 2x2 block level:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12], [B21, B22]] (each 2x2)
    # C = [[C11, C12], [C21, C22]] (each 2x2)
    #
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Each Mi is a 2x2 * 2x2 -> 2x2 multiplication, use Strassen (rank 7)
    sU, sV, sW = _strassen_2x2()
    
    # Block indices: A11 rows 0-3, A12 rows 4-7, A21 rows 8-11, A22 rows 12-15
    # B11 cols 0-3, B12 cols 4-7, B21 cols 8-11, B22 cols 12-15
    # C11 rows 0-3, C12 rows 4-7, C21 rows 8-11, C22 rows 12-15
    
    # For each of the 7 Strassen multiplications at block level:
    # Mi = Xi * Yi where Xi and Yi are 2x2 matrices (linear combos of blocks)
    # Each Mi contributes 7 rank-1 terms to the 4x4x4 tensor
    
    # Define the 7 block-level multiplications:
    # Each is (linear combo of A blocks) * (linear combo of B blocks)
    # contributing to (linear combo of C blocks)
    
    # M1: (A11+A22) * (B11+B22) -> contributes to C11 (+1), C22 (+1)
    # M2: (A21+A22) * B11 -> contributes to C21 (+1), C22 (-1)
    # M3: A11 * (B12-B22) -> contributes to C12 (+1), C22 (+1)
    # M4: A22 * (B21-B11) -> contributes to C21 (+1), C11 (+1)
    # M5: (A11+A12) * B22 -> contributes to C11 (-1), C22 (+1)
    # M6: (A21-A11) * (B11+B12) -> contributes to C21 (+1), C22 (+1)
    # M7: (A12-A22) * (B21+B22) -> contributes to C11 (-1), C22 (+1)
    
    # For each Mi, we need to express Xi * Yi as a 2x2*2x2->2x2 multiplication
    # using Strassen, and map the local indices to global indices.
    
    # Block A indices (row blocks): A11=0, A12=1, A21=2, A22=3 (each 2x2, 4 entries)
    # Block B indices (col blocks): B11=0, B12=1, B21=2, B22=3
    # Block C indices: C11=0, C12=1, C21=2, C22=3
    
    # For Mi: Xi is a 2x2 matrix made from A blocks, Yi from B blocks
    # We need to compute Xi * Yi (2x2 * 2x2) and add to C blocks.
    
    # Let's define for each Mi:
    # a_coeffs: dict mapping A-block-index -> coefficient
    # b_coeffs: dict mapping B-block-index -> coefficient  
    # c_coeffs: dict mapping C-block-index -> coefficient
    
    mults = [
        # (a_coeffs, b_coeffs, c_coeffs)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),    # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),          # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),          # M3
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),          # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 3:1}),          # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:1}),     # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1}),    # M7
    ]
    
    for mi, (a_c, b_c, c_c) in enumerate(mults):
        # For this Mi, we compute Xi * Yi where Xi is 2x2, Yi is 2x2
        # Xi[i][j] = sum over A-blocks with coefficient: but actually Xi is a single 2x2 matrix
        # formed by linear combination of A blocks.
        # 
        # Actually, Xi is a 2x2 matrix. If a_c = {0:1, 3:1}, then Xi = A11 + A22 (element-wise)
        # So Xi[i][j] = sum_{ab} a_c[ab] * A[ab][i][j] where ab is block index
        #
        # Similarly Yi[i][j] = sum_{bb} b_c[bb] * B[bb][i][j]
        #
        # Result Z[i][j] = sum_k Xi[i][k] * Yi[k][j]
        # This contributes to C blocks: Z[i][j] adds c_c[cb] * Z[i][j] to C[cb][i][j]
        
        # Use Strassen for Xi * Yi -> Z (2x2 * 2x2 -> 2x2)
        # Strassen gives: Z = sum_{s=0}^{6} sU[x][s] * sV[y][s] * sW[z][s]
        # where x,y,z are local 2x2 indices (0-3)
        
        for s in range(7):
            col = mi * 7 + s
            # For each local entry in Xi (2x2, indices 0-3):
            for xi in range(4):  # xi = 2*row + col within the 2x2
                xr = xi // 2
                xc = xi % 2
                u_val = sU[xi][s]
                if u_val == 0:
                    continue
                # Xi[xr][xc] = sum_{ab} a_c[ab] * A[ab*4 + xr*2 + xc]
                for ab, coeff in a_c.items():
                    global_u = ab * 4 + xr * 2 + xc
                    U[global_u][col] += u_val * coeff
            
            for yi in range(4):
                yr = yi // 2
                yc = yi % 2
                v_val = sV[yi][s]
                if v_val == 0:
                    continue
                # Yi[yr][yc] = sum_{bb} b_c[bb] * B[bb*4 + yr*2 + yc]
                for bb, coeff in b_c.items():
                    global_v = bb * 4 + yr * 2 + yc
                    V[global_v][col] += v_val * coeff
            
            for zi in range(4):
                zr = zi // 2
                zc = zi % 2
                w_val = sW[zi][s]
                if w_val == 0:
                    continue
                # Z[zr][zc] contributes to C[cb][zr][zc] with coefficient c_c[cb]
                for cb, coeff in c_c.items():
                    global_w = cb * 4 + zr * 2 + zc
                    W[global_w][col] += w_val * coeff
    
    return U, V, W

def _build_extra_column():
    """Build schoolbook decomposition for 4x4 * 4x1 -> 4x1 with rank 16.
    U (16x16), V (4x16), W (4x16)
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c] where k=0 (only 1 column)
    """
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 140
    
    # Build Strassen 4x4x4 part (rank 49)
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    
    # Build extra column part (rank 16)
    U_e, V_e, W_e = _build_extra_column()
    
    # Combine: total rank = 49 + 16 = 65
    # The Strassen part handles columns 0-3 of B (4x4 * 4x4)
    # The extra part handles column 4 of B (4x4 * 4x1)
    
    total_rank = 49 + 16
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Copy Strassen part (columns 0-3 of B, i.e., k=0..3)
    for i in range(n*m):
        for c in range(49):
            U[i][c] = U_s[i][c]
    
    for j in range(m):
        for k in range(4):  # only first 4 columns
            for c in range(49):
                V[j*p+k][c] = V_s[j*4+k][c]
    
    for k in range(4):  # only first 4 columns
        for i in range(n):
            for c in range(49):
                W[k*n+i][c] = W_s[k*4+i][c]
    
    # Copy extra column part (column 4 of B, i.e., k=4)
    for i in range(n*m):
        for c in range(16):
            U[i][49+c] = U_e[i][c]
    
    for j in range(m):
        for c in range(16):
            V[j*p+4][49+c] = V_e[j*1+0][c]  # V_e has shape (4, 16), j*p+k = j*5+4
    
    for i in range(n):
        for c in range(16):
            W[4*n+i][49+c] = W_e[0*n+i][c]  # W_e has shape (4, 16), k*n+i = 4+i
    
    # Verify
    if not _verify(U, V, W, n, m, p, total_rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
