"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2(U, V, W, base_col, n, m, p):
    """Add Strassen 7-term decomposition for a 2x2 x 2x2 block.
    U is a list of rows (each row is a list), we modify in place.
    V, W similarly.
    base_col: starting column index for the new terms.
    Returns base_col + 7.
    """
    # Strassen's algorithm for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The block A occupies U rows [0,1] x cols [0,1] relative to block start
    # The block B occupies V rows [0,1] x cols [0,1] relative to block start
    # The block C occupies W rows [0,1] x cols [0,1] relative to block start
    
    # We'll add 7 columns for the 7 Strassen products
    col = base_col
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is U[0][0], A22 is U[1][1]
    # B11 is V[0][0], B22 is V[1][1]
    # C11 gets +M1, C22 gets +M1
    U[0][0] = U[0][0]  # placeholder, we need to be careful
    # Actually, let me think about this differently.
    # The U matrix for this sub-block has 4 entries (2x2), V has 4, W has 4.
    # For Strassen, we need to express the 8 products as 7 rank-1 terms.
    
    # Let me define the sub-block indices:
    # U_sub[i][j] for i in 0..1, j in 0..1 -> global U row offset + i, col offset + j
    # V_sub[i][j] for i in 0..1, j in 0..1 -> global V row offset + i, col offset + j  
    # W_sub[i][j] for i in 0..1, j in 0..1 -> global W row offset + i, col offset + j
    
    # Strassen terms:
    # Term 1: (A11+A22) * (B11+B22)
    #   U: A11 and A22 get coefficient 1
    #   V: B11 and B22 get coefficient 1
    #   W: C11 and C22 get coefficient 1
    U[0][0] += 1
    U[1][1] += 1
    V[0][0] += 1
    V[1][1] += 1
    W[0][0] += 1
    W[1][1] += 1
    col += 1
    
    # Term 2: (A21+A22) * B11
    #   U: A21 and A22 get 1
    #   V: B11 gets 1
    #   W: C21 gets 1
    U[1][0] += 1
    U[1][1] += 1
    V[0][0] += 1
    W[1][0] += 1
    col += 1
    
    # Term 3: A11 * (B12-B22)
    #   U: A11 gets 1
    #   V: B12 gets 1, B22 gets -1
    #   W: C12 gets 1
    U[0][0] += 1
    V[0][1] += 1
    V[1][1] += -1
    W[0][1] += 1
    col += 1
    
    # Term 4: A22 * (B21-B11)
    #   U: A22 gets 1
    #   V: B21 gets 1, B11 gets -1
    #   W: C11 gets 1, C21 gets 1
    U[1][1] += 1
    V[1][0] += 1
    V[0][0] += -1
    W[0][0] += 1
    W[1][0] += 1
    col += 1
    
    # Term 5: (A11+A12) * B22
    #   U: A11 and A12 get 1
    #   V: B22 gets 1
    #   W: C11 gets -1, C12 gets 1
    U[0][0] += 1
    U[0][1] += 1
    V[1][1] += 1
    W[0][0] += -1
    W[0][1] += 1
    col += 1
    
    # Term 6: (A21-A11) * (B11+B12)
    #   U: A21 gets 1, A11 gets -1
    #   V: B11 and B12 get 1
    #   W: C22 gets 1
    U[1][0] += 1
    U[0][0] += -1
    V[0][0] += 1
    V[0][1] += 1
    W[1][1] += 1
    col += 1
    
    # Term 7: (A12-A22) * (B21+B22)
    #   U: A12 gets 1, A22 gets -1
    #   V: B21 and B22 get 1
    #   W: C11 gets 1
    U[0][1] += 1
    U[1][1] += -1
    V[1][0] += 1
    V[1][1] += 1
    W[0][0] += 1
    col += 1
    
    return col

def _schoolbook_block(U, V, W, u_row, u_col, v_row, v_col, w_row, w_col, h, w, base_col):
    """Add schoolbook decomposition for h x w x w block (A is h x w, B is w x w, C is h x w).
    Actually, let me generalize: A is h x k, B is k x l, C is h x l.
    u_row, u_col: top-left of A sub-block in U
    v_row, v_col: top-left of B sub-block in V
    w_row, w_col: top-left of C sub-block in W
    h, k, l: dimensions
    """
    col = base_col
    for i in range(h):
        for j in range(k):
            for l in range(l):
                U[u_row + i][u_col + j] = U[u_row + i][u_col + j] if False else 1
                # Wait, I need to set the values, not add. Let me reconsider.
                # Actually for schoolbook, each term has exactly one 1 in U, one in V, one in W.
                U[u_row + i][u_col + j] = 1
                V[v_row + j][v_col + l] = 1
                W[w_row + i][w_col + l] = 1
                col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # A is n x m = 4 x 5, B is m x p = 5 x 5, C is p x n = 5 x 4
    # Wait, the output is W[k*n+i, r] which is p*n = 5*4 = 20 rows.
    # C[i][k] where i is row of C (0..n-1=0..3), k is col of C (0..p-1=0..4)
    # W index: k*n + i
    
    # Block structure: split n=4 into 2+2, m=5 into 2+3, p=5 into 2+3
    # A = [[A11(2x2), A12(2x3)], [A21(2x2), A22(2x3)]]
    # B = [[B11(2x2), B12(2x3)], [B21(3x2), B22(3x3)]]
    # C = [[C11(2x2), C12(2x3)], [C21(2x2), C22(2x3)]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # U has n*m = 20 rows, V has m*p = 25 rows, W has p*n = 20 rows
    # Let me estimate the rank:
    # A11*B11: 2x2x2 -> Strassen 7
    # A12*B21: 2x3x2 -> schoolbook 6
    # A11*B12: 2x2x3 -> schoolbook 6
    # A12*B22: 2x3x3 -> schoolbook 6
    # A21*B11: 2x2x2 -> Strassen 7
    # A22*B21: 2x3x2 -> schoolbook 6
    # A21*B12: 2x2x3 -> schoolbook 6
    # A22*B22: 2x3x3 -> schoolbook 6
    # Total: 7+6+6+6+7+6+6+6 = 50
    
    # Let me build this. I'll use a helper that tracks column offsets.
    
    # U rows: A is 4x5, indexed as A[i][j] where i in 0..3, j in 0..4
    #   A11: i=0..1, j=0..1 -> U rows 0..1, cols 0..1
    #   A12: i=0..1, j=2..4 -> U rows 0..1, cols 2..4
    #   A21: i=2..3, j=0..1 -> U rows 2..3, cols 0..1
    #   A22: i=2..3, j=2..4 -> U rows 2..3, cols 2..4
    # V rows: B is 5x5, indexed as B[j][k] where j in 0..4, k in 0..4
    #   B11: j=0..1, k=0..1 -> V rows 0..1, cols 0..1
    #   B12: j=0..1, k=2..4 -> V rows 0..1, cols 2..4
    #   B21: j=2..4, k=0..1 -> V rows 2..4, cols 0..1
    #   B22: j=2..4, k=2..4 -> V rows 2..4, cols 2..4
    # W rows: C is 5x4 (p x n), indexed as C[k][i] where k in 0..4, i in 0..3
    #   W index = k*n + i = k*4 + i
    #   C11: k=0..1, i=0..1 -> W rows 0..5 (k*4+i: 0,1,4,5)
    #   C12: k=0..1, i=2..4 -> W rows 2..3, 6..7 (k*4+i: 2,3,6,7)
    #   C21: k=2..4, i=0..1 -> W rows 8..13 (k*4+i: 8,9,12,13,16,17)
    #   C22: k=2..4, i=2..4 -> W rows 10..11, 14..15, 18..19
    
    # This is getting complicated. Let me just build it directly with a column counter.
    
    U = [[0]*1 for _ in range(n*m)]  # will resize
    V = [[0]*1 for _ in range(m*p)]
    W = [[0]*1 for _ in range(p*n)]
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(u_r0, u_c0, h, k, v_r0, v_c0, l, w_r0, w_c0, col_start):
        # A sub-block: rows u_r0..u_r0+h-1, cols u_c0..u_c0+k-1 in U
        # B sub-block: rows v_r0..v_r0+k-1, cols v_c0..v_c0+l-1 in V
        # C sub-block: rows w_r0..w_r0+h-1, cols w_c0..w_c0+l-1 in W
        # Wait, C is h x l, so W rows w_r0..w_r0+h-1, cols w_c0..w_c0+l-1
        c = col_start
        for i in range(h):
            for j in range(k):
                for l_idx in range(l):
                    U[u_r0+i][c] = 1
                    V[v_r0+j][c] = 1
                    W[w_r0+i][c] = 1
                    c += 1
        return c
    
    # Helper to add Strassen for 2x2x2
    def add_strassen(u_r0, u_c0, v_r0, v_c0, w_r0, w_c0, col_start):
        # A: 2x2 at U[u_r0:u_r0+2][u_c0:u_c0+2]
        # B: 2x2 at V[v_r0:v_r0+2][v_c0:v_c0+2]
        # C: 2x2 at W[w_r0:w_r0+2][w_c0:w_c0+2]
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        U[u_r0+0][c] = 1; U[u_r0+1][c] = 1
        V[v_r0+0][c] = 1; V[v_r0+1][c] = 1
        W[w_r0+0][c] = 1; W[w_r0+1][c] = 1
        c += 1
        # M2 = (A21+A22)B11
        U[u_r0+1][c] = 1; U[u_r0+1][c] = 1  # A21 is [1][0], A22 is [1][1]
        # Wait, A21 is U[u_r0+1][u_c0+0], A22 is U[u_r0+1][u_c0+1]
        # I'm overwriting! I need to accumulate. Let me use a different approach.
        # Let me build the full matrices first.
        pass
        return c
    
    # OK this is getting messy. Let me just build it step by step with explicit columns.
    
    # I'll create the matrices with enough columns and fill them in.
    # Total rank = 50 as computed above.
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: Strassen 2x2x2
    # A11: U rows 0-1, cols 0-1
    # B11: V rows 0-1, cols 0-1
    # C11: W rows for k=0,1 and i=0,1 -> W[k*4+i]
    #   k=0,i=0: W[0]; k=0,i=1: W[1]; k=1,i=0: W[4]; k=1,i=1: W[5]
    # So C11 in W: rows 0,1,4,5 -> not contiguous!
    # W is indexed as k*n+i, so for a 2x2 block of C at (k0..k0+1, i0..i0+1):
    #   W[k0*n+i0], W[k0*n+i0+1], W[(k0+1)*n+i0], W[(k0+1)*n+i0+1]
    # For C11: k0=0, i0=0 -> W[0], W[1], W[4], W[5]
    
    # This non-contiguous indexing makes Strassen harder to apply directly.
    # Let me just use schoolbook for everything to be safe, but that gives rank 100.
    # 
    # Alternative: I can permute the W rows to make blocks contiguous, but the output
    # must follow W[k*n+i] convention. So I can't permute.
    #
    # Let me think about this differently. I can still use Strassen, I just need to
    # map the C indices correctly.
    
    # For Strassen on A11*B11 -> C11:
    # A11[i][j] = U[i*m+j] = U[i*5+j] for i in 0..1, j in 0..1
    #   A11[0][0]=U[0], A11[0][1]=U[1], A11[1][0]=U[5], A11[1][1]=U[6]
    # B11[j][k] = V[j*p+k] = V[j*5+k] for j in 0..1, k in 0..1
    #   B11[0][0]=V[0], B11[0][1]=V[1], B11[1][0]=V[5], B11[1][1]=V[6]
    # C11[i][k] = W[k*n+i] = W[k*4+i] for i in 0..1, k in 0..1
    #   C11[0][0]=W[0], C11[0][1]=W[1], C11[1][0]=W[4], C11[1][1]=W[5]
    
    # Strassen terms for A11*B11:
    # M1 = (A11[0][0]+A11[1][1])(B11[0][0]+B11[1][1])
    #   U: U[0]=1, U[6]=1
    #   V: V[0]=1, V[6]=1
    #   W: C11[0][0]=W[0]=1, C11[1][1]=W[5]=1
    U[0][col] = 1; U[6][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # M2 = (A11[1][0]+A11[1][1]) * B11[0][0]
    #   U: U[5]=1, U[6]=1
    #   V: V[0]=1
    #   W: C11[1][0]=W[4]=1
    U[5][col] = 1; U[6][col] = 1
    V[0][col] = 1
    W[4][col] = 1
    col += 1
    
    # M3 = A11[0][0] * (B11[0][1]-B11[1][1])
    #   U: U[0]=1
    #   V: V[1]=1, V[6]=-1
    #   W: C11[0][1]=W[1]=1
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = A11[1][1] * (B11[1][0]-B11[0][0])
    #   U: U[6]=1
    #   V: V[5]=1, V[0]=-1
    #   W: C11[0][0]=W[0]=1, C11[1][0]=W[4]=1
    U[6][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M5 = (A11[0][0]+A11[0][1]) * B11[1][1]
    #   U: U[0]=1, U[1]=1
    #   V: V[6]=1
    #   W: C11[0][0]=W[0]=-1, C11[0][1]=W[1]=1
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (A11[1][0]-A11[0][0]) * (B11[0][0]+B11[0][1])
    #   U: U[5]=1, U[0]=-1
    #   V: V[0]=1, V[1]=1
    #   W: C11[1][1]=W[5]=1
    U[5][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # M7 = (A11[0][1]-A11[1][1]) * (B11[1][0]+B11[1][1])
    #   U: U[1]=1, U[6]=-1
    #   V: V[5]=1, V[6]=1
    #   W: C11[0][0]=W[0]=1
    U[1][col] = 1; U[6][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # A12*B21: schoolbook 2x3x2
    # A12: U rows 0-1, cols 2-4 -> U[i*5+j] for i in 0..1, j in 2..4
    #   A12[0][2]=U[2], A12[0][3]=U[3], A12[0][4]=U[4]
    #   A12[1][2]=U[7], A12[1][3]=U[8], A12[1][4]=U[9]
    # B21: V rows 2-4, cols 0-1 -> V[j*5+k] for j in 2..4, k in 0..1
    #   B21[2][0]=V[10], B21[2][1]=V[11]
    #   B21[3][0]=V[15], B21[3][1]=V[16]
    #   B21[4][0]=V[20], B21[4][1]=V[21]
    # C11: W[0], W[1], W[4], W[5]
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*5 + (2+j)][col] = 1
                V[(2+j)*5 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    # col should be 7+6=13
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: schoolbook 2x2x3
    # A11: U[0],U[1],U[5],U[6]
    # B12: V rows 0-1, cols 2-4 -> V[j*5+k] for j in 0..1, k in 2..4
    #   B12[0][2]=V[2], B12[0][3]=V[3], B12[0][4]=V[4]
    #   B12[1][2]=V[7], B12[1][3]=V[8], B12[1][4]=V[9]
    # C12: k=0..1, i=2..4 -> W[k*4+i]
    #   k=0,i=2:W[2], k=0,i=3:W[3], k=0,i=4:W[4]... wait i only goes to 3 for n=4
    #   i in 2..3 (since i < n=4)
    #   k=0,i=2:W[2], k=0,i=3:W[3]
    #   k=1,i=2:W[6], k=1,i=3:W[7]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*5 + j][col] = 1
                V[j*5 + (2+k)][col] = 1
                W[(2+k)*4 + i][col] = 1
                col += 1
    # col = 13+6=19
    
    # A12*B22: schoolbook 2x3x3
    # A12: U[2],U[3],U[4],U[7],U[8],U[9]
    # B22: V rows 2-4, cols 2-4 -> V[j*5+k] for j in 2..4, k in 2..4
    # C12: W[2],W[3],W[6],W[7]
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*5 + (2+j)][col] = 1
                V[(2+j)*5 + (2+k)][col] = 1
                W[(2+k)*4 + i][col] = 1
                col += 1
    # col = 19+6=25
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: Strassen 2x2x2
    # A21: U rows 2-3, cols 0-1 -> U[i*5+j] for i in 2..3, j in 0..1
    #   A21[0][0]=U[10], A21[0][1]=U[11], A21[1][0]=U[15], A21[1][1]=U[16]
    # B11: V[0],V[1],V[5],V[6]
    # C21: k=2..4, i=0..1 -> W[k*4+i]
    #   k=2,i=0:W[8], k=2,i=1:W[9]
    #   k=3,i=0:W[12], k=3,i=1:W[13]
    #   k=4,i=0:W[16], k=4,i=1:W[17]
    # Wait, C21 is 2x2 (i in 2..3 of A, k in 0..1 of B), so C21[i'][k'] where i'=i-2 in 0..1, k' in 0..1
    # C21[i'][k'] = W[(k'*2+2)*4 + (i'*2+2)]... no.
    # C is 5x4 (p x n), C[k][i] = W[k*4+i]
    # C21 corresponds to rows i=2,3 of C and cols k=0,1 of C
    # So C21[0][0]=C[2][0]=W[2*4+0]=W[8]
    #    C21[0][1]=C[2][1]=W[2*4+1]=W[9]
    #    C21[1][0]=C[3][0]=W[3*4+0]=W[12]
    #    C21[1][1]=C[3][1]=W[3*4+1]=W[13]
    
    # Strassen for A21*B11 -> C21:
    # A21[0][0]=U[10], A21[0][1]=U[11], A21[1][0]=U[15], A21[1][1]=U[16]
    # B11[0][0]=V[0], B11[0][1]=V[1], B11[1][0]=V[5], B11[1][1]=V[6]
    # C21[0][0]=W[8], C21[0][1]=W[9], C21[1][0]=W[12], C21[1][1]=W[13]
    
    # M1 = (A21[0][0]+A21[1][1])(B11[0][0]+B11[1][1])
    U[10][col]=1; U[16][col]=1
    V[0][col]=1; V[6][col]=1
    W[8][col]=1; W[13][col]=1
    col += 1
    
    # M2 = (A21[1][0]+A21[1][1]) * B11[0][0]
    U[15][col]=1; U[16][col]=1
    V[0][col]=1
    W[12][col]=1
    col += 1
    
    # M3 = A21[0][0] * (B11[0][1]-B11[1][1])
    U[10][col]=1
    V[1][col]=1; V[6][col]=-1
    W[9][col]=1
    col += 1
    
    # M4 = A21[1][1] * (B11[1][0]-B11[0][0])
    U[16][col]=1
    V[5][col]=1; V[0][col]=-1
    W[8][col]=1; W[12][col]=1
    col += 1
    
    # M5 = (A21[0][0]+A21[0][1]) * B11[1][1]
    U[10][col]=1; U[11][col]=1
    V[6][col]=1
    W[8][col]=-1; W[9][col]=1
    col += 1
    
    # M6 = (A21[1][0]-A21[0][0]) * (B11[0][0]+B11[0][1])
    U[15][col]=1; U[10][col]=-1
    V[0][col]=1; V[1][col]=1
    W[13][col]=1
    col += 1
    
    # M7 = (A21[0][1]-A21[1][1]) * (B11[1][0]+B11[1][1])
    U[11][col]=1; U[16][col]=-1
    V[5][col]=1; V[6][col]=1
    W[8][col]=1
    col += 1
    # col = 25+7=32
    
    # A22*B21: schoolbook 2x3x2
    # A22: U rows 2-3, cols 2-4 -> U[i*5+j] for i in 2..3, j in 2..4
    #   A22[0][2]=U[12], A22[0][3]=U[13], A22[0][4]=U[14]
    #   A22[1][2]=U[17], A22[1][3]=U[18], A22[1][4]=U[19]
    # B21: V[10],V[11],V[15],V[16],V[20],V[21]
    # C21: W[8],W[9],W[12],W[13]
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[(2+i)*5 + (2+j)][col] = 1
                V[(2+j)*5 + k][col] = 1
                W[(2+k)*4 + (2+i)][col] = 1
                col += 1
    # col = 32+6=38
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: schoolbook 2x2x3
    # A21: U[10],U[11],U[15],U[16]
    # B12: V[2],V[3],V[4],V[7],V[8],V[9]
    # C22: k=2..4, i=2..3 -> W[k*4+i]
    #   k=2,i=2:W[10], k=2,i=3:W[11]
    #   k=3,i=2:W[14], k=3,i=3:W[15]
    #   k=4,i=2:W[18], k=4,i=3:W[19]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[(2+i)*5 + j][col] = 1
                V[j*5 + (2+k)][col] = 1
                W[(2+k)*4 + (2+i)][col] = 1
                col += 1
    # col = 38+6=44
    
    # A22*B22: schoolbook 2x3x3
    # A22: U[12],U[13],U[14],U[17],U[18],U[19]
    # B22: V rows 2-4, cols 2-4
    # C22: W[10],W[11],W[14],W[15],W[18],W[19]
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[(2+i)*5 + (2+j)][col] = 1
                V[(2+j)*5 + (2+k)][col] = 1
                W[(2+k)*4 + (2+i)][col] = 1
                col += 1
    # col = 44+6=50
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

