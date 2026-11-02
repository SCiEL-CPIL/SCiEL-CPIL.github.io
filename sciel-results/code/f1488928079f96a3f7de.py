import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # Split m=4 into two blocks of size 2.
    # Block 1: j in {0, 1}
    # Block 2: j in {2, 3}
    # Each block is a 2x2x7 multiplication.
    # We decompose 2x2x7 into three 2x2x2 and one 2x2x1.
    # 2x2x2 uses Winograd (rank 7).
    # 2x2x1 uses Schoolbook (rank 4).
    # Total rank per block: 3*7 + 4 = 25.
    # Total rank: 25 + 25 = 50.
    
    # Define Winograd 2x2x2 factors
    # U_w: 4x7, V_w: 4x7, W_w: 4x7
    U_w = np.array([
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, -1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, -1]
    ], dtype=float)
    
    V_w = np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    W_w = np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, -1, 1],
        [1, 0, 1, 0, 0, 0, 1]
    ], dtype=float)
    
    # Verify Winograd 2x2x2
    def is_valid_222(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        for a in range(4):
            i_a, j_a = divmod(a, 2)
            for b in range(4):
                j_b, k_b = divmod(b, 2)
                for c in range(4):
                    k_c, i_c = divmod(c, 2)
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s) > 1e-6:
                        return False
        return True

    winograd_ok = is_valid_222(U_w, V_w, W_w)
    
    if not winograd_ok:
        # Fallback to schoolbook for 2x2x2 (rank 8)
        # This results in rank 56 total
        rank_block = 3*8 + 4 # 28
        total_rank = 2 * rank_block
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Fill Block 1 (j=0,1)
        # Use schoolbook for 2x2x2 blocks and 2x2x1
        # This part is complex, so just use global schoolbook if Winograd fails
        return _schoolbook()

    # Construct 2x2x7 factors
    # We need U_227 (4x25), V_227 (14x25), W_227 (14x25)
    # Split p=7 into k in {0,1}, {2,3}, {4,5}, {6}
    # k0: 0,1 (Winograd)
    # k1: 2,3 (Winograd)
    # k2: 4,5 (Winograd)
    # k3: 6 (Schoolbook 2x2x1)
    
    rank_227 = 7 + 7 + 7 + 4 # 25
    U_227 = np.zeros((4, rank_227))
    V_227 = np.zeros((14, rank_227))
    W_227 = np.zeros((14, rank_227))
    
    col = 0
    # Block k=0,1
    U_227[:, col:col+7] = U_w
    V_227[0:4, col:col+7] = V_w # j=0,1 -> rows 0-3 of V (local)
    # Wait, V_227 rows are (j,k). j in {0,1}, k in {0,1}.
    # Local j=0,1. Local k=0,1.
    # V_w rows are 0..3 corresponding to (0,0),(0,1),(1,0),(1,1).
    # So V_227[0:4, ...] = V_w is correct for k=0,1.
    W_227[0:4, col:col+7] = W_w # k=0,1 -> rows 0-3 of W (local k*n+i)
    col += 7
    
    # Block k=2,3
    U_227[:, col:col+7] = U_w
    V_227[4:8, col:col+7] = V_w # j=0,1, k=2,3 -> rows 4-7
    W_227[4:8, col:col+7] = W_w # k=2,3 -> rows 4-7
    col += 7
    
    # Block k=4,5
    U_227[:, col:col+7] = U_w
    V_227[8:12, col:col+7] = V_w
    W_227[8:12, col:col+7] = W_w
    col += 7
    
    # Block k=6 (2x2x1)
    # Schoolbook for 2x2x1
    # A 2x2, B 2x1, C 2x1
    # U_sb (4x4), V_sb (2x4), W_sb (2x4)
    U_sb = np.eye(4)
    V_sb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    W_sb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    # Wait, 2x2x1 schoolbook:
    # C_0 = A_00 B_00 + A_01 B_10
    # C_1 = A_10 B_00 + A_11 B_10
    # U rows: A00, A01, A10, A11
    # V rows: B00, B10
    # W rows: C0, C1
    # U = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    # V = [[1,0,1,0], [0,1,0,1]]
    # W = [[1,1,0,0], [0,0,1,1]]
    U_sb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_sb = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_sb = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    U_227[:, col:col+4] = U_sb
    V_227[12:14, col:col+4] = V_sb
    W_227[12:14, col:col+4] = W_sb
    col += 4
    
    # Now combine Block 1 (j=0,1) and Block 2 (j=2,3)
    # Total rank 50
    total_rank = 50
    U = np.zeros((8, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((14, total_rank))
    
    # Block 1: j=0,1
    # U rows 0-3 (i=0,1; j=0,1)
    U[0:4, 0:25] = U_227
    # V rows 0-13 (j=0,1; k=0..6)
    V[0:14, 0:25] = V_227
    # W rows 0-13 (k=0..6; i=0,1)
    W[0:14, 0:25] = W_227
    
    # Block 2: j=2,3
    # U rows 4-7 (i=0,1; j=2,3)
    # Local j' = 0 for global 2, 1 for global 3.
    # U_227 rows are 0..3 (local i,j').
    # Global row index: i*4 + j.
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 6. i=1, j=3 -> 7.
    # Wait, U index is i*m+j. m=4.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    
    # My previous mapping for Block 1:
    # U[0:4] corresponds to i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1?
    # No, U_227 rows are (i, j') where i in 0,1 and j' in 0,1.
    # Row 0: i=0, j'=0. Global: i=0, j=0. Index 0.
    # Row 1: i=0, j'=1. Global: i=0, j=1. Index 1.
    # Row 2: i=1, j'=0. Global: i=1, j=0. Index 4.
    # Row 3: i=1, j'=1. Global: i=1, j=1. Index 5.
    
    # So I need to permute U_227 rows when assigning to U.
    # U_227 order: (0,0), (0,1), (1,0), (1,1)
    # Global U order: (0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)
    
    # Block 1 (j=0,1):
    # U_227[0] -> U[0]
    # U_227[1] -> U[1]
    # U_227[2] -> U[4]
    # U_227[3] -> U[5]
    
    U[0, 0:25] = U_227[0]
    U[1, 0:25] = U_227[1]
    U[4, 0:25] = U_227[2]
    U[5, 0:25] = U_227[3]
    
    # Block 2 (j=2,3):
    # Local j'=0 -> Global j=2. Local j'=1 -> Global j=3.
    # U_227[0] (i=0, j'=0) -> Global (0,2) -> Index 2.
    # U_227[1] (i=0, j'=1) -> Global (0,3) -> Index 3.
    # U_227[2] (i=1, j'=0) -> Global (1,2) -> Index 6.
    # U_227[3] (i=1, j'=1) -> Global (1,3) -> Index 7.
    
    U[2, 25:50] = U_227[0]
    U[3, 25:50] = U_227[1]
    U[6, 25:50] = U_227[2]
    U[7, 25:50] = U_227[3]
    
    # V mapping
    # V_227 rows (j', k). j' in 0,1. k in 0..6.
    # Global V rows (j, k). j in 0..3.
    # Block 1: j=0,1.
    # V_227[0:4] (j'=0) -> Global j=0. Rows 0..6.
    # V_227[4:8] (j'=1) -> Global j=1. Rows 7..13.
    # V_227[8:12] (j'=0, k=4,5)? No.
    # V_227 was constructed with rows 0..13.
    # Rows 0..3: j'=0, k=0,1 (Winograd 1)
    # Rows 4..7: j'=0, k=2,3 (Winograd 2) -- WAIT.
    
    # My V_227 construction:
    # V_227[0:4, 0:7] = V_w (j'=0, k=0,1)
    # V_227[4:8, 7:14] = V_w (j'=0, k=2,3) ?? No.
    # V_w is 4x7. Rows are (0,0),(0,1),(1,0),(1,1).
    # I assigned V_w to V_227[0:4] for k=0,1.
    # This implies V_227 rows 0..3 correspond to j'=0, k=0,1?
    # No, V_w row 0 is (j'=0, k=0). V_w row 1 is (j'=0, k=1).
    # So V_w covers j'=0,1 and k=0,1.
    # So V_227[0:4] covers (j'=0,k=0), (j'=0,k=1), (j'=1,k=0), (j'=1,k=1).
    
    # Let's re-verify V_227 structure.
    # V_227 should have 14 rows: (j', k) for j' in 0,1 and k in 0..6.
    # Order: (0,0), (0,1), ..., (0,6), (1,0), ..., (1,6).
    
    # My previous assignment:
    # V_227[0:4, 0:7] = V_w.
    # V_w rows: (0,0), (0,1), (1,0), (1,1).
    # This maps to V_227 rows 0,1,2,3.
    # This corresponds to (j'=0,k=0), (j'=0,k=1), (j'=1,k=0), (j'=1,k=1).
    # This is NOT the standard row order (0,0)..(0,6).
    # I need to permute V_227 rows to match Global V order.
    
    # Let's build V_227 correctly.
    V_227_correct = np.zeros((14, 25))
    # k=0,1 (Winograd 1)
    # V_w rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Target rows in V_227:
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 7
    # (1,1) -> 8
    V_227_correct[0, 0:7] = V_w[0]
    V_227_correct[1, 0:7] = V_w[1]
    V_227_correct[7, 0:7] = V_w[2]
    V_227_correct[8, 0:7] = V_w[3]
    
    # k=2,3 (Winograd 2)
    V_227_correct[2, 7:14] = V_w[0]
    V_227_correct[3, 7:14] = V_w[1]
    V_227_correct[9, 7:14] = V_w[2]
    V_227_correct[10, 7:14] = V_w[3]
    
    # k=4,5 (Winograd 3)
    V_227_correct[4, 14:21] = V_w[0]
    V_227_correct[5, 14:21] = V_w[1]
    V_227_correct[11, 14:21] = V_w[2]
    V_227_correct[12, 14:21] = V_w[3]
    
    # k=6 (Schoolbook)
    # V_sb rows: 0:(0,0), 1:(1,0)
    V_227_correct[6, 21:25] = V_sb[0]
    V_227_correct[13, 21:25] = V_sb[1]
    
    # Now assign to Global V
    # Block 1 (j=0,1):
    # Global V rows 0..13.
    # V_227_correct rows 0..13 correspond to (j'=0,k=0..6) and (j'=1,k=0..6).
    # Global j=0 -> j'=0. Global j=1 -> j'=1.
    # So V[0:7] = V_227_correct[0:7]
    # V[7:14] = V_227_correct[7:14]
    V[0:7, 0:25] = V_227_correct[0:7]
    V[7:14, 0:25] = V_227_correct[7:14]
    
    # Block 2 (j=2,3):
    # Global V rows 14..27.
    # Global j=2 -> j'=0. Global j=3 -> j'=1.
    # V[14:21] = V_227_correct[0:7]
    # V[21:28] = V_227_correct[7:14]
    V[14:21, 25:50] = V_227_correct[0:7]
    V[21:28, 25:50] = V_227_correct[7:14]
    
    # W mapping
    # W_227 rows (k, i). k in 0..6, i in 0,1.
    # Order: (0,0), (0,1), (1,0), (1,1), ... (6,0), (6,1).
    # W_w rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # W_227_correct construction similar to V.
    W_227_correct = np.zeros((14, 25))
    # k=0,1
    W_227_correct[0, 0:7] = W_w[0]
    W_227_correct[1, 0:7] = W_w[1]
    W_227_correct[2, 0:7] = W_w[2]
    W_227_correct[3, 0:7] = W_w[3]
    
    # k=2,3
    W_227_correct[4, 7:14] = W_w[0]
    W_227_correct[5, 7:14] = W_w[1]
    W_227_correct[6, 7:14] = W_w[2]
    W_227_correct[7, 7:14] = W_w[3]
    
    # k=4,5
    W_227_correct[8, 14:21] = W_w[0]
    W_227_correct[9, 14:21] = W_w[1]
    W_227_correct[10, 14:21] = W_w[2]
    W_227_correct[11, 14:21] = W_w[3]
    
    # k=6
    W_227_correct[12, 21:25] = W_sb[0]
    W_227_correct[13, 21:25] = W_sb[1]
    
    # Global W rows 0..13.
    # W depends on k and i.
    # Both blocks use the same W structure for their k range.
    # Block 1 uses W_227_correct.
    # Block 2 uses W_227_correct.
    # Since W is summed over blocks, we add them.
    W[0:14, 0:25] = W_227_correct
    W[0:14, 25:50] = W_227_correct
    
    # Final check
    def check_valid(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s) > 1e-6:
                        return False
        return True

    if check_valid(U, V, W, total_rank):
        U_list = [[float(U[i, j]) for j in range(total_rank)] for i in range(n*m)]
        V_list = [[float(V[i, j]) for j in range(total_rank)] for i in range(m*p)]
        W_list = [[float(W[i, j]) for j in range(total_rank)] for i in range(p*n)]
        
        def clean_val(x):
            if abs(x - round(x)) < 1e-10:
                return int(round(x))
            elif abs(2*x - round(2*x)) < 1e-10:
                return round(2*x) / 2
            else:
                return float(x)
        
        U_list = [[clean_val(x) for x in row] for row in U_list]
        V_list = [[clean_val(x) for x in row] for row in V_list]
        W_list = [[clean_val(x) for x in row] for row in W_list]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
