"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
_start = time.time()
_DEADLINE = 120.0  # 150s wall-clock limit with margin

def _time_ok():
    return (time.time() - _start) < _DEADLINE

def construct():
    """
    Build rank-78 decomposition for 3x4 * 4x7 matrix multiplication.
    
    Strategy:
    - Row 0: schoolbook 1x4x7 = 28 terms
    - Rows 1-2, Cols 0-1: 6 Strassen 2x2x2 blocks = 30 terms
    - Rows 1-2, Cols 2-3: 4 Strassen 2x2x2 blocks = 20 terms
    Total: 78
    
    U shape: (3*4, rank) = (12, rank)
    V shape: (4*7, rank) = (28, rank)  
    W shape: (7*3, rank) = (21, rank)
    
    Indexing:
    U[i*m + j, c] where m=4
    V[j*p + k, c] where p=7
    W[k*n + i, c] where n=3 (official trace convention)
    """
    n, m, p = 3, 4, 7
    
    # Accumulator for terms
    U = []  # Will be list of lists
    V = []
    W = []
    
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        """Add one rank-1 term to the decomposition."""
        # u_indices: list of (row_in_U, col_in_U) pairs with values
        # v_indices: list of (row_in_V, col_in_V) pairs with values
        # w_indices: list of (row_in_W, col_in_W) pairs with values
        
        # We'll append to global lists
        U_row = [0] * 0  # placeholder, will grow
        V_row = [0] * 0
        W_row = [0] * 0
        
        # Actually, let's just store terms as tuples and build at end
        pass
    
    # Store all terms as (u_sparse, v_sparse, w_sparse)
    terms = []
    
    def add_sparse_term(u_dict, v_dict, w_dict):
        """
        u_dict: {row: value} for U matrix
        v_dict: {row: value} for V matrix
        w_dict: {row: value} for W matrix
        """
        terms.append((u_dict, v_dict, w_dict))
    
    # ============================================================
    # PART 1: Row 0, all columns (1x4x7) - schoolbook, 28 terms
    # i=0, j=0..3, k=0..6
    # ============================================================
    for j in range(4):
        for k in range(7):
            u = {0 * 4 + j: 1}  # U row = i*m + j = 0*4+j
            v = {j * 7 + k: 1}  # V row = j*p + k
            w = {k * 3 + 0: 1}  # W row = k*n + i
            add_sparse_term(u, v, w)
    
    # ============================================================
    # PART 2: Rows 1-2, Cols 0-1 (2x2x2 blocks) - Strassen, 6 blocks
    # For each block: i_block in {1,2}, j_block in {0,1}, k_block in {0,1}
    # Actually: i in {1,2}, j in {0,1}, k in {0,1}
    # 6 blocks: (j_offset, k_offset) where j_offset in {0,1}, k_offset in {0,1}
    # Wait, we have 2 rows (1,2) x 2 cols (0,1) x 2 "k-groups" (0-1, 2-3)
    # Let me re-think the partitioning.
    # ============================================================
    
    # Better partitioning for rows 1-2:
    # Cols j=0,1: split k into {0,1}, {2,3}, {4,5}, {6}
    #   - k={0,1}: 2x2x2 Strassen
    #   - k={2,3}: 2x2x2 Strassen
    #   - k={4,5}: 2x2x2 Strassen
    #   - k={6}: 2x2x1 schoolbook = 4 terms
    # Cols j=2,3: same pattern
    #   - k={0,1}: 2x2x2 Strassen
    #   - k={2,3}: 2x2x2 Strassen
    #   - k={4,5}: 2x2x2 Strassen
    #   - k={6}: 2x2x1 schoolbook = 4 terms
    
    # Total for rows 1-2: 6*5 + 2*4 = 30 + 8 = 38
    # Total: 28 + 38 = 66
    
    # Strassen 2x2x2: for local indices (ii, jj, kk) in {0,1}^3
    # The 5 Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # C11 = M1 + M4 - M2 - M5
    # C12 = M3 + M1
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Add a 2x2x2 Strassen block.
        i_base: base row index in {1,2} (so i in {i_base, i_base+1})
        j_base: base col index in {0,1,2,3} (so j in {j_base, j_base+1})
        k_base: base k index in {0,1,2,3,4,5,6} (so k in {k_base, k_base+1})
        
        Local mapping:
        A[i][j] where i in {0,1} -> global i = i_base + i
        B[j][k] where j in {0,1} -> global j = j_base + j
        C[i][k] where i in {0,1}, k in {0,1} -> global i = i_base + i, k = k_base + k
        """
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        
        # Each term contributes to specific U,V,W entries
        # U row = i_global * m + j_global = (i_base+li)*4 + (j_base+ljj)
        # V row = j_global * p + k_global = (j_base+ljj)*7 + (k_base+lkk)
        # W row = k_global * n + i_global = (k_base+lkk)*3 + (i_base+li)
        
        # M1: (A00+A11)(B00+B11)
        # U: A00 and A11 both coefficient 1
        # V: B00 and B11 both coefficient 1
        # W: C11 += 1, C12 += 1, C21 += 1, C22 -= 1
        u1 = {}
        v1 = {}
        w1 = {}
        # A00: i=i_base, j=j_base
        u1[(i_base) * 4 + (j_base)] = u1.get((i_base) * 4 + (j_base), 0) + 1
        # A11: i=i_base+1, j=j_base+1
        u1[(i_base + 1) * 4 + (j_base + 1)] = u1.get((i_base + 1) * 4 + (j_base + 1), 0) + 1
        # B00: j=j_base, k=k_base
        v1[(j_base) * 7 + (k_base)] = v1.get((j_base) * 7 + (k_base), 0) + 1
        # B11: j=j_base+1, k=k_base+1
        v1[(j_base + 1) * 7 + (k_base + 1)] = v1.get((j_base + 1) * 7 + (k_base + 1), 0) + 1
        # C contributions: C11+=1, C12+=1, C21+=1, C22-=1
        # C11: i=i_base, k=k_base -> W row = k_base*3 + i_base
        w1[(k_base) * 3 + (i_base)] = w1.get((k_base) * 3 + (i_base), 0) + 1
        # C12: i=i_base, k=k_base+1 -> W row = (k_base+1)*3 + i_base
        w1[(k_base + 1) * 3 + (i_base)] = w1.get((k_base + 1) * 3 + (i_base), 0) + 1
        # C21: i=i_base+1, k=k_base -> W row = k_base*3 + (i_base+1)
        w1[(k_base) * 3 + (i_base + 1)] = w1.get((k_base) * 3 + (i_base + 1), 0) + 1
        # C22: i=i_base+1, k=k_base+1 -> W row = (k_base+1)*3 + (i_base+1)
        w1[(k_base + 1) * 3 + (i_base + 1)] = w1.get((k_base + 1) * 3 + (i_base + 1), 0) - 1
        add_sparse_term(u1, v1, w1)
        
        # M2: (A10+A11)B00
        # U: A10 coeff 1, A11 coeff 1
        # V: B00 coeff 1
        # W: C11 -= 1, C21 += 1
        u2 = {}
        v2 = {}
        w2 = {}
        # A10: i=i_base+1, j=j_base
        u2[(i_base + 1) * 4 + (j_base)] = u2.get((i_base + 1) * 4 + (j_base), 0) + 1
        # A11: i=i_base+1, j=j_base+1
        u2[(i_base + 1) * 4 + (j_base + 1)] = u2.get((i_base + 1) * 4 + (j_base + 1), 0) + 1
        # B00: j=j_base, k=k_base
        v2[(j_base) * 7 + (k_base)] = v2.get((j_base) * 7 + (k_base), 0) + 1
        # C11 -= 1
        w2[(k_base) * 3 + (i_base)] = w2.get((k_base) * 3 + (i_base), 0) - 1
        # C21 += 1
        w2[(k_base) * 3 + (i_base + 1)] = w2.get((k_base) * 3 + (i_base + 1), 0) + 1
        add_sparse_term(u2, v2, w2)
        
        # M3: A00(B10-B00)
        # U: A00 coeff 1
        # V: B10 coeff 1, B00 coeff -1
        # W: C12 += 1, C22 -= 1
        u3 = {}
        v3 = {}
        w3 = {}
        # A00: i=i_base, j=j_base
        u3[(i_base) * 4 + (j_base)] = u3.get((i_base) * 4 + (j_base), 0) + 1
        # B10: j=j_base+1, k=k_base
        v3[(j_base + 1) * 7 + (k_base)] = v3.get((j_base + 1) * 7 + (k_base), 0) + 1
        # B00: j=j_base, k=k_base
        v3[(j_base) * 7 + (k_base)] = v3.get((j_base) * 7 + (k_base), 0) - 1
        # C12 += 1
        w3[(k_base + 1) * 3 + (i_base)] = w3.get((k_base + 1) * 3 + (i_base), 0) + 1
        # C22 -= 1
        w3[(k_base + 1) * 3 + (i_base + 1)] = w3.get((k_base + 1) * 3 + (i_base + 1), 0) - 1
        add_sparse_term(u3, v3, w3)
        
        # M4: A11(B01-B11)
        # U: A11 coeff 1
        # V: B01 coeff 1, B11 coeff -1
        # W: C11 += 1, C21 += 1
        u4 = {}
        v4 = {}
        w4 = {}
        # A11: i=i_base+1, j=j_base+1
        u4[(i_base + 1) * 4 + (j_base + 1)] = u4.get((i_base + 1) * 4 + (j_base + 1), 0) + 1
        # B01: j=j_base, k=k_base+1
        v4[(j_base) * 7 + (k_base + 1)] = v4.get((j_base) * 7 + (k_base + 1), 0) + 1
        # B11: j=j_base+1, k=k_base+1
        v4[(j_base + 1) * 7 + (k_base + 1)] = v4.get((j_base + 1) * 7 + (k_base + 1), 0) - 1
        # C11 += 1
        w4[(k_base) * 3 + (i_base)] = w4.get((k_base) * 3 + (i_base), 0) + 1
        # C21 += 1
        w4[(k_base) * 3 + (i_base + 1)] = w4.get((k_base) * 3 + (i_base + 1), 0) + 1
        add_sparse_term(u4, v4, w4)
        
        # M5: (A00+A01)B11
        # U: A00 coeff 1, A01 coeff 1
        # V: B11 coeff 1
        # W: C11 -= 1, C22 += 1
        u5 = {}
        v5 = {}
        w5 = {}
        # A00: i=i_base, j=j_base
        u5[(i_base) * 4 + (j_base)] = u5.get((i_base) * 4 + (j_base), 0) + 1
        # A01: i=i_base, j=j_base+1
        u5[(i_base) * 4 + (j_base + 1)] = u5.get((i_base) * 4 + (j_base + 1), 0) + 1
        # B11: j=j_base+1, k=k_base+1
        v5[(j_base + 1) * 7 + (k_base + 1)] = v5.get((j_base + 1) * 7 + (k_base + 1), 0) + 1
        # C11 -= 1
        w5[(k_base) * 3 + (i_base)] = w5.get((k_base) * 3 + (i_base), 0) - 1
        # C22 += 1
        w5[(k_base + 1) * 3 + (i_base + 1)] = w5.get((k_base + 1) * 3 + (i_base + 1), 0) + 1
        add_sparse_term(u5, v5, w5)
    
    def add_schoolbook_2x2x1(i_base, j_base, k_val):
        """
        Add 2x2x1 schoolbook block (4 terms).
        i in {i_base, i_base+1}, j in {j_base, j_base+1}, k=k_val
        """
        for di in range(2):
            for dj in range(2):
                i_global = i_base + di
                j_global = j_base + dj
                u = {i_global * 4 + j_global: 1}
                v = {j_global * 7 + k_val: 1}
                w = {k_val * 3 + i_global: 1}
                add_sparse_term(u, v, w)
    
    # Rows 1-2, j=0,1:
    add_strassen_block(1, 0, 0)  # k=0,1
    add_strassen_block(1, 0, 2)  # k=2,3
    add_strassen_block(1, 0, 4)  # k=4,5
    add_schoolbook_2x2x1(1, 0, 6)  # k=6
    
    # Rows 1-2, j=2,3:
    add_strassen_block(1, 2, 0)  # k=0,1
    add_strassen_block(1, 2, 2)  # k=2,3
    add_strassen_block(1, 2, 4)  # k=4,5
    add_schoolbook_2x2x1(1, 2, 6)  # k=6
    
    # Build final matrices
    rank = len(terms)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for row, val in u_dict.items():
            if 0 <= row < n * m:
                U_mat[row][c] = val
        for row, val in v_dict.items():
            if 0 <= row < m * p:
                V_mat[row][c] = val
        for row, val in w_dict.items():
            if 0 <= row < p * n:
                W_mat[row][c] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

# EVOLVE-BLOCK-END
