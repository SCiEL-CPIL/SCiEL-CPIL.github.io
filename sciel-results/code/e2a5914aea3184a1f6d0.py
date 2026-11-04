"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    import random
    random.seed(42)
    
    # Laderman's 3x3x3 rank-23 decomposition
    # U3x3[i,k] = coefficient of A[i,k] in the i-th multiplication
    # V3x3[k,j] = coefficient of B[k,j] in the j-th multiplication  
    # W3x3[i,j] = coefficient of C[i,j] output
    # For each of 23 multiplications r: sum_r U[i,k]*V[k,j]*W[i,j] = delta
    
    # Laderman 1976 standard form (verified):
    # The 23 bilinear forms and their contributions
    L_U = [[0]*23 for _ in range(9)]  # indexed by (i*3+k)
    L_V = [[0]*23 for _ in range(9)]  # indexed by (k*3+j)
    L_W = [[0]*23 for _ in range(9)]  # indexed by (i*3+j)
    
    # Laderman's algorithm from the paper:
    # M1 = A00*B00
    # M2 = A00*B01 + A01*B00
    # M3 = A01*B01
    # M4 = A00*B02 + A01*B02
    # M5 = A02*B00
    # M6 = A02*B01
    # M7 = A02*B02
    # M8 = A10*B10
    # M9 = A10*B11 + A11*B10
    # M10 = A11*B11
    # M11 = A10*B12 + A11*B12
    # M12 = A12*B10
    # M13 = A12*B11
    # M14 = A12*B12
    # M15 = A20*B20
    # M16 = A20*B21 + A21*B20
    # M17 = A21*B21
    # M18 = A20*B22 + A21*B22
    # M19 = A22*B20
    # M20 = A22*B21
    # M21 = A22*B22
    # M22 = (A00+A01+A02+A10+A11+A12+A20+A21+A22)*(B00+B01+B02+B10+B11+B12+B20+B21+B22)  -- no, this isn't right
    
    # Let me use a cleaner verified Laderman:
    # From Laderman (1976), the 23 multiplications for 3x3:
    
    # I'll use the formulation where we have:
    # For each output C[i,j] = sum_k A[i,k]*B[k,j]
    # We need: sum_r U[i,k,r] * V[k,j,r] * W[i,j,r] = delta_{i,k,j} (Kronecker delta for the tensor)
    
    # Actually the tensor T[i,j,k] = 1 means C[i,j] gets A[i,k]*B[k,j]
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # Wait no. The standard decomposition is:
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # = sum_r (sum_k U[i,k,r]*A[i,k]) * (sum_k' V[k',j,r]*B[k',j]) * W[i,j,r]
    # Hmm, that's not quite right either.
    
    # The correct formulation:
    # A and B are the input matrices, C = A*B
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # We want: C[i,j] = sum_r alpha_r(i,j) * mu_r
    # where mu_r = (linear form in A) * (linear form in B)
    # So: mu_r = (sum_k U[i,k,r] * A[i,k]) * (sum_k V[k,j,r] * B[k,j])
    # And C[i,j] = sum_r W[i,j,r] * mu_r
    # For this to equal sum_k A[i,k]*B[k,j], we need:
    # sum_r W[i,j,r] * U[i',k',r] * V[k'',j',r] = delta_{i,i'}*delta_{j,j'}*delta_{k',k''}
    # Wait, that's the tensor condition.
    
    # The tensor T has entries T[i,j,k] where:
    # T[i,j,k] contributes to C[i,j] the term A[i,k]*B[k,j]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    # And the decomposition is: T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[i,j,r]
    # Wait, I need to be more careful.
    
    # In the standard tensor decomposition of matrix multiplication:
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_k ⊗ e_j ⊗ e_{(i,j)}  (using appropriate indexing)
    # For our convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r] is the coefficient for input A[i,j] in the r-th bilinear form
    # V[j*p+k, r] is the coefficient for input B[j,k] in the r-th bilinear form
    # W[k*n+i, r] is the coefficient for output C[i,k] from the r-th bilinear form
    
    # The condition is: for all (i, j, k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    # For Laderman 3x3x3: n=3, m=3, p=3
    # U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    # Condition: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k in {0,1,2}
    
    # Laderman's 23 multiplications (from the paper):
    # I'll define them directly.
    
    # Let me use a known correct set. Here's Laderman's algorithm:
    # The 23 scalar multiplications are:
    
    # I'll hardcode a verified Laderman decomposition.
    # Reference: Laderman, "On the multiplication of matrices" (1976)
    
    # Using the formulation from various references:
    L_U_data = [
        # row 0: i=0, j=0 (A[0,0])
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # row 1: i=0, j=1 (A[0,1])
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # row 2: i=0, j=2 (A[0,2])
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # row 3: i=1, j=0 (A[1,0])
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # row 4: i=1, j=1 (A[1,1])
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # row 5: i=1, j=2 (A[1,2])
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # row 6: i=2, j=0 (A[2,0])
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # row 7: i=2, j=1 (A[2,1])
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # row 8: i=2, j=2 (A[2,2])
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    # This isn't right - I need the actual Laderman coefficients.
    # Let me use a different, verified approach.
    
    # I'll construct the Laderman decomposition using the known structure.
    # Laderman's key insight: use Strassen-like combinations on 2x2 blocks
    # plus handle the 3rd row/column separately.
    
    # Actually, let me just use a simpler but still good approach:
    # Use multiple 2x2x2 Strassen blocks + schoolbook for the rest.
    # This is what the current program does, but let me try different tilings.
    
    # Better approach: use the "block" structure more carefully.
    # For 3x5x7, we can partition as:
    # i: [0,1], [2] (sizes 2, 1)
    # j: [0,1], [2,3], [4] (sizes 2, 2, 1)  
    # k: [0,1], [2,3], [4,5], [6] (sizes 2, 2, 2, 1)
    
    # For each (i_block, j_block, k_block) combination:
    # If all sizes are 2: use Strassen (7 mults)
    # Otherwise: use schoolbook (size_i * size_j * size_k mults)
    
    def add_strassen(U, V, W, i0, j0, k0, col, n, m, p):
        """Add a 2x2x2 Strassen block starting at (i0, j0, k0)"""
        # M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1])*(B[j0,k0])
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3 = (A[i0,j0])*(B[j0,k0+1]-B[j0+1,k0+1])
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4 = (A[i0+1,j0+1])*(B[j0+1,k0]-B[j0,k0])
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j0+1])*(B[j0+1,k0+1])
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6 = (A[i0+1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k0+1])
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1])*(B[j0+1,k0]+B[j0+1,k0+1])
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col
    
    def compute_rank(i_blocks, j_blocks, k_blocks, n, m, p):
        """Compute the total rank for a given tiling without building matrices."""
        total = 0
        for (i0, i1) in i_blocks:
            for (j0, j1) in j_blocks:
                for (k0, k1) in k_blocks:
                    i_size = i1 - i0 + 1
                    j_size = j1 - j0 + 1
                    k_size = k1 - k0 + 1
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        total += 7
                    else:
                        total += i_size * j_size * k_size
        return total
    
    def build_tiling(U, V, W, i_blocks, j_blocks, k_blocks, n, m, p):
        col = 0
        for (i0, i1) in i_blocks:
            for (j0, j1) in j_blocks:
                for (k0, k1) in k_blocks:
                    i_size = i1 - i0 + 1
                    j_size = j1 - j0 + 1
                    k_size = k1 - k0 + 1
                    
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        col = add_strassen(U, V, W, i0, j0, k0, col, n, m, p)
                    else:
                        for i in range(i0, i1 + 1):
                            for j in range(j0, j1 + 1):
                                for k in range(k0, k1 + 1):
                                    U[i * m + j][col] = 1
                                    V[j * p + k][col] = 1
                                    W[k * n + i][col] = 1
                                    col += 1
        return col
    
    def generate_random_tilings(num_tilings, n, m, p):
        """Generate random tilings with 2-element blocks where possible."""
        tilings = []
        
        def random_partition(size):
            """Generate a random partition of [0, size-1] into blocks of size 1 or 2."""
            blocks = []
            pos = 0
            while pos < size:
                if pos + 1 < size and random.random() < 0.6:
                    blocks.append((pos, pos + 1))
                    pos += 2
                else:
                    blocks.append((pos, pos))
                    pos += 1
            return blocks
        
        for _ in range(num_tilings):
            i_blocks = random_partition(n)
            j_blocks = random_partition(m)
            k_blocks = random_partition(p)
            tilings.append((i_blocks, j_blocks, k_blocks))
        
        return tilings
    
    # Try multiple tilings
    tilings = [
        # Tiling 1: original
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,5),(6,6)]),
        # Tiling 2: different k partition
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,2),(3,4),(5,6)]),
        # Tiling 3: different k partition
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,4),(5,6)]),
        # Tiling 4: different j partition
        ([(0,1),(2,2)], [(0,2),(3,4)], [(0,1),(2,3),(4,5),(6,6)]),
        # Tiling 5: different j partition
        ([(0,1),(2,2)], [(0,2),(3,4)], [(0,2),(3,4),(5,6)]),
        # Tiling 6: different i partition
        ([(0,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,5),(6,6)]),
        # Tiling 7: more 2x2x2 blocks
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,5),(6,6)]),
        # Tiling 8: j as [0,1],[2,3,4]
        ([(0,1),(2,2)], [(0,1),(2,4)], [(0,1),(2,3),(4,5),(6,6)]),
        # Tiling 9: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6] (6 Strassen)
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,5),(6,6)]),
        # Tiling 10: maximize 2x2x2: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Same as 9, but let's try: i=[0,1],[2], j=[0,1],[2,4], k=[0,1],[2,3],[4,5],[6]
        ([(0,1),(2,2)], [(0,1),(2,4)], [(0,1),(2,3),(4,5),(6,6)]),
        # Tiling 11: i=[0,1],[2], j=[0,2],[3,4], k=[0,1],[2,3],[4,5],[6]
        ([(0,1),(2,2)], [(0,2),(3,4)], [(0,1),(2,3),(4,5),(6,6)]),
        # Tiling 12: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,2],[3,4],[5,6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,2),(3,4),(5,6)]),
        # Tiling 13: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,4],[5,6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,4),(5,6)]),
        # Tiling 14: i=[0,1],[2], j=[0,2],[3,4], k=[0,2],[3,4],[5,6]
        ([(0,1),(2,2)], [(0,2),(3,4)], [(0,2),(3,4),(5,6)]),
        # Tiling 15: i=[0,1],[2], j=[0,1],[2,4], k=[0,2],[3,4],[5,6]
        ([(0,1),(2,2)], [(0,1),(2,4)], [(0,2),(3,4),(5,6)]),
        # Tiling 16: i=[0,1],[2], j=[0,2],[3,4], k=[0,1],[2,4],[5,6]
        ([(0,1),(2,2)], [(0,2),(3,4)], [(0,1),(2,4),(5,6)]),
        # Tiling 17: i=[0,1],[2], j=[0,1],[2,4], k=[0,1],[2,4],[5,6]
        ([(0,1),(2,2)], [(0,1),(2,4)], [(0,1),(2,4),(5,6)]),
        # Tiling 18: i=[0,1],[2], j=[0,2],[3,4], k=[0,1],[2,3],[4,6]
        ([(0,1),(2,2)], [(0,2),(3,4)], [(0,1),(2,3),(4,6)]),
        # Tiling 19: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,6)]),
        # Tiling 20: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,3],[4,6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,3),(4,6)]),
    ]
    
    # Add random tilings
    if time.time() < deadline - 10:
        random_tilings = generate_random_tilings(200, n, m, p)
        tilings.extend(random_tilings)
    
    candidates = []
    max_rank = 150
    
    # First, rank all tilings by their theoretical rank and only build the best ones
    tiling_ranks = []
    for idx, (i_blocks, j_blocks, k_blocks) in enumerate(tilings):
        r = compute_rank(i_blocks, j_blocks, k_blocks, n, m, p)
        tiling_ranks.append((r, idx, i_blocks, j_blocks, k_blocks))
    
    tiling_ranks.sort(key=lambda x: x[0])
    
    # Only build the top candidates by rank
    num_to_build = min(len(tiling_ranks), 30)
    
    for r_est, idx, i_blocks, j_blocks, k_blocks in tiling_ranks[:num_to_build]:
        if time.time() > deadline - 20:
            break
        
        U_c = [[0] * max_rank for _ in range(n * m)]
        V_c = [[0] * max_rank for _ in range(m * p)]
        W_c = [[0] * max_rank for _ in range(p * n)]
        
        col = build_tiling(U_c, V_c, W_c, i_blocks, j_blocks, k_blocks, n, m, p)
        
        U_c = [row[:col] for row in U_c]
        V_c = [row[:col] for row in V_c]
        W_c = [row[:col] for row in W_c]
        candidates.append((U_c, V_c, W_c))
    
    # Fallback: pure schoolbook
    U4 = [[0] * (n * m * p) for _ in range(n * m)]
    V4 = [[0] * (n * m * p) for _ in range(m * p)]
    W4 = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U4[i * m + j][col] = 1
                V4[j * p + k][col] = 1
                W4[k * n + i][col] = 1
                col += 1
    candidates.append((U4, V4, W4))
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(len(U[0])):
                        u_val = U[i * m + j][r]
                        v_val = V[j * p + k][r]
                        w_val = W[k * n + i][r]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            total += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                    if total != 1:
                        return False
        return True
    
    def build_tensor_matrix(U, V, W, n, m, p):
        r = len(U[0])
        M = np.zeros((n * m * p, r), dtype=object)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        M[idx, r_idx] = U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    idx += 1
        return M
    
    def exact_rank_and_pivots(M, deadline):
        rows = M.shape[0]
        cols = M.shape[1]
        A = [list(M[row]) for row in range(rows)]
        pivot_cols = []
        current_row = 0
        
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                break
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1
        
        return pivot_cols
    
    def reduce_rank(U, V, W, n, m, p, deadline):
        """Try to reduce the rank of a decomposition via column elimination."""
        r = len(U[0])
        
        # Build the tensor matrix
        M = build_tensor_matrix(U, V, W, n, m, p)
        
        # Quick SVD check
        M_float = np.array(M, dtype=float)
        try:
            sv = np.linalg.svd(M_float, compute_uv=False)
            tol = max(M_float.shape) * np.finfo(float).eps * (sv[0] if sv[0] > 0 else 1.0)
            num_rank = int(np.sum(sv > tol))
        except:
            num_rank = r
        
        if num_rank >= r:
            return U, V, W, r
        
        # Exact rank computation
        pivot_cols = exact_rank_and_pivots(M, deadline)
        exact_rank = len(pivot_cols)
        
        if exact_rank >= r:
            return U, V, W, r
        
        # Keep non-pivot columns (these are the linearly independent ones)
        # Actually, we should keep the pivot columns, not remove them
        # The pivot columns span the column space, so we keep them
        keep_cols = pivot_cols
        
        if len(keep_cols) >= r:
            return U, V, W, r
        
        U_new = [[U[i][c] for c in keep_cols] for i in range(n * m)]
        V_new = [[V[i][c] for c in keep_cols] for i in range(m * p)]
        W_new = [[W[i][c] for c in keep_cols] for i in range(p * n)]
        
        # Verify the reduced decomposition
        if verify(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new, len(keep_cols)
        
        return U, V, W, r
    
    best_result = None
    best_rank = float('inf')
    
    # Sort candidates by rank (smallest first)
    candidates.sort(key=lambda x: len(x[0][0]))
    
    for U_c, V_c, W_c in candidates:
        if time.time() > deadline - 5:
            break
        
        r = len(U_c[0])
        
        if r >= best_rank and best_result is not None:
            continue
        
        # Verify correctness
        if not verify(U_c, V_c, W_c, n, m, p):
            continue
        
        if r < best_rank:
            best_result = {"U": U_c, "V": V_c, "W": W_c}
            best_rank = r
        
        # Try to reduce rank via column elimination
        if time.time() < deadline - 10:
            U_red, V_red, W_red, r_red = reduce_rank(U_c, V_c, W_c, n, m, p, deadline)
            if r_red < best_rank:
                best_result = {"U": U_red, "V": V_red, "W": W_red}
                best_rank = r_red
    
    if best_result is not None:
        return best_result
    
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
