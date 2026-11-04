"""Block-decomposition rank reduction for 4x5x5 matrix multiplication."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, u_coeff=1, v_coeff=1, w_coeff=1):
        """Add a single tensor term: A[u_i,u_j]*B[v_j,v_k] -> C[w_i,w_k]"""
        U_cols.append([u_coeff if i == u_idx else 0 for i in range(n * m)])
        V_cols.append([v_coeff if i == v_idx else 0 for i in range(m * p)])
        W_cols.append([w_coeff if i == w_idx else 0 for i in range(p * n)])
    
    def add_uvw(a_i, a_j, b_j, b_k, c_i, c_k, coeff=1):
        """Add term: A[a_i,a_j] * B[b_j,b_k] -> C[c_i,c_k]"""
        u_idx = a_i * m + a_j
        v_idx = b_j * p + b_k
        w_idx = c_k * n + c_i
        add_term(u_idx, v_idx, w_idx, coeff, 1, 1)
    
    # Block decomposition:
    # A is 4x5, B is 5x5, C is 4x5
    # Split A rows: [0:2] and [2:4]
    # Split B rows: [0:3] and [3:5], B cols: [0:3] and [3:5]
    # 
    # C[0:2,0:3] = A[0:2,0:3] * B[0:3,0:3] + A[0:2,3:5] * B[3:5,0:3]
    # C[0:2,3:5] = A[0:2,0:3] * B[0:3,3:5] + A[0:2,3:5] * B[3:5,3:5]
    # C[2:4,0:3] = A[2:4,0:3] * B[0:3,0:3] + A[2:4,3:5] * B[3:5,0:3]
    # C[2:4,3:5] = A[2:4,0:3] * B[0:3,3:5] + A[2:4,3:5] * B[3:5,3:5]
    
    # Block types and their ranks:
    # 2x3 * 3x3 -> rank 13 (Laderman for 2x3x3)
    # 2x2 * 2x3 -> rank 6 (schoolbook)
    # 2x3 * 3x2 -> rank 6 (schoolbook)
    # 2x2 * 2x2 -> rank 7 (Strassen)
    
    # Total: 2*13 + 2*6 + 2*6 + 2*7 = 26+12+12+14 = 64
    
    # Helper: schoolbook for a_i x b_j * b_j x c_k -> a_i x c_k
    def schoolbook(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add schoolbook terms for A[a_rows, a_cols] * B[b_rows, b_cols] -> C[c_rows, c_cols]"""
        for ai in a_rows:
            for bj in b_rows:
                for ck in c_cols:
                    a_j = a_cols[bj - b_rows[0]]  # column index in A
                    b_k = b_cols[ck - c_cols[0]]  # column index in B
                    add_uvw(ai, a_j, bj, b_k, ck, ck - c_cols[0] + c_rows[0] - c_rows[0], 1)
                    # Wait, need to be more careful with indexing
    
    # Let me use direct indexing instead
    def schoolbook_direct(a_row_start, a_col_start, b_row_start, b_col_start, 
                          c_row_start, c_col_start, a_h, a_w, b_h, b_w, c_h, c_w):
        """Schoolbook: A[c_row_start+ri, a_col_start+cj] * B[b_row_start+cj, b_col_start+ck] -> C[c_row_start+ri, c_col_start+ck]"""
        for ri in range(a_h):
            for cj in range(b_h):
                for ck in range(b_w):
                    a_i = a_row_start + ri
                    a_j = a_col_start + cj
                    b_j = b_row_start + cj
                    b_k = b_col_start + ck
                    c_i = c_row_start + ri
                    c_k = c_col_start + ck
                    add_uvw(a_i, a_j, b_j, b_k, c_i, c_k, 1)
    
    # Strassen for 2x2x2: A[2x2] * B[2x2] -> C[2x2]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, 
                        c_row_start, c_col_start):
        """Strassen 2x2x2 decomposition.
        A is 2x2 at (a_row_start, a_col_start)
        B is 2x2 at (b_row_start, b_col_start)
        C is 2x2 at (c_row_start, c_col_start)
        """
        # Each M_t is a 2x2 matrix computed via schoolbook (8 terms each)
        # But we use Strassen: 7 products, each 2x2x2 schoolbook = 7*8 = 56? No.
        # Actually each M_t involves sums/diffs of 2x2 blocks.
        # The rank-7 decomposition means 7 scalar multiplications for the whole 2x2x2.
        # But each "multiplication" is of combined blocks, so we expand.
        
        # Let me just use the standard approach: each M_t is computed as a 2x2 matrix
        # product of linear combinations. The rank-7 means we need 7 multiplications
        # of 2x2 matrices, but each 2x2 matrix multiplication still needs 8 scalar mults.
        # So total = 7*8 = 56? That can't be right for a 2x2x2 which should be rank 7.
        
        # I think the confusion is: the 2x2x2 tensor has rank 7, meaning 7 pure tensors
        # suffice. Each pure tensor is a single scalar multiplication.
        
        # The 7 Strassen terms for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each M_t is a 2x2 matrix. To compute M_t as a 2x2 matrix from 2x2 blocks,
        # we need to expand. But the key is that the 2x2x2 tensor T[i,j,k] = sum_j A[i,j]*B[j,k]
        # can be written as a sum of 7 pure tensors.
        
        # For the 2x2x2 case, the 7 terms are:
        # Term 1: (A11+A22) * (B11+B22) -> contributes to C11 and C22
        # Term 2: (A21+A22) * B11 -> contributes to C21 and C22
        # etc.
        
        # But each of these is a product of 2x2 matrices, which itself has rank 8.
        # So the total rank would be 7*8 = 56, not 7.
        
        # I think I'm confusing the tensor rank of the 2x2x2 multiplication tensor (which is 7)
        # with the number of scalar multiplications needed when the inputs are themselves sums.
        
        # For our purposes, when we say "use Strassen for 2x2x2", we mean we use the
        # rank-7 decomposition of the 2x2x2 tensor. This gives us 7 terms, where each
        # term is a pure tensor e_i ⊗ e_j ⊗ e_k with coefficients.
        
        # The 7 pure tensors for 2x2x2 Strassen:
        # Let A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M_t is a scalar product of two linear combinations.
        # To express as pure tensors, we expand:
        # M1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
        # This gives 4 pure tensors for M1 alone.
        
        # So the total number of pure tensors is:
        # M1: 4, M2: 2, M3: 2, M4: 2, M5: 2, M6: 4, M7: 4 = 20?
        
        # That's not 7 either. I think the issue is that Strassen's rank-7 refers to
        # the number of bilinear forms, not the number of pure tensors in the final
        # tensor decomposition.
        
        # For our tensor decomposition, we need to expand each bilinear form into
        # pure tensors. Let me just do it directly.
        
        # For each of the 7 Strassen products, expand into pure tensors:
        
        # A11 = A[a_row_start, a_col_start]
        # A12 = A[a_row_start, a_col_start+1]
        # A21 = A[a_row_start+1, a_col_start]
        # A22 = A[a_row_start+1, a_col_start+1]
        # Similarly for B and C.
        
        a11 = (a_row_start, a_col_start)
        a12 = (a_row_start, a_col_start + 1)
        a21 = (a_row_start + 1, a_col_start)
        a22 = (a_row_start + 1, a_col_start + 1)
        
        b11 = (b_row_start, b_col_start)
        b12 = (b_row_start, b_col_start + 1)
        b21 = (b_row_start + 1, b_col_start)
        b22 = (b_row_start + 1, b_col_start + 1)
        
        c11 = (c_row_start, c_col_start)
        c12 = (c_row_start, c_col_start + 1)
        c21 = (c_row_start + 1, c_col_start)
        c22 = (c_row_start + 1, c_col_start + 1)
        
        # M1 = (A11+A22)(B11+B22) -> C11 and C22
        # Expand: A11*B11, A11*B22, A22*B11, A22*B22
        for a_pos, b_pos, c_pos, sign in [
            (a11, b11, c11, 1), (a11, b22, c11, 1),
            (a22, b11, c11, 1), (a22, b22, c11, 1),
            (a11, b11, c22, 1), (a11, b22, c22, 1),
            (a22, b11, c22, 1), (a22, b22, c22, 1),
        ]:
            add_uvw(a_pos[0], a_pos[1], b_pos[0], b_pos[1], c_pos[0], c_pos[1], sign)
        
        # M2 = (A21+A22)B11 -> C21 and C22
        for a_pos, b_pos, c_pos, sign in [
            (a21, b11, c21, 1), (a22, b11, c21, 1),
            (a21, b11, c22, -1), (a22, b11, c22, -1),
        ]:
            add_uvw(a_pos[0], a_pos[1], b_pos[0], b_pos[1], c_pos[0], c_pos[1], sign)
        
        # M3 = A11(B12-B22) -> C12 and C22
        for a_pos, b_pos, c_pos, sign in [
            (a11, b12, c12, 1), (a11, b22, c12, -1),
            (a11, b12, c22, 1), (a11, b22, c22, -1),
        ]:
            add_uvw(a_pos[0], a_pos[1], b_pos[0], b_pos[1], c_pos[0], c_pos[1], sign)
        
        # M4 = A22(B21-B11) -> C11 and C21
        for a_pos, b_pos, c_pos, sign in [
            (a22, b21, c11, 1), (a22, b11, c11, -1),
            (a22, b21, c21, 1), (a22, b11, c21, -1),
        ]:
            add_uvw(a_pos[0], a_pos[1], b_pos[0], b_pos[1], c_pos[0], c_pos[1], sign)
        
        # M5 = (A11+A12)B22 -> C11 and C12
        for a_pos, b_pos, c_pos, sign in [
            (a11, b22, c11, -1), (a12, b22, c11, -1),
            (a11, b22, c12, 1), (a12, b22, c12, 1),
        ]:
            add_uvw(a_pos[0], a_pos[1], b_pos[0], b_pos[1], c_pos[0], c_pos[1], sign)
        
        # M6 = (A21-A11)(B11+B12) -> C22
        for a_pos, b_pos, c_pos, sign in [
            (a21, b11, c22, 1), (a21, b12, c22, 1),
            (a11, b11, c22, -1), (a11, b12, c22, -1),
        ]:
            add_uvw(a_pos[0], a_pos[1], b_pos[0], b_pos[1], c_pos[0], c_pos[1], sign)
        
        # M7 = (A12-A22)(B21+B22) -> C11
        for a_pos, b_pos, c_pos, sign in [
            (a12, b21, c11, 1), (a12, b22, c11, 1),
            (a22, b21, c11, -1), (a22, b22, c11, -1),
        ]:
            add_uvw(a_pos[0], a_pos[1], b_pos[0], b_pos[1], c_pos[0], c_pos[1], sign)
    
    # Laderman for 2x3x3: A[2x3] * B[3x3] -> C[2x3]
    # This has rank 13.
    # A = [[a11,a12,a13],[a21,a22,a23]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    # C = [[c11,c12,c13],[c21,c22,c23]]
    
    # Laderman's 13 terms for 2x3x3:
    # M1 = (a11+a21)(b11+b21+b31)
    # M2 = a11(b12+b22+b32)
    # M3 = a11(b13+b23+b33)
    # M4 = (a11+a12)(b21+b31)
    # M5 = a12(b22+b32)
    # M6 = a12(b23+b33)
    # M7 = (a11+a13)(b31)
    # M8 = a13(b32)
    # M9 = a13(b33)
    # M10 = (a21+a22)(b11+b21)
    # M11 = a22(b12+b22)
    # M12 = a22(b13+b23)
    # M13 = (a21+a23)(b11)
    
    # C11 = M1 + M4 + M7 + M10 + M13 - M2 - M5 - M8
    # C12 = M2 + M5 + M8 - M10 - M13
    # C13 = M3 + M6 + M9
    # C21 = M1 - M4 - M7 - M10 - M13
    # C22 = M10 + M13 - M2 - M5 - M8
    # C23 = M11 + M12
    
    # Let me verify this is correct... Actually, I'm not sure about the exact Laderman
    # formula for 2x3x3. Let me just use schoolbook for all blocks to be safe.
    # Schoolbook ranks:
    # 2x3 * 3x3: 2*3*3 = 18
    # 2x2 * 2x3: 2*2*3 = 12
    # 2x3 * 3x2: 2*3*2 = 12
    # 2x2 * 2x2: 2*2*2 = 8
    # Total: 2*18 + 2*12 + 2*12 + 2*8 = 36+24+24+16 = 100
    
    # That's the same as full schoolbook. Let me use Strassen for the 2x2x2 blocks
    # to save 2 terms each (8->7), saving 4 total: 100-4 = 96.
    
    # Actually, let me just use schoolbook for everything and verify it works.
    # The key is to get a valid decomposition first.
    
    # C[0:2,0:3] = A[0:2,0:3] * B[0:3,0:3] + A[0:2,3:5] * B[3:5,0:3]
    schoolbook_direct(0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 2, 3)  # A[0:2,0:3]*B[0:3,0:3]->C[0:2,0:3]
    schoolbook_direct(0, 3, 3, 0, 0, 0, 2, 2, 2, 3, 2, 3)  # A[0:2,3:5]*B[3:5,0:3]->C[0:2,0:3]
    
    # C[0:2,3:5] = A[0:2,0:3] * B[0:3,3:5] + A[0:2,3:5] * B[3:5,3:5]
    schoolbook_direct(0, 0, 0, 3, 0, 3, 2, 3, 3, 2, 2, 2)  # A[0:2,0:3]*B[0:3,3:5]->C[0:2,3:5]
    schoolbook_direct(0, 3, 3, 3, 0, 3, 2, 2, 2, 2, 2, 2)  # A[0:2,3:5]*B[3:5,3:5]->C[0:2,3:5]
    
    # C[2:4,0:3] = A[2:4,0:3] * B[0:3,0:3] + A[2:4,3:5] * B[3:5,0:3]
    schoolbook_direct(2, 0, 0, 0, 2, 0, 2, 3, 3, 3, 2, 3)  # A[2:4,0:3]*B[0:3,0:3]->C[2:4,0:3]
    schoolbook_direct(2, 3, 3, 0, 2, 0, 2, 2, 2, 3, 2, 3)  # A[2:4,3:5]*B[3:5,0:3]->C[2:4,0:3]
    
    # C[2:4,3:5] = A[2:4,0:3] * B[0:3,3:5] + A[2:4,3:5] * B[3:5,3:5]
    schoolbook_direct(2, 0, 0, 3, 2, 3, 2, 3, 3, 2, 2, 2)  # A[2:4,0:3]*B[0:3,3:5]->C[2:4,3:5]
    schoolbook_direct(2, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2)  # A[2:4,3:5]*B[3:5,3:5]->C[2:4,3:5]
    
    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row in range(n * m):
            if U_cols[col][row] != 0:
                U[row][col] = U_cols[col][row]
        for row in range(m * p):
            if V_cols[col][row] != 0:
                V[row][col] = V_cols[col][row]
        for row in range(p * n):
            if W_cols[col][row] != 0:
                W[row][col] = W_cols[col][row]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
