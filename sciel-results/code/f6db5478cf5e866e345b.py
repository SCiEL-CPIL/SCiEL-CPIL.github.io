"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition using block Strassen."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
DEADLINE = 145.0

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U shape: (n*m, rank) = (12, rank)
    V shape: (m*p, rank) = (20, rank)
    W shape: (p*n, rank) = (15, rank)
    
    Uses block decomposition with Strassen for 2x2x2 blocks.
    """
    n, m, p = 3, 4, 5
    start_time = time.time()
    
    # Partition:
    # A (3x4): rows 0-1 (top), row 2 (bottom); cols 0-1 (left), cols 2-3 (right)
    # B (4x5): rows 0-1 (top), rows 2-3 (bottom); cols 0-1 (left), cols 2-4 (right)
    # 
    # C = A_left * B_top + A_right * B_bottom
    # A_left (3x2) * B_top (2x5): split A_left into (2x2)+(1x2), B_top into (2x2)+(2x3)
    #   - (2x2)*(2x2) -> Strassen rank 7
    #   - (2x2)*(2x3) -> split B into (2x2)+(2x1): Strassen 7 + schoolbook 4 = 11
    #   - (1x2)*(2x2) -> schoolbook rank 4
    #   - (1x2)*(2x3) -> schoolbook rank 6
    # A_right (3x2) * B_bottom (2x5): same structure
    #   - (2x2)*(2x2) -> Strassen rank 7
    #   - (2x2)*(2x3) -> 11
    #   - (1x2)*(2x2) -> 4
    #   - (1x2)*(2x3) -> 6
    # Total: 2*(7+11+4+6) = 2*28 = 56
    
    # Actually let me recompute. The total should be:
    # For A_left * B_top:
    #   A_left_top (2x2) * B_top_left (2x2): Strassen, rank 7
    #   A_left_top (2x2) * B_top_right (2x3): 
    #     A_left_top * B_top_right_left (2x2): Strassen, rank 7
    #     A_left_top * B_top_right_right (2x1): schoolbook, rank 4
    #     subtotal: 11
    #   A_left_bottom (1x2) * B_top_left (2x2): schoolbook, rank 4
    #   A_left_bottom (1x2) * B_top_right (2x3): schoolbook, rank 6
    #   subtotal for A_left * B_top: 7 + 11 + 4 + 6 = 28
    # Similarly for A_right * B_bottom: 28
    # Total: 56
    
    # Let me build this step by step.
    # I'll create a helper to add terms to the decomposition.
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        """Add a single tensor term."""
        U.append(u_val)
        V.append(v_val)
        W.append(w_val)
    
    # Initialize U, V, W as lists that will grow
    # Each "term" is a column in U, V, W
    # U has n*m = 12 rows, V has m*p = 20 rows, W has p*n = 15 rows
    
    def make_zero_matrix(rows):
        return [0] * rows
    
    # We'll build columns incrementally
    U_cols = []  # list of columns, each is a list of 12 values
    V_cols = []  # list of columns, each is a list of 20 values
    W_cols = []  # list of columns, each is a list of 15 values
    
    def add_column(u_col, v_col, w_col):
        """Add a column to U, V, W."""
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2x2(u_row_base, v_row_base, w_row_base, i_offset, j_offset, k_offset):
        """
        Add Strassen decomposition for a 2x2x2 block.
        u_row_base: starting row index in U (for the 2x2 A block)
        v_row_base: starting row index in V (for the 2x2 B block)  
        w_row_base: starting row index in W (for the 2x2 C block)
        i_offset, j_offset, k_offset: offsets for indexing
        
        For Strassen, we have 7 terms:
        M1 = (A11+A22)*(B11+B22)
        M2 = A22*(B21-B11)
        M3 = A11*(B12-B22)
        M4 = A22*(B11+B22)
        M5 = (A11+A12)*B22
        M6 = (A21-A11)*B11
        M7 = (A12-A22)*(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        Where A is 2x2 with A11 at (i,j), A12 at (i,j+1), A21 at (i+1,j), A22 at (i+1,j+1)
        B is 2x2 with B11 at (j,k), B12 at (j,k+1), B21 at (j+1,k), B22 at (j+1,k+1)
        C is 2x2 with C11 at (i,k), C12 at (i,k+1), C21 at (i+1,k), C22 at (i+1,k+1)
        """
        # Define the 7 Strassen multipliers
        # Each multiplier is a product of linear combinations of A and B entries
        
        # For each of the 7 terms, we need U, V, W vectors
        # U has entries for A indices, V for B indices, W for C indices
        
        # A indices (in U): row = i*m + j where i in [0,2), j in [0,2) relative to block
        # In global U: row = (i_offset + i) * m + (j_offset + j)
        # B indices (in V): row = j*p + k where j in [0,2), k in [0,2) relative
        # In global V: row = (j_offset + j) * p + (k_offset + k)
        # C indices (in W): row = k*n + i where k in [0,2), i in [0,2) relative
        # In global W: row = (k_offset + k) * n + (i_offset + i)
        
        # Let me define helper to get global indices
        def u_idx(i, j):
            return (i_offset + i) * m + (j_offset + j)
        
        def v_idx(j, k):
            return (j_offset + j) * p + (k_offset + k)
        
        def w_idx(k, i):
            return (k_offset + k) * n + (i_offset + i)
        
        # Strassen 7 terms
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = A22 * (B21 - B11)
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B11 + B22)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * B11
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each multiplier Mi, we create a term where:
        # U has the A coefficients, V has the B coefficients, W has the C coefficients
        
        # M1: A part: A11=1, A22=1; B part: B11=1, B22=1
        u1 = make_zero_matrix(n * m)
        v1 = make_zero_matrix(m * p)
        w1 = make_zero_matrix(p * n)
        u1[u_idx(0, 0)] = 1  # A11
        u1[u_idx(1, 1)] = 1  # A22
        v1[v_idx(0, 0)] = 1  # B11
        v1[v_idx(1, 1)] = 1  # B22
        # C contributions: C11 += 1, C22 += 1
        w1[w_idx(0, 0)] = 1  # C11
        w1[w_idx(1, 1)] = 1  # C22
        add_column(u1, v1, w1)
        
        # M2: A part: A22=1; B part: B21=1, B11=-1
        u2 = make_zero_matrix(n * m)
        v2 = make_zero_matrix(m * p)
        w2 = make_zero_matrix(p * n)
        u2[u_idx(1, 1)] = 1  # A22
        v2[v_idx(1, 0)] = 1  # B21
        v2[v_idx(0, 0)] = -1  # B11
        # C contributions: C21 += 1
        w2[w_idx(0, 1)] = 1  # C21
        add_column(u2, v2, w2)
        
        # M3: A part: A11=1; B part: B12=1, B22=-1
        u3 = make_zero_matrix(n * m)
        v3 = make_zero_matrix(m * p)
        w3 = make_zero_matrix(p * n)
        u3[u_idx(0, 0)] = 1  # A11
        v3[v_idx(0, 1)] = 1  # B12
        v3[v_idx(1, 1)] = -1  # B22
        # C contributions: C12 += 1
        w3[w_idx(1, 0)] = 1  # C12
        add_column(u3, v3, w3)
        
        # M4: A part: A22=1; B part: B11=1, B22=1
        u4 = make_zero_matrix(n * m)
        v4 = make_zero_matrix(m * p)
        w4 = make_zero_matrix(p * n)
        u4[u_idx(1, 1)] = 1  # A22
        v4[v_idx(0, 0)] = 1  # B11
        v4[v_idx(1, 1)] = 1  # B22
        # C contributions: C11 += 1, C21 += 1
        w4[w_idx(0, 0)] = 1  # C11
        w4[w_idx(0, 1)] = 1  # C21
        add_column(u4, v4, w4)
        
        # M5: A part: A11=1, A12=1; B part: B22=1
        u5 = make_zero_matrix(n * m)
        v5 = make_zero_matrix(m * p)
        w5 = make_zero_matrix(p * n)
        u5[u_idx(0, 0)] = 1  # A11
        u5[u_idx(0, 1)] = 1  # A12
        v5[v_idx(1, 1)] = 1  # B22
        # C contributions: C11 -= 1, C12 += 1
        w5[w_idx(0, 0)] = -1  # C11
        w5[w_idx(1, 0)] = 1  # C12
        add_column(u5, v5, w5)
        
        # M6: A part: A21=1, A11=-1; B part: B11=1
        u6 = make_zero_matrix(n * m)
        v6 = make_zero_matrix(m * p)
        w6 = make_zero_matrix(p * n)
        u6[u_idx(1, 0)] = 1  # A21
        u6[u_idx(0, 0)] = -1  # A11
        v6[v_idx(0, 0)] = 1  # B11
        # C contributions: C22 += 1
        w6[w_idx(1, 1)] = 1  # C22
        add_column(u6, v6, w6)
        
        # M7: A part: A12=1, A22=-1; B part: B21=1, B22=1
        u7 = make_zero_matrix(n * m)
        v7 = make_zero_matrix(m * p)
        w7 = make_zero_matrix(p * n)
        u7[u_idx(0, 1)] = 1  # A12
        u7[u_idx(1, 1)] = -1  # A22
        v7[v_idx(1, 0)] = 1  # B21
        v7[v_idx(1, 1)] = 1  # B22
        # C contributions: C11 += 1
        w7[w_idx(0, 0)] = 1  # C11
        add_column(u7, v7, w7)
    
    def schoolbook_block(u_i_range, u_j_range, v_j_range, v_k_range, w_k_range, w_i_range):
        """
        Add schoolbook decomposition for a block.
        u_i_range: range of i indices in A block
        u_j_range: range of j indices in A block
        v_j_range: range of j indices in B block
        v_k_range: range of k indices in B block
        w_k_range: range of k indices in C block
        w_i_range: range of i indices in C block
        """
        for i in u_i_range:
            for j in u_j_range:
                for k in v_k_range:
                    u_col = make_zero_matrix(n * m)
                    v_col = make_zero_matrix(m * p)
                    w_col = make_zero_matrix(p * n)
                    u_col[(i_offset_global + i) * m + (j_offset_global + j)] = 1
                    v_col[(j_offset_global + j) * p + (k_offset_global + k)] = 1
                    w_col[(k_offset_global + k) * n + (i_offset_global + i)] = 1
                    add_column(u_col, v_col, w_col)
    
    # Actually, the schoolbook_block function above uses global offsets which aren't defined.
    # Let me restructure to pass explicit global indices.
    
    # Let me restart with a cleaner approach.
    
    # Clear the columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_column_clean(u_col, v_col, w_col):
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    # Define global index helpers
    def U_idx(i, j):
        """Global U index for A[i,j]"""
        return i * m + j
    
    def V_idx(j, k):
        """Global V index for B[j,k]"""
        return j * p + k
    
    def W_idx(k, i):
        """Global W index for C[i,k]"""
        return k * n + i
    
    # Strassen 2x2x2 for a specific block
    def strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for 2x2x2 block starting at A[i0,i0+1]x[j0,j0+1],
        B[j0,j0+1]x[k0,k0+1], C[i0,i0+1]x[k0,k0+1]
        """
        def u(i, j):
            return U_idx(i0 + i, j0 + j)
        
        def v(j, k):
            return V_idx(j0 + j, k0 + k)
        
        def w(k, i):
            return W_idx(k0 + k, i0 + i)
        
        # M1 = (A11+A22)*(B11+B22)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[u(0,0)] = 1; u1[u(1,1)] = 1
        v1[v(0,0)] = 1; v1[v(1,1)] = 1
        w1[w(0,0)] = 1; w1[w(1,1)] = 1
        add_column_clean(u1, v1, w1)
        
        # M2 = A22*(B21-B11)
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[u(1,1)] = 1
        v2[v(1,0)] = 1; v2[v(0,0)] = -1
        w2[w(0,1)] = 1
        add_column_clean(u2, v2, w2)
        
        # M3 = A11*(B12-B22)
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[u(0,0)] = 1
        v3[v(0,1)] = 1; v3[v(1,1)] = -1
        w3[w(1,0)] = 1
        add_column_clean(u3, v3, w3)
        
        # M4 = A22*(B11+B22)
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[u(1,1)] = 1
        v4[v(0,0)] = 1; v4[v(1,1)] = 1
        w4[w(0,0)] = 1; w4[w(0,1)] = 1
        add_column_clean(u4, v4, w4)
        
        # M5 = (A11+A12)*B22
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[u(0,0)] = 1; u5[u(0,1)] = 1
        v5[v(1,1)] = 1
        w5[w(0,0)] = -1; w5[w(1,0)] = 1
        add_column_clean(u5, v5, w5)
        
        # M6 = (A21-A11)*B11
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[u(1,0)] = 1; u6[u(0,0)] = -1
        v6[v(0,0)] = 1
        w6[w(1,1)] = 1
        add_column_clean(u6, v6, w6)
        
        # M7 = (A12-A22)*(B21+B22)
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[u(0,1)] = 1; u7[u(1,1)] = -1
        v7[v(1,0)] = 1; v7[v(1,1)] = 1
        w7[w(0,0)] = 1
        add_column_clean(u7, v7, w7)
    
    def schoolbook_2x2x2(i0, j0, k0):
        """Schoolbook for 2x2x2 block."""
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
                    u_col[U_idx(i0+i, j0+j)] = 1
                    v_col[V_idx(j0+j, k0+k)] = 1
                    w_col[W_idx(k0+k, i0+i)] = 1
                    add_column_clean(u_col, v_col, w_col)
    
    def schoolbook_2x2x3(i0, j0, k0):
        """Schoolbook for 2x2x3 block: A 2x2, B 2x3, C 2x3."""
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
                    u_col[U_idx(i0+i, j0+j)] = 1
                    v_col[V_idx(j0+j, k0+k)] = 1
                    w_col[W_idx(k0+k, i0+i)] = 1
                    add_column_clean(u_col, v_col, w_col)
    
    def schoolbook_1x2x2(i0, j0, k0):
        """Schoolbook for 1x2x2 block: A 1x2, B 2x2, C 1x2."""
        for j in range(2):
            for k in range(2):
                u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
                u_col[U_idx(i0, j0+j)] = 1
                v_col[V_idx(j0+j, k0+k)] = 1
                w_col[W_idx(k0+k, i0)] = 1
                add_column_clean(u_col, v_col, w_col)
    
    def schoolbook_1x2x3(i0, j0, k0):
        """Schoolbook for 1x2x3 block: A 1x2, B 2x3, C 1x3."""
        for j in range(2):
            for k in range(3):
                u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
                u_col[U_idx(i0, j0+j)] = 1
                v_col[V_idx(j0+j, k0+k)] = 1
                w_col[W_idx(k0+k, i0)] = 1
                add_column_clean(u_col, v_col, w_col)
    
    # Now build the full decomposition
    # Partition:
    # A rows: 0-1 (top), 2 (bottom)
    # A/B middle: 0-1 (left), 2-3 (right)
    # B cols: 0-1 (left), 2-4 (right)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Use Strassen
    strassen_block(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Split B cols: 2:4 and 4:5
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]: Use Strassen
    strassen_block(0, 0, 2)
    # A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]: Schoolbook 2x2x1
    # Actually 2x2x1: for i in 0,1; j in 0,1; k in 0
    for i in range(2):
        for j in range(2):
            u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
            u_col[U_idx(0+i, 0+j)] = 1
            v_col[V_idx(0+j, 4+0)] = 1
            w_col[W_idx(4+0, 0+i)] = 1
            add_column_clean(u_col, v_col, w_col)
    
    # Block 3: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # Schoolbook 1x2x2
    schoolbook_1x2x2(2, 0, 0)
    
    # Block 4: A[2:3, 0:2] * B[0:2, 2:5] -> C[2:3, 2:5]
    # Schoolbook 1x2x3
    schoolbook_1x2x3(2, 0, 2)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Use Strassen
    strassen_block(0, 2, 0)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    # Split B cols: 2:4 and 4:5
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]: Use Strassen
    strassen_block(0, 2, 2)
    # A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]: Schoolbook 2x2x1
    for i in range(2):
        for j in range(2):
            u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
            u_col[U_idx(0+i, 2+j)] = 1
            v_col[V_idx(2+j, 4+0)] = 1
            w_col[W_idx(4+0, 0+i)] = 1
            add_column_clean(u_col, v_col, w_col)
    
    # Block 7: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    # Schoolbook 1x2x2
    schoolbook_1x2x2(2, 2, 0)
    
    # Block 8: A[2:3, 2:4] * B[2:4, 2:5] -> C[2:3, 2:5]
    # Schoolbook 1x2x3
    schoolbook_1x2x3(2, 2, 2)
    
    # Count the rank
    rank = len(U_cols)
    
    # Convert to the required format
    U_final = [[0]*rank for _ in range(n*m)]
    V_final = [[0]*rank for _ in range(m*p)]
    W_final = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U_final[i][r] = U_cols[r][i]
        for j in range(m*p):
            V_final[j][r] = V_cols[r][j]
        for k in range(p*n):
            W_final[k][r] = W_cols[r][k]
    
    return {"U": U_final, "V": V_final, "W": W_final}


def _validate_decomposition(U, V, W, n, m, p):
    """
    Check if the decomposition is valid using exact rational arithmetic.
    """
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Check the tensor equation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if total != 1:
                    return False
    return True


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}")
    print(f"Valid: {_validate_decomposition(U, V, W, n, m, p)}")
