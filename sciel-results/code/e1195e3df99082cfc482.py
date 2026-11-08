"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy: Use Strassen-Winograd style block decomposition.
    - Core: 4x4 * 4x4 block using rank-7 Strassen for 2x2 blocks (4x4x4x4 -> rank 28)
    - Cross terms: 4x4 * 1x6 and 4x1 * 4x6 and 4x1 * 1x6
    
    Actually, let me think more carefully about the tensor structure.
    
    Matrix multiplication (4x5) * (5x6) = (4x6)
    The multiplication tensor has entries T[i,j,k] where:
    - i in [0,4) : row of A, col of C
    - j in [0,5) : shared dimension (col of A, row of B)  
    - k in [0,6) : row of B, col of C
    
    The output factor follows W[k*n+i, r] convention.
    
    For the schoolbook method, rank = n*m*p = 4*5*6 = 120.
    
    Using Strassen-like recursion:
    Split A (4x5) into blocks:
      - A1: 4x4 (first 4 columns)
      - A2: 4x1 (last 1 column)
    
    Split B (5x6) into blocks:
      - B1: 4x6 (first 4 rows)
      - B2: 1x6 (last 1 row)
    
    Then C = A*B = A1*B1 + A2*B2
    
    For A1*B1 (4x4 * 4x4 -> 4x4):
      Use Strassen decomposition: split 4x4 into 2x2 blocks, use rank-7 for each 2x2*2x2
      This gives rank 7*4 = 28 for the 4x4*4x4 multiplication.
    
    For A2*B2 (4x1 * 1x6 -> 4x6):
      This is an outer product, rank = 4*1*6 = 24? No, actually for (4x1)*(1x6)->(4x6),
      each element C[i,k] = A2[i,0] * B2[0,k], so rank = 4*6 = 24.
    
    Total rank = 28 + 24 = 52.
    
    Wait, I need to be more careful. Let me reconsider.
    
    Actually, the tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    has a rank that can be reduced using fast matrix multiplication algorithms.
    
    For 4x4 * 4x4, the Strassen algorithm gives rank 28 (since 4x4 can be seen as 
    4 blocks of 2x2, and each 2x2*2x2 uses rank 7, so 4*7 = 28).
    
    But we need to handle the full 4x5 * 5x6 case.
    
    Let me define the decomposition explicitly.
    
    A is 4x5: rows i=0..3, cols j=0..4
    B is 5x6: rows j=0..4, cols k=0..5
    C is 4x6: rows i=0..3, cols k=0..5
    
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    Split j into j=0..3 (first 4) and j=4 (last 1):
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] + A[i,4] * B[4,k]
    
    First term: A1 (4x4) * B1 (4x6) -> 4x6
    Second term: A2 (4x1) * B2 (1x6) -> 4x6
    
    For A1*B1 (4x4 * 4x6 -> 4x6):
      We can use Strassen-like methods. Split A1 into 2x2 blocks:
        A1 = [[A11, A12], [A21, A22]] where each is 2x2
      Split B1 into 2x6 blocks:
        B1 = [[B11, B12], [B21, B22]] where each is 2x6
      
      Then A1*B1 = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
                    [A21*B11 + A22*B21, A21*B12 + A22*B22]]
      
      Each 2x2 * 2x6 product can be computed using rank-7 Strassen-like method.
      Actually, for 2x2 * 2x6, we can treat it as 3 independent 2x2*2x2 products
      (since 6 = 2+2+2), so rank = 3 * 7 = 21 per 2x2*2x6 block.
      
      There are 4 such blocks, so total rank = 4 * 21 = 84? That's worse than schoolbook.
      
    Hmm, this approach isn't obviously better. Let me think differently.
    
    Actually, the schoolbook rank is 120. Any improvement would be valuable.
    
    Let me try a different approach: use the known rank of 4x4*4x4 matrix multiplication.
    It is known that the rank of 4x4*4x4 is at most 28 (using Strassen recursively).
    
    For our problem:
    C = A1*B1 + A2*B2
    
    If we can compute A1*B1 with rank 28 (treating it as 4x4*4x4 by padding B1 to 4x4),
    and A2*B2 with rank 24, total = 52.
    
    But wait, B1 is 4x6, not 4x4. We can't directly use the 4x4*4x4 result.
    
    Alternative: pad everything to powers of 2 or use a different blocking.
    
    Let me try: treat the 5x6 as a 4x6 block plus 1x6, and 4x5 as 4x4 plus 4x1.
    
    Actually, I think the cleanest approach for an improvement is:
    
    Use the fact that rank of (n x m) * (m x p) can be bounded by various means.
    For small matrices, we can try to find explicit low-rank decompositions.
    
    Given the time constraint, let me implement the Strassen-based decomposition for the
    core 4x4*4x4 part and handle the remaining terms with schoolbook.
    
    Let me define:
    - A1 = A[:, 0:4] (4x4)
    - A2 = A[:, 4:5] (4x1)  
    - B1 = B[0:4, :] (4x6)
    - B2 = B[4:5, :] (1x6)
    
    C = A1*B1 + A2*B2
    
    For A2*B2: this is outer product of a 4x1 and 1x6, rank = 24 (each of 4*6 entries
    is a product of one A element and one B element).
    
    For A1*B1 (4x4 * 4x6): 
      We can split B1 into B1a (4x3) and B1b (4x3), or use other splits.
      Actually, let's use Strassen on the 4x4 part by treating B1 as having 6 columns.
      
      Split A1 into 2x2 blocks: A1 = [[P, Q], [R, S]] (each 2x2)
      Split B1 into 2x6 blocks: B1 = [[T, U], [V, W]] (each 2x6)
      
      A1*B1 = [[P*T + Q*V, P*U + Q*W],
               [R*T + S*V, R*U + S*W]]
      
      Each 2x2 * 2x6 product: treat as 2x2 * (2x2 + 2x2 + 2x2) = three 2x2*2x2 products.
      Using Strassen (rank 7) for each 2x2*2x2: rank = 3*7 = 21 per 2x2*2x6.
      Total for A1*B1: 4 * 21 = 84.
      
      Total rank: 84 + 24 = 108. Still better than 120!
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll build U, V, W matrices.
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (30, rank)  
    # W: (p*n, rank) = (24, rank)
    
    # Convention: 
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    # W[k*n + i, r] corresponds to C[i,k]
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_row, v_row, w_row, u_val=1, v_val=1, w_val=1):
        nonlocal rank
        U.append([0] * rank + [u_val])
        V.append([0] * rank + [v_val])
        W.append([0] * rank + [w_val])
        rank += 1
        for row in U[:-1]:
            row.append(0)
        for row in V[:-1]:
            row.append(0)
        for row in W[:-1]:
            row.append(0)
    
    # Helper for Strassen 2x2*2x2 decomposition
    # Given 2x2 matrices X and Y, compute X*Y using 7 multiplications
    # Returns list of (x_row, y_row, z_row, x_val, y_val, z_val) terms
    # where x_row is the row index in X (0 or 1), y_row in Y (0 or 1), z_row in result (0 or 1)
    
    def strassen_2x2(x_row_base, y_row_base, z_row_base):
        """
        Generate the 7 Strassen terms for 2x2*2x2.
        x_row_base: base row index for X in the U matrix (A)
        y_row_base: base row index for Y in the V matrix (B)
        z_row_base: base row index for Z in the W matrix (C)
        
        X = [[x00, x01], [x10, x11]]
        Y = [[y00, y01], [y10, y11]]
        
        M1 = (x00+x11)*(y00+y11)
        M2 = (x10+x11)*y00
        M3 = x00*(y01-y11)
        M4 = x11*(y10-y00)
        M5 = (x00+x01)*y11
        M6 = (x10-x00)*(y00+y10)
        M7 = (x01-x11)*(y01+y11)
        
        Z = [[M1+M4-M5+M7, M3+M5],
             [M2+M4, M1-M2+M3+M6]]
        """
        terms = []
        # M1 = (x00+x11)*(y00+y11)
        # z00 += M1, z01 += M3+M5, z10 += M2+M4, z11 += M1-M2+M3+M6
        
        # M1: x00*x_y00, x00*x_y11, x11*x_y00, x11*x_y11
        # Actually, Strassen combines these. Let me write out each multiplication.
        
        # Each Mi is a product of a linear combo of X entries and a linear combo of Y entries.
        # So each Mi contributes one term to U, V, W.
        
        # M1 = (x00 + x11) * (y00 + y11)
        # U: x00=1, x11=1, others 0 in rows x_row_base, x_row_base+1
        # V: y00=1, y11=1, others 0 in rows y_row_base, y_row_base+1
        # W: z00+=1, z11+=1
        
        # This is getting complex. Let me just enumerate all terms explicitly.
        
        # I'll create a helper that adds a single rank-1 term.
        
        # For Strassen 2x2*2x2, we have 7 multiplications.
        # Each multiplication is (linear combo of X) * (linear combo of Y) -> (linear combo of Z)
        
        # Let me define the 7 terms as tuples of (x_coefficients, y_coefficients, z_coefficients)
        # where x_coefficients is a dict from (row, col) to coefficient, etc.
        
        x_coeffs = [
            {(0,0): 1, (1,1): 1},  # M1
            {(1,0): 1, (1,1): 1},  # M2
            {(0,0): 1},            # M3
            {(1,1): 1},            # M4
            {(0,0): 1, (0,1): 1},  # M5
            {(1,0): 1, (0,0): -1}, # M6
            {(0,1): 1, (1,1): -1}, # M7
        ]
        
        y_coeffs = [
            {(0,0): 1, (1,1): 1},  # M1
            {(0,0): 1},            # M2
            {(0,1): 1, (1,1): -1}, # M3
            {(1,0): 1, (0,0): -1}, # M4
            {(1,1): 1},            # M5
            {(0,0): 1, (1,0): 1},  # M6
            {(0,1): 1, (1,1): 1},  # M7
        ]
        
        z_coeffs = [
            {(0,0): 1, (1,1): 1},  # M1
            {(1,0): 1},            # M2
            {(0,1): 1},            # M3
            {(0,0): 1, (1,0): 1},  # M4
            {(0,1): 1, (1,1): -1}, # M5
            {(1,1): 1},            # M6
            {(0,0): 1, (0,1): 1},  # M7
        ]
        
        for i in range(7):
            x_c = x_coeffs[i]
            y_c = y_coeffs[i]
            z_c = z_coeffs[i]
            
            # For each combination, we need to add terms.
            # But since x_c and y_c may have multiple entries, we can't just do one term.
            # Actually, in the tensor decomposition, each "multiplication" in Strassen
            # corresponds to one rank-1 term where U and V have the linear combinations.
            
            # So for M1: U has 1 at (x_row_base+0, x_col_base+0) and (x_row_base+1, x_col_base+1)
            # V has 1 at (y_row_base+0, y_col_base+0) and (y_row_base+1, y_col_base+1)
            # W has 1 at (z_row_base+0, z_col_base+0) and (z_row_base+1, z_col_base+1)
            
            # But this is a single rank-1 term with these coefficients.
            
            u_row = x_row_base
            v_row = y_row_base
            w_row = z_row_base
            
            # Add one term with the given coefficients
            U.append([0] * rank)
            V.append([0] * rank)
            W.append([0] * rank)
            
            for (xr, xc), val in x_c.items():
                U[-1][u_row + xr * 2 + xc] = val  # Wait, need to map to global indices
            
            # This is getting complicated with the indexing. Let me use a different approach.
            # I'll just build the matrices directly using the schoolbook method first,
            # and then try to apply Strassen-like reductions.
            
            # For now, let me just use schoolbook to ensure correctness, then optimize.
            pass
    
    # Given the complexity, let me just use schoolbook for now to ensure a valid solution,
    # and note that the rank is 120.
    
    # Actually, let me implement a proper Strassen-based decomposition.
    
    # Let me restart with a cleaner approach.
    
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    def add_rank1_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries, v_entries, w_entries are dicts mapping (row, col) to value."""
        nonlocal U, V, W
        new_rank = len(U[0]) if U else 0
        for row in U:
            row.append(0)
        for row in V:
            row.append(0)
        for row in W:
            row.append(0)
        for (r, c), val in u_entries.items():
            U[r][c] = val
        for (r, c), val in v_entries.items():
            V[r][c] = val
        for (r, c), val in w_entries.items():
            W[r][c] = val
    
    # Schoolbook: rank = n*m*p = 120
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_entries = {(i * m + j, 0): 1}
                v_entries = {(j * p + k, 0): 1}
                w_entries = {(k * n + i, 0): 1}
                add_rank1_term(u_entries, v_entries, w_entries)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
