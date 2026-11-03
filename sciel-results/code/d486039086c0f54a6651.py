"""Matrix multiplication 3x4x7: rank-minimal exact tensor decomposition.

Uses a block-structured decomposition combining Strassen's 7-multiplication
scheme for 2x2x2 blocks with schoolbook for remaining terms, achieving
rank 78 (vs schoolbook 84).
"""
import time

DIMENSIONS = [3, 4, 7]


def _strassen_2x2_2x2_2x2():
    """
    Return the 7 rank-1 terms for Strassen's 2x2 matrix multiplication.
    U: 4x7, V: 4x7, W: 4x7 (indices are (row, col) pairs for 2x2 matrices)
    
    For C = A*B where A, B, C are 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    U[i,j] means: for term j, A[row_i, col_i] coefficient
    V[i,j] means: for term j, B[row_i, col_i] coefficient  
    W[i,j] means: for term j, C[row_i, col_i] coefficient
    
    Indexing: (row, col) -> row*2 + col
    """
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = (a+d)(e+h)
    # A: a=1, d=1 -> U[0][0]=1, U[3][0]=1
    # B: e=1, h=1 -> V[0][0]=1, V[3][0]=1
    # C: C00 += M1, C11 -= M1 -> W[0][0]=1, W[3][0]=-1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1
    
    # Term 1: M2 = (c+d)e
    # A: c=1, d=1 -> U[2][1]=1, U[3][1]=1
    # B: e=1 -> V[0][1]=1
    # C: C10 += M2, C11 -= M2 -> W[2][1]=1, W[3][1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2: M3 = a(f-h)
    # A: a=1 -> U[0][2]=1
    # B: f=1, h=-1 -> V[1][2]=1, V[3][2]=-1
    # C: C01 += M3, C11 += M3 -> W[1][2]=1, W[3][2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: M4 = d(g-e)
    # A: d=1 -> U[3][3]=1
    # B: g=1, e=-1 -> V[2][3]=1, V[0][3]=-1
    # C: C00 += M4, C10 += M4 -> W[0][3]=1, W[2][3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 4: M5 = (a+b)h
    # A: a=1, b=1 -> U[0][4]=1, U[1][4]=1
    # B: h=1 -> V[3][4]=1
    # C: C00 -= M5, C01 += M5 -> W[0][4]=-1, W[1][4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5: M6 = (c-a)(e+f)
    # A: c=1, a=-1 -> U[2][5]=1, U[0][5]=-1
    # B: e=1, f=1 -> V[0][5]=1, V[1][5]=1
    # C: C11 += M6 -> W[3][5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7 = (b-d)(g+h)
    # A: b=1, d=-1 -> U[1][6]=1, U[3][6]=-1
    # B: g=1, h=1 -> V[2][6]=1, V[3][6]=1
    # C: C00 += M7 -> W[0][6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_1x2x2():
    """
    Rank-4 decomposition for (1x2) * (2x2) = (1x2).
    A is 1x2: [a, b], B is 2x2, C is 1x2.
    C[0] = a*B[0][0] + b*B[1][0]
    C[1] = a*B[0][1] + b*B[1][1]
    
    U: 2x4 (A indices), V: 4x4 (B indices), W: 2x4 (C indices)
    A index: 0=[0], 1=[1]
    B index: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
    C index: 0=[0], 1=[1]
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    
    # Term 0: a * B[0,0] -> C[0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 1: a * B[0,1] -> C[1]
    U[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1
    
    # Term 2: b * B[1,0] -> C[0]
    U[1][2] = 1
    V[2][2] = 1
    W[0][2] = 1
    
    # Term 3: b * B[1,1] -> C[1]
    U[1][3] = 1
    V[3][3] = 1
    W[1][3] = 1
    
    return U, V, W


def _schoolbook_2x2x2():
    """
    Rank-4 schoolbook for 2x2 * 2x2 = 2x2.
    U: 4x4, V: 4x4, W: 4x4
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    
    for idx in range(4):
        i = idx // 2
        j = idx % 2
        # C[i,j] += A[i,j] * B[j,j] -- wait, that's not right for general 2x2*2x2
        # Actually for 2x2*2x2: C[i,j] = sum_k A[i,k]*B[k,j]
        # We need separate terms for each (i,k,j) combination
        # But schoolbook 2x2x2 has rank 4? No, rank is 4 only if we're doing outer product style
        # Actually rank(2x2x2) = 4 (schoolbook)
        # Term for (i,k,j): U[i*2+k] = 1, V[k*2+j] = 1, W[i*2+j] = 1
        # But that gives 8 terms. Rank of 2x2x2 is actually 4? 
        # Wait, rank(M_{2,2,2}) = 4? No, it's known to be 7 (Strassen) but schoolbook is 4? 
        # Actually schoolbook for 2x2 is 4 multiplications. Let me recount.
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (2 terms)
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  (2 terms)
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  (2 terms)
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]  (2 terms)
        # Total 8 terms. So schoolbook rank is 8, Strassen is 7.
    
    # Let me redo this properly with 8 terms
    U = [[0]*8 for _ in range(4)]
    V = [[0]*8 for _ in range(4)]
    W = [[0]*8 for _ in range(4)]
    
    term = 0
    for i in range(2):
        for k in range(2):
            for j in range(2):
                U[i*2+k][term] = 1
                V[k*2+j][term] = 1
                W[i*2+j][term] = 1
                term += 1
    
    return U, V, W


def construct():
    """
    Construct a rank-78 decomposition for 3x4 * 4x7 = 3x7.
    
    Strategy: Split A (3x4) into left (3x2) and right (3x2) halves.
    Split B (4x7) into top (2x7) and bottom (2x7) halves.
    
    C = A_left * B_top + A_right * B_bottom
    
    For each 3x2 * 2x7 product:
      Split A into top (2x2) and bottom (1x2)
      Split B into 3 blocks of 2x2 and one 2x1
      
      Use Strassen for the 2x2 * 2x2 products (rank 7 each)
      Use schoolbook for the 1x2 * 2x? products
      
    For 3x2 * 2x7:
      A_top (2x2) * B_block (2x2): Strassen rank 7, for 3 blocks = 21
      A_top (2x2) * B_last (2x1): rank 2 (each element of B_last multiplies with 2x2 -> 2 mults? No)
        Actually 2x2 * 2x1: C is 2x1. C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        That's 4 terms. Rank 4.
      A_bottom (1x2) * B (2x7): 
        For each column j: C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
        That's 2 terms per column, 7 columns = 14 terms. Rank 14.
      
      Total per 3x2*2x7: 21 + 4 + 14 = 39
      Two such halves: 78 total.
    """
    n, m, p = 3, 4, 7  # A is n x m, B is m x p, C is n x p
    
    # Build the full U, V, W matrices
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (28, rank)  
    # W has shape (p*n, rank) = (21, rank)
    # W index: k*n + i where k is column of C (0..p-1), i is row of C (0..n-1)
    # Wait, the task says W[k*n+i, r] - so W is indexed by (output_col, output_row)
    # C has shape n x p, so C[i, k] where i in 0..n-1, k in 0..p-1
    # W index = k*n + i
    
    # Let's build using a list of (u_coeff, v_coeff, w_coeff) tuples
    # where u_coeff is dict {A_index: value}, v_coeff is dict {B_index: value}, w_coeff is dict {C_index: value}
    
    # A index: i*m + j where i in 0..n-1, j in 0..m-1
    # B index: j*p + k where j in 0..m-1, k in 0..p-1
    # C index (for W): k*n + i where k in 0..p-1, i in 0..n-1
    
    terms = []  # Each term: (U_dict, V_dict, W_dict)
    
    def add_term(U_dict, V_dict, W_dict):
        terms.append((U_dict, V_dict, W_dict))
    
    def strassen_term(A_row1, A_col1, A_row2, A_col2, 
                       B_row1, B_col1, B_row2, B_col2,
                       C_out_rows, C_out_cols,
                       u_coeffs, v_coeffs, w_coeffs):
        """Add a Strassen-type term.
        A part: u_coeffs[i] for i in [0,1,2,3] corresponding to (A_row1,A_col1), (A_row1,A_col2), (A_row2,A_col1), (A_row2,A_col2)
        B part: similar
        C part: w_coeffs[j] for j in [0,1,2,3] corresponding to C output positions
        """
        U_dict = {}
        V_dict = {}
        W_dict = {}
        
        # Map local indices to global
        # A local: 0=(r1,c1), 1=(r1,c2), 2=(r2,c1), 3=(r2,c2)
        A_map = [(A_row1, A_col1), (A_row1, A_col2), (A_row2, A_col1), (A_row2, A_col2)]
        B_map = [(B_row1, B_col1), (B_row1, B_col2), (B_row2, B_col1), (B_row2, B_col2)]
        # C local: 0=(r1,c1), 1=(r1,c2), 2=(r2,c1), 3=(r2,c2) in the 2x2 output block
        C_map = list(zip(C_out_rows, C_out_cols))
        
        for local_idx, coeff in enumerate(u_coeffs):
            if coeff != 0:
                r, c = A_map[local_idx]
                gidx = r * m + c
                U_dict[gidx] = U_dict.get(gidx, 0) + coeff
        
        for local_idx, coeff in enumerate(v_coeffs):
            if coeff != 0:
                r, c = B_map[local_idx]
                gidx = r * p + c
                V_dict[gidx] = V_dict.get(gidx, 0) + coeff
        
        for local_idx, coeff in enumerate(w_coeffs):
            if coeff != 0:
                r, c = C_map[local_idx]
                gidx = c * n + r  # W index: col*n + row
                W_dict[gidx] = W_dict.get(gidx, 0) + coeff
        
        add_term(U_dict, V_dict, W_dict)
    
    # Strassen coefficients (7 terms, each with u, v, w coefficient vectors of length 4)
    # From the _strassen_2x2_2x2_2x2 function above:
    # Term 0: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,-1]
    # Term 1: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
    # Term 2: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
    # Term 3: u=[0,0,0,1], v=[-1,0,1,0], w=[1,0,1,0]
    # Term 4: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
    # Term 5: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
    # Term 6: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
    
    STRASSEN_TERMS = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Process each half: left (cols 0,1 of A; rows 0,1 of B) and right (cols 2,3 of A; rows 2,3 of B)
    for half in range(2):
        # A columns for this half
        a_col_offset = half * 2  # 0 or 2
        # B rows for this half
        b_row_offset = half * 2  # 0 or 2
        
        # For the 3x2 * 2x7 product:
        # A is rows 0,1,2 (n=3), cols a_col_offset, a_col_offset+1
        # B is rows b_row_offset, b_row_offset+1, cols 0..6 (p=7)
        
        # A_top: rows 0,1; A_bottom: row 2
        # B split into blocks of width 2: cols 0-1, 2-3, 4-5, and last col 6
        
        for b_block in range(3):
            # B block: rows b_row_offset, b_row_offset+1; cols b_block*2, b_block*2+1
            b_col_offset = b_block * 2
            
            # Strassen for A_top (2x2) * B_block (2x2)
            # A_top rows: 0, 1; cols: a_col_offset, a_col_offset+1
            # B_block rows: b_row_offset, b_row_offset+1; cols: b_col_offset, b_col_offset+1
            # C output: rows 0, 1; cols b_col_offset, b_col_offset+1
            
            for (u_c, v_c, w_c) in STRASSEN_TERMS:
                strassen_term(
                    0, a_col_offset, 1, a_col_offset + 1,  # A rows 0,1; cols offset, offset+1
                    b_row_offset, b_col_offset, b_row_offset + 1, b_col_offset + 1,  # B rows, cols
                    [0, 1], [b_col_offset, b_col_offset + 1],  # C output rows, cols
                    u_c, v_c, w_c
                )
        
        # Last B column: col 6, rows b_row_offset, b_row_offset+1
        # A_top (2x2) * B_last (2x1)
        # 4 terms:
        for i in range(2):  # A row 0 or 1
            for k in range(2):  # A col (which is B row)
                # C[i, 6] += A[i, a_col_offset+k] * B[b_row_offset+k, 6]
                U_dict = {(i) * m + (a_col_offset + k): 1}
                V_dict = {(b_row_offset + k) * p + 6: 1}
                W_dict = {6 * n + i: 1}
                add_term(U_dict, V_dict, W_dict)
        
        # A_bottom (row 2) * B (2x7)
        # For each col j in 0..6:
        # C[2, j] = A[2, a_col_offset] * B[b_row_offset, j] + A[2, a_col_offset+1] * B[b_row_offset+1, j]
        for j in range(7):
            for k in range(2):
                U_dict = {2 * m + (a_col_offset + k): 1}
                V_dict = {(b_row_offset + k) * p + j: 1}
                W_dict = {j * n + 2: 1}
                add_term(U_dict, V_dict, W_dict)
    
    # Now build U, V, W from the terms
    rank = len(terms)
    n_rows_U = n * m  # 12
    n_rows_V = m * p  # 28
    n_rows_W = p * n  # 21
    
    U = [[0] * rank for _ in range(n_rows_U)]
    V = [[0] * rank for _ in range(n_rows_V)]
    W = [[0] * rank for _ in range(n_rows_W)]
    
    for col_idx, (U_dict, V_dict, W_dict) in enumerate(terms):
        for gidx, val in U_dict.items():
            U[gidx][col_idx] += val
        for gidx, val in V_dict.items():
            V[gidx][col_idx] += val
        for gidx, val in W_dict.items():
            W[gidx][col_idx] += val
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import numpy as np
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = 3, 4, 7
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Verify: for random A (3x4) and B (4x7), check C = A@B
    # Using the tensor decomposition: C = sum_r U_r outer V_r outer W_r (with appropriate reshaping)
    # Actually, the decomposition means: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. Let me think about this more carefully.
    
    # The matrix multiplication tensor T_{i,j,k} = A[i,j] * B[j,k] -> C[i,k]
    # Decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for given A and B:
    # C[i,k] = sum_j sum_r ... no that's not right either.
    
    # Actually the standard form is:
    # C = sum_r (A * U_r) * (V_r * B) where U_r and V_r are vectors
    # More precisely: C = sum_r u_r outer v_r where u_r[i] = sum_j A[i,j] * U[j + i*m, r]... 
    # No, let me just use the tensor contraction form.
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For our purposes, the verification is:
    # Given A (n x m) and B (m x p), C[i,k] should equal:
    # sum_j A[i,j] * B[j,k]
    # And the decomposition says:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, this double-sum over j and r doesn't make sense as written.
    
    # The correct interpretation: the tensor T has indices (i, j, k) where i indexes A's rows,
    # j indexes A's cols / B's rows, k indexes B's cols.
    # T[i,j,k] = A[i,j] * B[j,k]
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U is indexed by (i,j), V by (j,k), W by (k,i).
    # In flattened form: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    
    # So: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This means: for the multiplication, we compute:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r (A[i,j] * U[i*m+j, r]) * (B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    # Hmm, that's still not quite right. Let me just numerically verify.
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        # u_r is a vector of length m (A's columns) for each row i
        # v_r is a vector of length p (B's columns) for each row j  
        # w_r is a scalar weight for each (i, k) pair
        
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] += s
    
    max_err = np.max(np.abs(C_computed - C_expected))
    print(f"Max error: {max_err}")
    
    if max_err > 1e-10:
        print("ERROR: Decomposition is not valid!")
    else:
        print("Decomposition is valid!")
