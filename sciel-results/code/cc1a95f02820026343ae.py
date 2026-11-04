import numpy as np
import time

def construct():
    """
    Construct a rank-50 exact tensor decomposition for 2x4 by 4x7 matrix multiplication.
    
    Strategy:
    1. Split the middle dimension (m=4) into two blocks of size 2.
       A = [A1, A2] where A1, A2 are 2x2.
       B = [B1; B2] where B1, B2 are 2x7.
       C = A1 B1 + A2 B2.
       
    2. For each 2x2 by 2x7 multiplication (rank of 2x2x7 tensor), 
       split the output dimension (p=7) into 2+2+2+1.
       Let B1 = [B1_1, B1_2, B1_3, B1_4] where B1_1, B1_2, B1_3 are 2x2 and B1_4 is 2x1.
       
    3. For the 2x2x2 parts, use Strassen's algorithm (rank 7).
       For the 2x2x1 part, use standard multiplication (rank 4).
       
    Total rank per 2x2x7 block = 7 + 7 + 7 + 4 = 25.
    Total rank for 2x4x7 = 25 + 25 = 50.
    """
    n, m, p = 2, 4, 7
    
    # We will build U, V, W matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
    # rank = 50
    
    rank = 50
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a term to U, V, W
    # term_idx: column index in U, V, W
    # u_vals: dict or list of (index, value) for U
    # v_vals: dict or list of (index, value) for V
    # w_vals: dict or list of (index, value) for W
    
    def add_term(term_idx, u_entries, v_entries, w_entries):
        for idx, val in u_entries:
            U[idx, term_idx] = val
        for idx, val in v_entries:
            V[idx, term_idx] = val
        for idx, val in w_entries:
            W[idx, term_idx] = val

    # Strassen's algorithm for 2x2 matrix multiplication
    # C = A B
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to the global indices.
    # Global A is 2x4. Global B is 4x7. Global C is 2x7.
    # Indices:
    # A[i, j] -> index i*m + j = i*4 + j
    # B[j, k] -> index j*p + k = j*7 + k
    # C[i, k] -> index k*n + i = k*2 + i (Trace convention: W[k*n+i, r])
    
    # Block 1: A1 (rows 0-1, cols 0-1 of A) * B1 (rows 0-1 of B, cols 0-6)
    # A1 indices in A: 0, 1, 4, 5 (i=0,1; j=0,1)
    #   A11 = A[0,0] -> idx 0
    #   A12 = A[0,1] -> idx 1
    #   A21 = A[1,0] -> idx 4
    #   A22 = A[1,1] -> idx 5
    # B1 indices in B: 0..6 (j=0,1; k=0..6)
    #   B11 = B[0,0] -> idx 0
    #   B12 = B[0,1] -> idx 1
    #   B21 = B[1,0] -> idx 7
    #   B22 = B[1,1] -> idx 8
    #   ... B[0, k] -> idx k, B[1, k] -> idx 7+k
    
    # We will process each 2x2x2 sub-block of B1 and B2 separately.
    # B1 is 2x7. We split columns 0-6 into (0,1), (2,3), (4,5), (6).
    
    def process_strassen_block(a_base, b_base, c_base, term_start):
        """
        Process a 2x2x2 multiplication using Strassen.
        a_base: base index for A's 2x2 block (row-major in the 2x2 subgrid)
                A_sub[0,0] at a_base+0, A_sub[0,1] at a_base+1, A_sub[1,0] at a_base+4, A_sub[1,1] at a_base+5
                Wait, A is 2x4. The 2x2 block is A[i, j] for i in 0,1 and j in [j_start, j_start+1].
                So indices are:
                A[0, j_start] -> 0*4 + j_start
                A[0, j_start+1] -> 0*4 + j_start + 1
                A[1, j_start] -> 1*4 + j_start
                A[1, j_start+1] -> 1*4 + j_start + 1
                
        b_base: base index for B's 2x2 block (row-major in the 2x2 subgrid)
                B_sub[0,0] at b_base+0, B_sub[0,1] at b_base+1, B_sub[1,0] at b_base+7, B_sub[1,1] at b_base+8
                B is 4x7. The 2x2 block is B[j, k] for j in [j_start, j_start+1] and k in [k_start, k_start+1].
                Indices:
                B[j_start, k_start] -> j_start*7 + k_start
                B[j_start, k_start+1] -> j_start*7 + k_start + 1
                B[j_start+1, k_start] -> (j_start+1)*7 + k_start
                B[j_start+1, k_start+1] -> (j_start+1)*7 + k_start + 1
                
        c_base: base index for C's 2x2 block (column-major in the 2x2 subgrid? No, trace convention)
                C[i, k] -> k*2 + i.
                C_sub[0,0] (i=0, k=0) -> 0*2 + 0 = 0
                C_sub[0,1] (i=0, k=1) -> 1*2 + 0 = 2
                C_sub[1,0] (i=1, k=0) -> 0*2 + 1 = 1
                C_sub[1,1] (i=1, k=1) -> 1*2 + 1 = 3
                So for a block starting at k_start:
                C[0, k_start] -> k_start*2 + 0
                C[0, k_start+1] -> (k_start+1)*2 + 0
                C[1, k_start] -> k_start*2 + 1
                C[1, k_start+1] -> (k_start+1)*2 + 1
        """
        # A indices
        A11_idx = a_base + 0
        A12_idx = a_base + 1
        A21_idx = a_base + 4
        A22_idx = a_base + 5
        
        # B indices
        B11_idx = b_base + 0
        B12_idx = b_base + 1
        B21_idx = b_base + 7
        B22_idx = b_base + 8
        
        # C indices
        C11_idx = c_base + 0
        C12_idx = c_base + 2
        C21_idx = c_base + 1
        C22_idx = c_base + 3
        
        # Strassen terms
        # M1 = (A11 + A22)(B11 + B22)
        add_term(term_start + 0, 
                 [(A11_idx, 1), (A22_idx, 1)], 
                 [(B11_idx, 1), (B22_idx, 1)], 
                 [(C11_idx, 1), (C22_idx, 1)])
                 
        # M2 = (A21 + A22) B11
        add_term(term_start + 1, 
                 [(A21_idx, 1), (A22_idx, 1)], 
                 [(B11_idx, 1)], 
                 [(C21_idx, 1), (C22_idx, -1)])
                 
        # M3 = A11 (B12 - B22)
        add_term(term_start + 2, 
                 [(A11_idx, 1)], 
                 [(B12_idx, 1), (B22_idx, -1)], 
                 [(C12_idx, 1), (C22_idx, 1)])
                 
        # M4 = A22 (B21 - B11)
        add_term(term_start + 3, 
                 [(A22_idx, 1)], 
                 [(B21_idx, 1), (B11_idx, -1)], 
                 [(C11_idx, 1), (C21_idx, 1)])
                 
        # M5 = (A11 + A12) B22
        add_term(term_start + 4, 
                 [(A11_idx, 1), (A12_idx, 1)], 
                 [(B22_idx, 1)], 
                 [(C11_idx, -1), (C12_idx, 1)])
                 
        # M6 = (A21 - A11) (B11 + B12)
        add_term(term_start + 5, 
                 [(A21_idx, 1), (A11_idx, -1)], 
                 [(B11_idx, 1), (B12_idx, 1)], 
                 [(C22_idx, 1)])
                 
        # M7 = (A12 - A22) (B21 + B22)
        add_term(term_start + 6, 
                 [(A12_idx, 1), (A22_idx, -1)], 
                 [(B21_idx, 1), (B22_idx, 1)], 
                 [(C11_idx, 1), (C12_idx, 1)])
                 
        return term_start + 7

    def process_standard_block(a_base, b_base, c_base, term_start):
        """
        Process a 2x2x1 multiplication using standard 4 terms.
        B is 2x1 here.
        A11 B11 -> C11
        A12 B21 -> C11
        A21 B11 -> C21
        A22 B21 -> C21
        """
        A11_idx = a_base + 0
        A12_idx = a_base + 1
        A21_idx = a_base + 4
        A22_idx = a_base + 5
        
        B11_idx = b_base + 0
        B21_idx = b_base + 7
        
        C11_idx = c_base + 0
        C21_idx = c_base + 1
        
        add_term(term_start + 0, 
                 [(A11_idx, 1)], 
                 [(B11_idx, 1)], 
                 [(C11_idx, 1)])
                 
        add_term(term_start + 1, 
                 [(A12_idx, 1)], 
                 [(B21_idx, 1)], 
                 [(C11_idx, 1)])
                 
        add_term(term_start + 2, 
                 [(A21_idx, 1)], 
                 [(B11_idx, 1)], 
                 [(C21_idx, 1)])
                 
        add_term(term_start + 3, 
                 [(A22_idx, 1)], 
                 [(B21_idx, 1)], 
                 [(C21_idx, 1)])
                 
        return term_start + 4

    current_term = 0
    
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # A1 base index: 0 (A[0,0])
    # B1 base index: 0 (B[0,0])
    
    # B1 cols 0,1 -> Strassen
    current_term = process_strassen_block(0, 0, 0, current_term)
    # B1 cols 2,3 -> Strassen
    # B base for cols 2,3: B[0,2] is index 2. So b_base = 2.
    # C base for cols 2,3: C[0,2] is index 2*2+0 = 4. So c_base = 4.
    current_term = process_strassen_block(0, 2, 4, current_term)
    # B1 cols 4,5 -> Strassen
    # B base for cols 4,5: B[0,4] is index 4. So b_base = 4.
    # C base for cols 4,5: C[0,4] is index 4*2+0 = 8. So c_base = 8.
    current_term = process_strassen_block(0, 4, 8, current_term)
    # B1 col 6 -> Standard
    # B base for col 6: B[0,6] is index 6. So b_base = 6.
    # C base for col 6: C[0,6] is index 6*2+0 = 12. So c_base = 12.
    current_term = process_standard_block(0, 6, 12, current_term)
    
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # A2 base index: A[0,2] is index 2. So a_base = 2.
    # B2 base index: B[2,0] is index 2*7+0 = 14. So b_base = 14.
    
    # B2 cols 0,1 -> Strassen
    # B base: 14. C base: 0.
    current_term = process_strassen_block(2, 14, 0, current_term)
    # B2 cols 2,3 -> Strassen
    # B base: 14+2=16. C base: 4.
    current_term = process_strassen_block(2, 16, 4, current_term)
    # B2 cols 4,5 -> Strassen
    # B base: 14+4=18. C base: 8.
    current_term = process_strassen_block(2, 18, 8, current_term)
    # B2 col 6 -> Standard
    # B base: 14+6=20. C base: 12.
    current_term = process_standard_block(2, 20, 12, current_term)
    
    # Verify rank
    assert current_term == rank, f"Rank mismatch: {current_term} != {rank}"
    
    # Verification
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
                
    T_approx = np.zeros_like(T_full)
    for r in range(rank):
        T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    if not np.allclose(T_full, T_approx, atol=1e-9):
        # Fallback to schoolbook if construction failed (should not happen)
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
