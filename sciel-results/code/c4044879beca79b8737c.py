"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7
    
    # Schoolbook as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if s != 1:
                        return False
        return True
    
    # Strategy: Recursive block-partitioning with Strassen on 2x2x2 blocks
    # Split n=3 into 2+1, m=5 into 2+2+1, p=7 into 2+2+2+1
    # This gives sub-blocks of various sizes. The 2x2x2 blocks can use Strassen (rank 7 instead of 8).
    # All other blocks use schoolbook.
    
    # Strassen decomposition for 2x2x2
    # C = A*B where A, B are 2x2
    # C[0,0] = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) - (A[0,1]+A[1,1])*(B[0,1]+B[1,1]) ... 
    # Actually, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # In tensor form, for each of the 7 Strassen terms, we have:
    # U factors (2x2 -> 4 entries), V factors (2x2 -> 4 entries), W factors (2x2 -> 4 entries)
    # Each term contributes to multiple C entries with coefficients.
    
    # Let's build the decomposition by partitioning.
    # Partition indices:
    # n: [0,1] (size 2), [2] (size 1)
    # m: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # p: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2), [6] (size 1)
    
    # Sub-blocks: for each (n_part, m_part, p_part)
    # n_parts: [(0,1), (2,)]
    # m_parts: [(0,1), (2,3), (4,)]
    # p_parts: [(0,1), (2,3), (4,5), (6,)]
    
    # For each combination, determine the size and use Strassen if 2x2x2, else schoolbook.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen(n_start, m_start, p_start):
        """Add Strassen decomposition for a 2x2x2 block starting at (n_start, m_start, p_start)."""
        # The 2x2x2 block involves:
        # n indices: n_start, n_start+1
        # m indices: m_start, m_start+1
        # p indices: p_start, p_start+1
        #
        # Strassen terms:
        # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
        # M2 = (A21+A22)B11 -> contributes to C21, C22
        # M3 = A11(B21-B22) -> contributes to C12, C21
        # M4 = A22(B11-B12) -> contributes to C11, C12
        # M5 = (A11+A12)B22 -> contributes to C11, C12
        # M6 = (A21-A11)(B11+B12) -> contributes to C22
        # M7 = (A12-A22)(B21+B22) -> contributes to C11
        
        # Let's define the 7 terms with their U, V, W coefficients
        # U coefficients for A entries (2x2):
        # A11 -> U[n_start, m_start]
        # A12 -> U[n_start, m_start+1]
        # A21 -> U[n_start+1, m_start]
        # A22 -> U[n_start+1, m_start+1]
        
        # V coefficients for B entries (2x2):
        # B11 -> V[m_start, p_start]
        # B12 -> V[m_start, p_start+1]
        # B21 -> V[m_start+1, p_start]
        # B22 -> V[m_start+1, p_start+1]
        
        # W coefficients for C entries (2x2):
        # C11 -> W[p_start, n_start]
        # C12 -> W[p_start+1, n_start]
        # C21 -> W[p_start, n_start+1]
        # C22 -> W[p_start+1, n_start+1]
        
        # Strassen terms (each is a triple of linear forms):
        # Term 1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1) -> C11 += 1, C22 += 1
        # Term 2: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,1) -> C21 += 1, C22 += 1
        # Term 3: U=(1,0,0,0), V=(0,-1,1,0), W=(0,1,1,0) -> C12 += 1, C21 += 1
        # Term 4: U=(0,0,0,1), V=(1,-1,0,0), W=(1,1,0,0) -> C11 += 1, C12 += 1
        # Term 5: U=(1,1,0,0), V=(0,0,0,1), W=(1,1,0,0) -> C11 += 1, C12 += 1
        # Term 6: U=(-1,0,1,0), V=(1,1,0,0), W=(0,0,0,1) -> C22 += 1
        # Term 7: U=(0,1,0,-1), V=(0,0,1,1), W=(1,0,0,0) -> C11 += 1
        
        # Wait, I need to be more careful. Let me re-derive.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B12)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for each term, we need to specify:
        # - The linear form in A (U coefficients)
        # - The linear form in B (V coefficients)
        # - The contribution to each C entry (W coefficients)
        
        # Term 1: A form = A11+A22, B form = B11+B22
        #   Contributes: +1 to C11, +1 to C22
        # Term 2: A form = A21+A22, B form = B11
        #   Contributes: +1 to C21, -1 to C22
        # Term 3: A form = A11, B form = B21-B22
        #   Contributes: +1 to C12, +1 to C22
        # Term 4: A form = A22, B form = B11-B12
        #   Contributes: +1 to C11, +1 to C12
        # Term 5: A form = A11+A12, B form = B22
        #   Contributes: -1 to C11, +1 to C12
        # Term 6: A form = A21-A11, B form = B11+B12
        #   Contributes: +1 to C22
        # Term 7: A form = A12-A22, B form = B21+B22
        #   Contributes: +1 to C11
        
        # Now, each term is a rank-1 tensor in the decomposition.
        # U coefficients (for A entries [A11, A12, A21, A22]):
        # V coefficients (for B entries [B11, B12, B21, B22]):
        # W coefficients (for C entries [C11, C12, C21, C22]):
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # Term 1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # Term 2
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),   # Term 3
            ([0, 0, 0, 1], [1, -1, 0, 0], [1, 1, 0, 0]),   # Term 4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # Term 5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # Term 6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # Term 7
        ]
        
        for u_coeff, v_coeff, w_coeff in terms:
            # Create U, V, W vectors for this term
            u_vec = np.zeros(n * m, dtype=np.int64)
            v_vec = np.zeros(m * p, dtype=np.int64)
            w_vec = np.zeros(p * n, dtype=np.int64)
            
            # Set U coefficients
            u_vec[n_start * m + m_start] = u_coeff[0]     # A11
            u_vec[n_start * m + m_start + 1] = u_coeff[1] # A12
            u_vec[(n_start + 1) * m + m_start] = u_coeff[2]   # A21
            u_vec[(n_start + 1) * m + m_start + 1] = u_coeff[3] # A22
            
            # Set V coefficients
            v_vec[m_start * p + p_start] = v_coeff[0]     # B11
            v_vec[m_start * p + p_start + 1] = v_coeff[1] # B12
            v_vec[(m_start + 1) * p + p_start] = v_coeff[2]   # B21
            v_vec[(m_start + 1) * p + p_start + 1] = v_coeff[3] # B22
            
            # Set W coefficients
            w_vec[p_start * n + n_start] = w_coeff[0]     # C11
            w_vec[p_start * n + n_start + 1] = w_coeff[1] # C12
            w_vec[(p_start + 1) * n + n_start] = w_coeff[2]   # C21
            w_vec[(p_start + 1) * n + n_start + 1] = w_coeff[3] # C22
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
    
    def add_schoolbook_block(n_start, n_end, m_start, m_end, p_start, p_end):
        """Add schoolbook decomposition for a block from n_start to n_end-1, m_start to m_end-1, p_start to p_end-1."""
        for i in range(n_start, n_end):
            for j in range(m_start, m_end):
                for k in range(p_start, p_end):
                    u_vec = np.zeros(n * m, dtype=np.int64)
                    v_vec = np.zeros(m * p, dtype=np.int64)
                    w_vec = np.zeros(p * n, dtype=np.int64)
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k] = 1
                    w_vec[k * n + i] = 1
                    U_list.append(u_vec)
                    V_list.append(v_vec)
                    W_list.append(w_vec)
    
    # Define partitions
    n_parts = [(0, 2), (2, 3)]  # (start, end) exclusive
    m_parts = [(0, 2), (2, 4), (4, 5)]
    p_parts = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    for n_start, n_end in n_parts:
        for m_start, m_end in m_parts:
            for p_start, p_end in p_parts:
                n_size = n_end - n_start
                m_size = m_end - m_start
                p_size = p_end - p_start
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(n_start, m_start, p_start)
                else:
                    add_schoolbook_block(n_start, n_end, m_start, m_end, p_start, p_end)
    
    # Assemble the full matrices
    rank = len(U_list)
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Verify
    if verify(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    else:
        # Fallback to schoolbook
        return schoolbook()
# EVOLVE-BLOCK-END
