"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)
        T_full = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T_full, 1.0, atol=1e-8)
    
    # Try to find lower-rank decompositions using block-based Strassen construction
    # Split 5 = 2 + 2 + 1 and use Strassen for 2x2x2 sub-problems
    
    # Strassen's 7-multiplication algorithm for 2x2 matrices:
    # m1 = (A11+A22)*(B11+B22)
    # m2 = (A21+A22)*B11
    # m3 = A11*(B12-B22)
    # m4 = A22*(B21-B11)
    # m5 = (A11+A12)*B22
    # m6 = (A21-A11)*(B11+B12)
    # m7 = (A12-A22)*(B21+B22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For the tensor decomposition, each multiplication m_r corresponds to a rank-1 term
    # U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for the relevant (i,j,k)
    
    # Block decomposition: rows 0-1, 2-3, 4 of A; cols 0-1, 2-3, 4 of B
    # For each pair of row-block and col-block, we compute C[row_block, col_block]
    
    # Let's build the decomposition manually for the block structure
    # Row blocks: R0={0,1}, R1={2,3}, R2={4}
    # Col blocks: C0={0,1}, C1={2,3}, C2={4}
    # Middle blocks: M0={0,1}, M1={2,3}, M2={4}
    
    # For C[R0, C0] = A[R0,M0]*B[M0,C0] + A[R0,M1]*B[M1,C0] + A[R0,M2]*B[M2,C0]
    # The first term is 2x2*2x2 -> use Strassen (7 terms)
    # The second term is 2x2*2x2 -> use Strassen (7 terms)
    # The third term is 2x1*1x2 -> schoolbook (4 terms)
    # Total for C[R0,C0]: 7+7+4 = 18
    
    # Similarly for other blocks. Let's compute total rank:
    # C[R0,C0]: 7+7+4=18
    # C[R0,C1]: 7+7+4=18
    # C[R0,C2]: 2*1*2=4 (2x1 * 1x2, but wait: A[R0,M0]*B[M0,C2] is 2x2*2x1=4 terms)
    #   Actually A[R0,M0] is 2x2, B[M0,C2] is 2x1, product is 2x1, needs 4 mults
    #   A[R0,M1] is 2x2, B[M1,C2] is 2x1, product is 2x1, needs 4 mults
    #   A[R0,M2] is 2x1, B[M2,C2] is 1x1, product is 2x1, needs 2 mults
    #   Total: 4+4+2=10
    # C[R1,C0]: same as C[R0,C0] = 18
    # C[R1,C1]: same as C[R0,C0] = 18
    # C[R1,C2]: same as C[R0,C2] = 10
    # C[R2,C0]: A[R2,M0] is 1x2, B[M0,C0] is 2x2 -> 4 mults
    #            A[R2,M1] is 1x2, B[M1,C0] is 2x2 -> 4 mults
    #            A[R2,M2] is 1x1, B[M2,C0] is 1x2 -> 2 mults
    #            Total: 4+4+2=10
    # C[R2,C1]: same as C[R2,C0] = 10
    # C[R2,C2]: A[R2,M0] is 1x2, B[M0,C2] is 2x1 -> 2 mults
    #            A[R2,M1] is 1x2, B[M1,C2] is 2x1 -> 2 mults
    #            A[R2,M2] is 1x1, B[M2,C2] is 1x1 -> 1 mult
    #            Total: 2+2+1=5
    
    # Grand total: 18+18+10+18+18+10+10+10+5 = 117
    
    # That's better than 125! Let's implement this.
    
    # Define block indices
    R_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end) for row blocks
    C_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end) for col blocks
    M_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end) for middle blocks
    
    # Build the decomposition
    new_U = []
    new_V = []
    new_W = []
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a rank-1 term. u_vals, v_vals, w_vals are dicts mapping (i,j), (j,k), (k,i) to value"""
        # We need to store full vectors
        U_vec = np.zeros(n * m)
        V_vec = np.zeros(m * p)
        W_vec = np.zeros(p * n)
        
        for (i, j), val in u_vals.items():
            if val != 0:
                U_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            if val != 0:
                V_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            if val != 0:
                W_vec[k * n + i] = val
        
        new_U.append(U_vec)
        new_V.append(V_vec)
        new_W.append(W_vec)
    
    def strassen_2x2(row_start, col_start, mid_start, U_list, V_list, W_list):
        """Add Strassen terms for 2x2*2x2 block"""
        r0, r1 = row_start, row_start + 1
        c0, c1 = col_start, col_start + 1
        m0, m1 = mid_start, mid_start + 1
        
        # m1 = (A11+A22)*(B11+B22)
        add_term(
            {(r0,m0): 1, (r1,m1): 1},
            {(m0,c0): 1, (m1,c1): 1},
            {(c0,r0): 1, (c1,r0): 1, (c0,r1): 1, (c1,r1): 1}
        )
        # m2 = (A21+A22)*B11
        add_term(
            {(r1,m0): 1, (r1,m1): 1},
            {(m0,c0): 1},
            {(c0,r1): 1}
        )
        # m3 = A11*(B12-B22)
        add_term(
            {(r0,m0): 1},
            {(m0,c1): 1, (m1,c1): -1},
            {(c1,r0): 1}
        )
        # m4 = A22*(B21-B11)
        add_term(
            {(r1,m1): 1},
            {(m1,c0): 1, (m0,c0): -1},
            {(c0,r1): 1}
        )
        # m5 = (A11+A12)*B22
        add_term(
            {(r0,m0): 1, (r0,m1): 1},
            {(m1,c1): 1},
            {(c1,r0): 1}
        )
        # m6 = (A21-A11)*(B11+B12)
        add_term(
            {(r1,m0): 1, (r0,m0): -1},
            {(m0,c0): 1, (m0,c1): 1},
            {(c0,r1): 1, (c1,r1): 1}
        )
        # m7 = (A12-A22)*(B21+B22)
        add_term(
            {(r0,m1): 1, (r1,m1): -1},
            {(m1,c0): 1, (m1,c1): 1},
            {(c0,r0): 1, (c1,r0): 1}
        )
    
    def schoolbook_block(row_start, row_end, mid_start, mid_end, col_start, col_end, U_list, V_list, W_list):
        """Add schoolbook terms for a block multiplication"""
        for i in range(row_start, row_end):
            for j in range(mid_start, mid_end):
                for k in range(col_start, col_end):
                    add_term(
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    )
    
    # Iterate over all block combinations
    for rb_idx, (r_start, r_end) in enumerate(R_blocks):
        for cb_idx, (c_start, c_end) in enumerate(C_blocks):
            for mb_idx, (m_start, m_end) in enumerate(M_blocks):
                r_size = r_end - r_start
                m_size = m_end - m_start
                c_size = c_end - c_start
                
                if r_size == 2 and m_size == 2 and c_size == 2:
                    # Use Strassen
                    strassen_2x2(r_start, c_start, m_start, new_U, new_V, new_W)
                else:
                    # Use schoolbook
                    schoolbook_block(r_start, r_end, m_start, m_end, c_start, c_end, new_U, new_V, new_W)
    
    # Convert to arrays
    if new_U:
        U_new = np.array(new_U).T  # Shape (n*m, rank)
        V_new = np.array(new_V).T  # Shape (m*p, rank)
        W_new = np.array(new_W).T  # Shape (p*n, rank)
        
        new_rank = U_new.shape[1]
        if verify(U_new, V_new, W_new, new_rank):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = new_rank
    
    # If the block construction didn't work or wasn't better, try random search
    if best_rank >= rank_schoolbook:
        np.random.seed(42)
        candidate_vals = [0.0, 0.5, -0.5, 1.0, -1.0]
        
        for target_rank in [110, 100, 90, 80, 70]:
            if time.time() > deadline - 20:
                break
            
            for trial in range(100):
                if time.time() > deadline - 15:
                    break
                
                U_try = np.array([[np.random.choice(candidate_vals) for _ in range(target_rank)] for _ in range(n * m)], dtype=np.float64)
                V_try = np.array([[np.random.choice(candidate_vals) for _ in range(target_rank)] for _ in range(m * p)], dtype=np.float64)
                W_try = np.array([[np.random.choice(candidate_vals) for _ in range(target_rank)] for _ in range(p * n)], dtype=np.float64)
                
                if verify(U_try, V_try, W_try, target_rank):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = target_rank
                    break
            if best_rank < rank_schoolbook:
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
