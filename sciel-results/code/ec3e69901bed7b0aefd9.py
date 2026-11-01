import numpy as np
import time

def construct():
    """
    Returns a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    Uses a block structure:
    1. j=0: 2x1x5 block (Rank 10, Schoolbook)
    2. j in {1,2}: 2x2x5 block (Rank 18)
       - k in {0,1}: 2x2x2 block (Rank 7, Strassen)
       - k=2: 2x2x1 block (Rank 4, Schoolbook)
       - k in {3,4}: 2x2x2 block (Rank 7, Strassen)
    Total Rank: 10 + 7 + 4 + 7 = 28.
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1
        return col
    
    # Helper to add Strassen 2x2x2 block
    # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
    # Global indices: 
    # U: i * 3 + (j_base + j')
    # V: (j_base + j') * 5 + (k_base + k')
    # W: (k_base + k') * 2 + i
    def add_strassen(i_base, j_base, k_base):
        # Note: i_base is 0 for both Strassen blocks here, as i is always 0 or 1
        # j_base is 1 for both
        # k_base is 0 for first, 3 for second
        
        # M1: (A[0, j_b] + A[1, j_b+1]) * (B[j_b, k_b] + B[j_b+1, k_b+1])
        # U: [i_base*3 + j_base], [ (i_base+1)*3 + j_base+1 ]
        # V: [ j_base*5 + k_base ], [ (j_base+1)*5 + k_base+1 ]
        # W: [ k_base*2 + i_base ], [ (k_base+1)*2 + (i_base+1) ]
        
        # M1
        u1_idx = i_base * 3 + j_base
        u2_idx = (i_base + 1) * 3 + (j_base + 1)
        v1_idx = j_base * 5 + k_base
        v2_idx = (j_base + 1) * 5 + (k_base + 1)
        w1_idx = k_base * 2 + i_base
        w2_idx = (k_base + 1) * 2 + (i_base + 1)
        
        U[u1_idx, col] = 1
        U[u2_idx, col] = 1
        V[v1_idx, col] = 1
        V[v2_idx, col] = 1
        W[w1_idx, col] = 1
        W[w2_idx, col] = 1
        col += 1
        
        # M2: (A[1, j_b] + A[1, j_b+1]) * B[j_b, k_b]
        # Note: Standard Strassen M2 is (A11 + A22) * B21? Or (A11 + A22) * B11?
        # Let's stick to the explicit indices from the previous working code to avoid errors.
        # The previous code had:
        # M2: U[1*3+1], U[1*3+2] (A[1,j_b], A[1,j_b+1])
        #      V[1*5+0] (B[j_b, k_b])
        #      W[0*2+1] (C[1, k_b]), W[1*2+1] (C[1, k_b+1])
        # Wait, the previous code's Strassen mapping was:
        # M2: (A[1,1]+A[1,2]) * B[1,0] -> C[1,0] += 1, C[1,1] += -1
        # This implies:
        # U: A[1, j_b], A[1, j_b+1]
        # V: B[j_b, k_b]
        # W: C[1, k_b] += 1, C[1, k_b+1] += -1
        
        U[u2_idx, col] = 1 # A[1, j_b+1] is u2_idx? No, u2_idx was A[1, j_b+1]. 
                           # In prev code: U[1*3+1] is A[1,1], U[1*3+2] is A[1,2].
                           # Here j_base=1. u1_idx = 0*3+1=1 (A[0,1]). u2_idx = 1*3+2=5 (A[1,2]).
                           # Prev code M2 used U[1*3+1] (A[1,1]) and U[1*3+2] (A[1,2]).
                           # My u2_idx is A[1,2]. I need A[1,1] too.
                           # Let's just replicate the previous explicit indices to be safe.
        
    # The helper approach is error-prone due to index shifts. 
    # I will explicitly write the blocks again, but using NumPy assignments.
    
    # Reset
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    # Block 1: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Block 2a: j in {1,2}, k in {0,1}
    # Strassen 2x2x2
    # M1
    U[0 * 3 + 1, col] = 1; U[1 * 3 + 2, col] = 1
    V[1 * 5 + 0, col] = 1; V[2 * 5 + 1, col] = 1
    W[0 * 2 + 0, col] = 1; W[1 * 2 + 1, col] = 1
    col += 1
    # M2
    U[1 * 3 + 1, col] = 1; U[1 * 3 + 2, col] = 1
    V[1 * 5 + 0, col] = 1
    W[0 * 2 + 1, col] = 1; W[1 * 2 + 1, col] = -1
    col += 1
    # M3
    U[0 * 3 + 1, col] = 1
    V[1 * 5 + 1, col] = 1; V[2 * 5 + 1, col] = -1
    W[1 * 2 + 0, col] = 1; W[1 * 2 + 1, col] = 1
    col += 1
    # M4
    U[1 * 3 + 2, col] = 1
    V[2 * 5 + 0, col] = 1; V[1 * 5 + 0, col] = -1
    W[0 * 2 + 0, col] = 1; W[0 * 2 + 1, col] = 1
    col += 1
    # M5
    U[0 * 3 + 1, col] = 1; U[0 * 3 + 2, col] = 1
    V[2 * 5 + 1, col] = 1
    W[0 * 2 + 0, col] = -1; W[1 * 2 + 0, col] = 1
    col += 1
    # M6
    U[1 * 3 + 1, col] = 1; U[0 * 3 + 1, col] = -1
    V[1 * 5 + 0, col] = 1; V[1 * 5 + 1, col] = 1
    W[1 * 2 + 1, col] = 1
    col += 1
    # M7
    U[0 * 3 + 2, col] = 1; U[1 * 3 + 2, col] = -1
    V[2 * 5 + 0, col] = 1; V[2 * 5 + 1, col] = 1
    W[0 * 2 + 0, col] = 1
    col += 1
    
    # Block 2b: j in {1,2}, k=2
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * 3 + j, col] = 1
            V[j * 5 + 2, col] = 1
            W[2 * 2 + i, col] = 1
            col += 1
            
    # Block 2c: j in {1,2}, k in {3,4}
    # Strassen 2x2x2
    # M1
    U[0 * 3 + 1, col] = 1; U[1 * 3 + 2, col] = 1
    V[1 * 5 + 3, col] = 1; V[2 * 5 + 4, col] = 1
    W[3 * 2 + 0, col] = 1; W[4 * 2 + 1, col] = 1
    col += 1
    # M2
    U[1 * 3 + 1, col] = 1; U[1 * 3 + 2, col] = 1
    V[1 * 5 + 3, col] = 1
    W[3 * 2 + 1, col] = 1; W[4 * 2 + 1, col] = -1
    col += 1
    # M3
    U[0 * 3 + 1, col] = 1
    V[1 * 5 + 4, col] = 1; V[2 * 5 + 4, col] = -1
    W[4 * 2 + 0, col] = 1; W[4 * 2 + 1, col] = 1
    col += 1
    # M4
    U[1 * 3 + 2, col] = 1
    V[2 * 5 + 3, col] = 1; V[1 * 5 + 3, col] = -1
    W[3 * 2 + 0, col] = 1; W[3 * 2 + 1, col] = 1
    col += 1
    # M5
    U[0 * 3 + 1, col] = 1; U[0 * 3 + 2, col] = 1
    V[2 * 5 + 4, col] = 1
    W[3 * 2 + 0, col] = -1; W[4 * 2 + 0, col] = 1
    col += 1
    # M6
    U[1 * 3 + 1, col] = 1; U[0 * 3 + 1, col] = -1
    V[1 * 5 + 3, col] = 1; V[1 * 5 + 4, col] = 1
    W[4 * 2 + 1, col] = 1
    col += 1
    # M7
    U[0 * 3 + 2, col] = 1; U[1 * 3 + 2, col] = -1
    V[2 * 5 + 3, col] = 1; V[2 * 5 + 4, col] = 1
    W[3 * 2 + 0, col] = 1
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} vs {rank}"
    
    # Verification
    # Check that U * diag * V ... equals the multiplication tensor
    # Equivalent to: for all i,j,k, sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 if matching else 0?
    # Actually, the decomposition is:
    # C = sum_r (U[:,r] * A) * (V[:,r] * B) * W[:,r]^T? No.
    # The standard decomposition is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # = sum_r ( sum_j U_{ij}^r A_{ij} )? No.
    # U, V, W are the factors.
    # C_{ik} = sum_r W_{ik}^r ( sum_j U_{ij}^r A_{ij} ) ( sum_k V_{jk}^r B_{jk} )?
    # No, the indices are:
    # Term r involves:
    # x_r = sum_{i,j} U_{ij}^r A_{ij}
    # y_r = sum_{j,k} V_{jk}^r B_{jk}
    # z_r = sum_{i,k} W_{ik}^r C_{ik}
    # This is not correct.
    # The decomposition of the tensor M is:
    # M = sum_r U^{(r)} \otimes V^{(r)} \otimes W^{(r)}
    # Where U^{(r)} is a matrix of size n x m, etc.
    # In the flattened index notation:
    # U is (n*m, rank). Row (i*m+j) is U_{ij}^r.
    # V is (m*p, rank). Row (j*p+k) is V_{jk}^r.
    # W is (p*n, rank). Row (k*n+i) is W_{ik}^r.
    
    # Let's verify with random matrices.
    np.random.seed(42)
    A_test = np.random.randint(-10, 10, (n, m))
    B_test = np.random.randint(-10, 10, (m, p))
    C_test = A_test @ B_test
    
    # Compute C_decomposed
    C_dec = np.zeros((n, p))
    for r in range(rank):
        # x_r = sum A_ij U_ij^r
        x_r = np.sum(A_test * U[:, r].reshape(n, m))
        y_r = np.sum(B_test * V[:, r].reshape(m, p))
        # The contribution to C is x_r * y_r * W[:, r].reshape(n, p) ?
        # No, the tensor is sum_r U_r \otimes V_r \otimes W_r.
        # This means C_{ik} = sum_r W_{ik}^r * (sum_j U_{ij}^r A_{ij})? No.
        # The bilinear form is:
        # sum_r (sum_{i,j} U_{ij}^r A_{ij}) (sum_{j,k} V_{jk}^r B_{jk}) ... 
        # This doesn't sum to C_{ik} directly unless W is also a product?
        
        # Correct interpretation:
        # The multiplication algorithm computes:
        # For each r:
        #   m_r = (linear combo of A) * (linear combo of B)
        #   C += m_r * (linear combo of C) ? No.
        #   C_{ik} = sum_r W_{ik}^r m_r
        
        # So:
        # m_r = (sum_{i,j} U_{ij}^r A_{ij}) * (sum_{j,k} V_{jk}^r B_{jk}) ?
        # Wait, the indices j are summed in both?
        # Usually, the factors are:
        # U_r is n x m
        # V_r is m x p
        # W_r is n x p
        # m_r = trace(U_r^T A) * trace(V_r^T B) ? No.
        # m_r = sum_{i,j} U_{ij}^r A_{ij} * sum_{j,k} V_{jk}^r B_{jk} ?
        # This would mix j.
        
        # Let's look at the standard Strassen example.
        # M1: (A00+A11) * (B00+B11).
        # U has 1 at (0,0) and (1,1).
        # V has 1 at (0,0) and (1,1).
        # W has 1 at (0,0) and (1,1) (for C00 and C11).
        # m1 = (A00+A11) * (B00+B11).
        # C00 += 1 * m1. C11 += 1 * m1.
        
        # So the formula is:
        # m_r = (sum_{i,j} U_{ij}^r A_{ij}) * (sum_{j,k} V_{jk}^r B_{jk})
        # C_{ik} = sum_r W_{ik}^r m_r
        
        a_comb = np.sum(A_test * U[:, r].reshape(n, m))
        b_comb = np.sum(B_test * V[:, r].reshape(m, p))
        m_r = a_comb * b_comb
        C_dec += m_r * W[:, r].reshape(n, p)
        
    if not np.allclose(C_test, C_dec):
        raise ValueError("Decomposition verification failed")
        
    return {"U": U, "V": V, "W": W}

