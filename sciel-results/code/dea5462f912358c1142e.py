"""Matrix multiplication 2x2 by 2x16 with Strassen-based rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# Strassen 2x2 multiplication has rank 7.
# For 2x2 * 2x16, we group the 16 columns into 8 blocks of 2.
# Each block uses a rank-7 Strassen decomposition.
# Total rank = 8 * 7 = 56.

# Strassen rank-7 decomposition for 2x2 * 2x2:
# M1 = (A11 - A21) * (B11 + B21)
# M2 = (A11 + A21) * B21
# M3 = A12 * (B21 - B11)
# M4 = A22 * (B11 - B21)
# M5 = (A11 + A12) * B11
# M6 = (A21 - A22) * (B11 + B22)
# M7 = (A12 + A22) * (B21 - B22)
#
# C11 = M5 + M4 - M2 + M6
# C12 = M1 + M3
# C21 = M2 + M4
# C22 = M5 + M1 - M3 + M7

# For the 2x2 * 2x16 problem:
# A is 2x2, B is 2x16.
# We split B into 8 blocks of 2 columns each: B[:, 2b:2b+2] for b in 0..7.
# For each block, we apply Strassen.

# The tensor decomposition:
# U has shape (n*m, rank) = (4, 56)
# V has shape (m*p, rank) = (32, 56)  
# W has shape (p*n, rank) = (32, 56)

# Indexing:
# A[i,j] -> U row index: i*m + j = i*2 + j
# B[j,k] -> V row index: j*p + k = j*16 + k
# C[k,i] -> W row index: k*n + i = k*2 + i (trace convention W[k*n+i, r])

# Strassen factors for one 2x2 block:
# For block b (columns 2b, 2b+1 of B):
# B11_b = B[0, 2b], B12_b = B[0, 2b+1]
# B21_b = B[1, 2b], B22_b = B[1, 2b+1]

# The 7 Strassen multipliers for block b:
# M1_b = (A11 - A21) * (B11_b + B21_b)
# M2_b = (A11 + A21) * B21_b
# M3_b = A12 * (B21_b - B11_b)
# M4_b = A22 * (B11_b - B21_b)
# M5_b = (A11 + A12) * B11_b
# M6_b = (A21 - A22) * (B11_b + B22_b)
# M7_b = (A12 + A22) * (B21_b - B22_b)

# U factors (4 entries per A element, 7 multipliers per block, 8 blocks):
# For each multiplier t in 0..6 and block b in 0..7:
#   column = b*7 + t
#   U[0, col] = coefficient of A11 in M_t
#   U[1, col] = coefficient of A12 in M_t  (row index 0*2+1=1)
#   U[2, col] = coefficient of A21 in M_t  (row index 1*2+0=2)
#   U[3, col] = coefficient of A22 in M_t  (row index 1*2+1=3)

# V factors (2*16=32 entries per B element):
#   V[row, col] = coefficient of B[j, k] in M_t for block b

# W factors (16*2=32 entries per C element):
#   C[0,0] = M5 + M4 - M2 + M6  -> W[0*2+0, col] contributions
#   C[0,1] = M1 + M3            -> W[1*2+0, col]
#   C[1,0] = M2 + M4            -> W[2*2+0, col]
#   C[1,1] = M5 + M1 - M3 + M7  -> W[3*2+0, col]
# Wait, C is 2x16, so C[row, col] where row in 0..1, col in 0..15.
# W[k*n+i, r] where k is the column of C (0..15), i is the row of C (0..1).
# So W index = k*2 + i.

# Let me redo:
# C has shape 2x16. C[i, k] for i in 0..1, k in 0..15.
# W index = k*n + i = k*2 + i.

# For block b, the output columns are 2b and 2b+1.
# C[0, 2b] = M5_b + M4_b - M2_b + M6_b
# C[0, 2b+1] = M1_b + M3_b
# C[1, 2b] = M2_b + M4_b
# C[1, 2b+1] = M5_b + M1_b - M3_b + M7_b

# W row indices for block b:
# C[0, 2b] -> W[2b*2 + 0] = W[4b]
# C[0, 2b+1] -> W[(2b+1)*2 + 0] = W[4b+2]
# C[1, 2b] -> W[2b*2 + 1] = W[4b+1]
# C[1, 2b+1] -> W[(2b+1)*2 + 1] = W[4b+3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    rank = 56  # 8 blocks * 7 Strassen multipliers
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen coefficients for each multiplier t (0-6):
    # M1: (A11 - A21) * (B11 + B21)
    # M2: (A11 + A21) * B21
    # M3: A12 * (B21 - B11)
    # M4: A22 * (B11 - B21)
    # M5: (A11 + A12) * B11
    # M6: (A21 - A22) * (B11 + B22)
    # M7: (A12 + A22) * (B21 - B22)
    
    # U coefficients: [A11, A12, A21, A22] for each multiplier
    U_coeffs = {
        0: [1, 0, -1, 0],   # M1
        1: [1, 0, 1, 0],    # M2
        2: [0, 1, 0, 0],    # M3
        3: [0, 0, 0, 1],    # M4
        4: [1, 1, 0, 0],    # M5
        5: [0, 0, 1, -1],   # M6
        6: [0, 1, 0, 1],    # M7
    }
    
    # V coefficients: for each multiplier, which B elements are involved
    # For block b, B elements are B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
    # V row index: j*p + k = j*16 + k
    # B[0, 2b] -> row 2b, B[0, 2b+1] -> row 2b+1
    # B[1, 2b] -> row 16+2b, B[1, 2b+1] -> row 16+2b+1
    
    # V coefficients per multiplier: {B_element_index: coeff}
    # B_element_index: 0=B11, 1=B12, 2=B21, 3=B22
    V_coeffs = {
        0: {0: 1, 2: 1},    # M1: B11 + B21
        1: {2: 1},           # M2: B21
        2: {2: 1, 0: -1},   # M3: B21 - B11
        3: {0: 1, 2: -1},   # M4: B11 - B21
        4: {0: 1},           # M5: B11
        5: {0: 1, 3: 1},    # M6: B11 + B22
        6: {2: 1, 3: -1},   # M7: B21 - B22
    }
    
    # W coefficients: for each multiplier, which C elements it contributes to
    # C[0, 2b] gets: M5 + M4 - M2 + M6
    # C[0, 2b+1] gets: M1 + M3
    # C[1, 2b] gets: M2 + M4
    # C[1, 2b+1] gets: M5 + M1 - M3 + M7
    
    # W row indices: 4b, 4b+1, 4b+2, 4b+3 for C[0,2b], C[1,2b], C[0,2b+1], C[1,2b+1]
    # Wait: W[k*n+i] where k is column of C, i is row of C
    # C[0, 2b]: k=2b, i=0 -> W[2b*2+0] = W[4b]
    # C[1, 2b]: k=2b, i=1 -> W[2b*2+1] = W[4b+1]
    # C[0, 2b+1]: k=2b+1, i=0 -> W[(2b+1)*2+0] = W[4b+2]
    # C[1, 2b+1]: k=2b+1, i=1 -> W[(2b+1)*2+1] = W[4b+3]
    
    # W contributions per multiplier: {W_offset: coeff}
    W_coeffs = {
        0: {2: 1, 3: 1},     # M1: C[0,2b+1] + C[1,2b+1]
        1: {0: -1, 1: 1},    # M2: -C[0,2b] + C[1,2b]
        2: {2: 1, 3: -1},    # M3: C[0,2b+1] - C[1,2b+1]
        3: {0: 1, 1: 1},     # M4: C[0,2b] + C[1,2b]
        4: {0: 1, 3: 1},     # M5: C[0,2b] + C[1,2b+1]
        5: {0: 1},            # M6: C[0,2b]
        6: {3: 1},            # M7: C[1,2b+1]
    }
    
    for b in range(8):
        for t in range(7):
            col = b * 7 + t
            
            # Fill U
            for ai, coeff in enumerate(U_coeffs[t]):
                if coeff != 0:
                    U[ai][col] = coeff
            
            # Fill V
            for b_elem, coeff in V_coeffs[t].items():
                if coeff != 0:
                    if b_elem == 0:  # B11 = B[0, 2t]
                        row = 2 * t
                    elif b_elem == 1:  # B12 = B[0, 2t+1]
                        row = 2 * t + 1
                    elif b_elem == 2:  # B21 = B[1, 2t]
                        row = 16 + 2 * t
                    else:  # b_elem == 3: B22 = B[1, 2t+1]
                        row = 16 + 2 * t + 1
                    V[row][col] = coeff
            
            # Fill W
            for w_offset, coeff in W_coeffs[t].items():
                if coeff != 0:
                    row = 4 * b + w_offset
                    W[row][col] = coeff
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for random A, B, verify C = A @ B
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    # C[k, i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually: C[i, k] = sum_{j} A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_r sum_{j} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. The standard tensor decomposition is:
    # C[i,k] = sum_r U[i, r] * V[k, r] * ... no.
    # 
    # For matrix multiplication C = A @ B:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, no. The standard form is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k, r] * B[j,k']) ... 
    # 
    # Actually the standard decomposition is:
    # C[i,k] = sum_r U[i, r] * V[k, r] where U and V depend on A and B.
    # But in the matrix multiplication tensor:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    # where U, V, W are the factor matrices.
    # 
    # Wait, I need to be more careful. The task says:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # 
    # The decomposition should satisfy:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k} contribution
    # 
    # Actually, the matrix multiplication tensor T is defined by:
    # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # The tensor T has T[i,j,k] = 1 for the "trivial" decomposition.
    # 
    # A rank-r decomposition means: T[i,j,k] = sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # So let's verify: for each (i,j,k), the sum over r should equal 1 if this is a valid term, else 0.
    # But actually T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
    # 
    # Wait no. The multiplication tensor T_{i,j,k} is defined such that:
    # C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{j} T_{i,j,k} A[i,j] B[j,k]
    # So T_{i,j,k} = 1 for all valid i,j,k.
    # 
    # The rank-r decomposition: T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # 
    # Let's verify this.
    
    # Build the tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    
    # Compute the reconstructed tensor from the decomposition
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U_np[:, r]  # shape (4,)
        v_col = V_np[:, r]  # shape (32,)
        w_col = W_np[:, r]  # shape (32,)
        
        # u_col[i*m+j] for i in 0..1, j in 0..1
        # v_col[j*p+k] for j in 0..1, k in 0..15
        # w_col[k*n+i] for k in 0..15, i in 0..1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    
    # Check if T_recon == T
    max_err = np.max(np.abs(T_recon - T))
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
