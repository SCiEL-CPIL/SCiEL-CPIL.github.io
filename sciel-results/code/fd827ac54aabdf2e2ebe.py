"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 tensor decomposition for 2x2 by 2x14 matrix multiplication
    using Strassen's algorithm on 7 blocks of 2x2 matrices.
    
    The key insight is that we can decompose the 2x14 matrix B into 7 blocks of 2x2 matrices,
    and apply Strassen's algorithm (rank 7) to each block independently. Since the A matrix
    is shared across all blocks, we can reuse the same U forms for all blocks.
    
    Total rank = 7 blocks * 7 Strassen multiplications = 49
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard: return best result before 150s wall-clock limit
    deadline = start_time + 120  # Safety margin
    
    # Attempt Strassen-based decomposition (rank 49)
    result = _construct_strassen_49(n, m, p)
    
    if result is not None:
        # Verify the decomposition to ensure correctness
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    # Fallback: Schoolbook decomposition (rank 56)
    return _construct_schoolbook(n, m, p)

def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    
    The multiplication C = A*B where A is 2x2 and B is 2x14 can be viewed as:
    - Split B into 7 blocks of 2x2: B = [B_0, B_1, ..., B_6]
    - For each block r, compute C_r = A * B_r using Strassen (7 scalar mults)
    - Total rank = 7 * 7 = 49
    
    Strassen formulas for 2x2 multiplication:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 49)
    V = [[0] * rank for _ in range(m * p)]  # shape (28, 49)
    W = [[0] * rank for _ in range(p * n)]  # shape (28, 49)
    
    # Strassen linear forms for A (2x2)
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U_forms[s][i] gives coefficient of A[i] in M_s
    U_forms = [
        [1, 0, 0, 1],    # M1: A00+A11
        [0, 0, 1, 1],    # M2: A10+A11
        [1, 0, 0, 0],    # M3: A00
        [0, 0, 0, 1],    # M4: A11
        [1, 1, 0, 0],    # M5: A00+A01
        [-1, 0, 1, 0],   # M6: A10-A00
        [0, 1, 0, -1]    # M7: A01-A11
    ]
    
    # V_forms[s][j] gives coefficient of B[j] in M_s
    # B indices within a block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V_forms = [
        [1, 0, 0, 1],    # M1: B00+B11
        [1, 0, 0, 0],    # M2: B00
        [0, 0, 1, -1],   # M3: B10-B11
        [0, 1, 0, -1],   # M4: B01-B00
        [0, 0, 0, 1],    # M5: B11
        [1, 0, 1, 0],    # M6: B00+B10
        [0, 1, 0, 1]     # M7: B01+B11
    ]
    
    # W_forms[s][k] gives coefficient of M_s in C[k]
    # C indices within a block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W_forms = [
        [1, 0, 0, 1],    # M1: C00 + C11
        [0, 0, 1, -1],   # M2: C10 - C11
        [0, 1, 0, 1],    # M3: C01 + C11
        [1, 0, 1, 0],    # M4: C00 + C10
        [-1, 1, 0, 0],   # M5: -C00 + C01
        [0, 0, 0, 1],    # M6: C11
        [1, 0, 0, 0]     # M7: C00
    ]
    
    for r in range(num_blocks):
        # B column indices for block r
        # B is 2x14, so B[j, k] has index j*p + k
        # Block r uses columns 2r and 2r+1
        b_idx = [2*r, 2*r+1, p+2*r, p+2*r+1]  # B(0,2r), B(0,2r+1), B(1,2r), B(1,2r+1)
        
        # W row indices for block r
        # W[k*n+i] where k is output column, i is output row
        # C[i, k] corresponds to W[k*n + i]
        # For block r, output columns are 2r and 2r+1
        # C(0, 2r) = W[2r*2+0] = W[4r]
        # C(1, 2r) = W[2r*2+1] = W[4r+1]
        # C(0, 2r+1) = W[(2r+1)*2+0] = W[4r+2]
        # C(1, 2r+1) = W[(2r+1)*2+1] = W[4r+3]
        w_idx = [4*r, 4*r+1, 4*r+2, 4*r+3]  # C(0,2r), C(1,2r), C(0,2r+1), C(1,2r+1)
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U: A indices 0..3 map to U rows 0..3
            for i in range(4):
                U[i][c] = U_forms[s][i]
                
            # Fill V: B indices for this block
            for j in range(4):
                V[b_idx[j]][c] = V_forms[s][j]
                
            # Fill W: C indices for this block
            for k in range(4):
                W[w_idx[k]][c] = W_forms[s][k]
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    
    The multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    (since C[i,k] = sum_j A[i,j] * B[j,k]).
    
    The decomposition satisfies:
    T[i,j,k] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    
    So we need this sum to equal 1 for all i,j,k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p = 2*2*14 = 56.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
