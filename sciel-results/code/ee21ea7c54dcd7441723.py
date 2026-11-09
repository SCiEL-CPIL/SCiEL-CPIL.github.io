import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    # Rank 50
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a scalar to U[i][col], V[j][col], W[k][col]
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        # Note: Strassen terms are products of sums. 
        # The decomposition is T = sum u_l v_l w_l.
        # Each term is a rank-1 tensor.
        # So for each multiplier M, we have a column in U, V, W.
        # U column contains coefficients for A.
        # V column contains coefficients for B.
        # W column contains coefficients for C.
        
    # This helper is wrong for Strassen because M = (sum A)(sum B).
    # It is not a single term. It is one column in U, one in V, one in W.
    # The column in U has the A coefficients.
    # The column in V has the B coefficients.
    # The column in W has the C coefficients.
    
    # So I need a helper that sets a whole column.
    
    def set_column(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
        
    # Indices for A (2x3)
    # A[i, j] -> U row i*3 + j
    # A1 is A[:, 0:2]
    # A2 is A[:, 2]
    
    # Indices for B (3x9)
    # B[j, k] -> V row j*9 + k
    
    # Indices for C (2x9)
    # C[i, k] -> W row k*2 + i (Task: W[k*n+i, r])
    
    # Part 1: A1 B1
    # A1 is A[0:2, 0:2]
    # B1 is B[0:2, 0:9]
    
    # B1 blocks
    # 4 blocks of 2x2
    # 1 block of 2x1
    
    for b in range(4):
        # B block cols: b*2, b*2+1
        c1 = b * 2
        c2 = b * 2 + 1
        
        # Strassen M1: (A00+A11)(B0c1+B1c2)
        u = {0*3+0: 1, 1*3+1: 1} # A00, A11
        v = {0*9+c1: 1, 1*9+c2: 1} # B0c1, B1c2
        w = {c1*2+0: 1, c2*2+1: 1} # C0c1, C1c2
        set_column(u, v, w)
        
        # Strassen M2: (A10+A11)B0c1
        u = {1*3+0: 1, 1*3+1: 1} # A10, A11
        v = {0*9+c1: 1} # B0c1
        w = {c1*2+1: 1, c2*2+1: -1} # C1c1, C1c2
        set_column(u, v, w)
        
        # Strassen M3: A00(B1c2-B0c2)
        u = {0*3+0: 1} # A00
        v = {1*9+c2: 1, 0*9+c2: -1} # B1c2, B0c2
        w = {c2*2+0: 1, c2*2+1: 1} # C0c2, C1c2
        set_column(u, v, w)
        
        # Strassen M4: A11(B1c1-B0c1)
        u = {1*3+1: 1} # A11
        v = {1*9+c1: 1, 0*9+c1: -1} # B1c1, B0c1
        w = {c1*2+0: 1, c1*2+1: 1} # C0c1, C1c1
        set_column(u, v, w)
        
        # Strassen M5: (A00+A01)B1c2
        u = {0*3+0: 1, 0*3+1: 1} # A00, A01
        v = {1*9+c2: 1} # B1c2
        w = {c1*2+0: -1, c2*2+0: 1} # C0c1, C0c2
        set_column(u, v, w)
        
        # Strassen M6: (A10-A00)(B0c1+B1c1)
        u = {1*3+0: 1, 0*3+0: -1} # A10, A00
        v = {0*9+c1: 1, 1*9+c1: 1} # B0c1, B1c1
        w = {c2*2+1: 1} # C1c2
        set_column(u, v, w)
        
        # Strassen M7: (A01-A11)(B1c1+B0c2)
        u = {0*3+1: 1, 1*3+1: -1} # A01, A11
        v = {1*9+c1: 1, 0*9+c2: 1} # B1c1, B0c2
        w = {c1*2+0: 1} # C0c1
        set_column(u, v, w)
        
    # B1 block 5: 2x1 at col 8
    c1 = 8
    
    # Schoolbook 2x2 x 2x1
    # C0c1 = A00*B0c1 + A01*B1c1
    # C1c1 = A10*B0c1 + A11*B1c1
    
    u = {0*3+0: 1} # A00
    v = {0*9+c1: 1} # B0c1
    w = {c1*2+0: 1} # C0c1
    set_column(u, v, w)
    
    u = {0*3+1: 1} # A01
    v = {1*9+c1: 1} # B1c1
    w = {c1*2+0: 1} # C0c1
    set_column(u, v, w)
    
    u = {1*3+0: 1} # A10
    v = {0*9+c1: 1} # B0c1
    w = {c1*2+1: 1} # C1c1
    set_column(u, v, w)
    
    u = {1*3+1: 1} # A11
    v = {1*9+c1: 1} # B1c1
    w = {c1*2+1: 1} # C1c1
    set_column(u, v, w)
    
    # Part 2: A2 B2
    # A2 is A[:, 2] (2x1)
    # B2 is B[2, :] (1x9)
    
    # C0k = A02 * B2k
    # C1k = A12 * B2k
    
    for k in range(9):
        u = {0*3+2: 1} # A02
        v = {2*9+k: 1} # B2k
        w = {k*2+0: 1} # C0k
        set_column(u, v, w)
        
        u = {1*3+2: 1} # A12
        v = {2*9+k: 1} # B2k
        w = {k*2+1: 1} # C1k
        set_column(u, v, w)
        
    return {"U": U, "V": V, "W": W}
