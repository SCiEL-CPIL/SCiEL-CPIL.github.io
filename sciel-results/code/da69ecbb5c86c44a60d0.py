import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    Returns: dict with 'U', 'V', 'W' factor matrices
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Strategy: Split the 3 middle dimension into 2+1
    # Part 1: 2x2x9 (using A[0:2,0:2] and B[0:2,:])
    # Part 2: 2x1x9 (using A[0:2,2] and B[2,:])
    
    # For Part 1 (2x2x9):
    # Split the 9 columns into 4 blocks of 2 and 1 block of 1
    # Each 2x2x2 block uses Strassen's algorithm (rank 7)
    # Each 2x2x1 block uses direct multiplication (rank 4)
    # Total for Part 1: 4*7 + 4 = 32
    
    # For Part 2 (2x1x9):
    # This is just 18 scalar multiplications (2x9 = 18)
    # Total for Part 2: 18
    
    # Grand total: 32 + 18 = 50
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    col = 0
    
    # === Part 1: 2x2x9 using Strassen ===
    # Process 4 blocks of 2 columns each, then 1 block of 1 column
    
    for b in range(4):
        k0 = 2 * b      # First column of this block
        k1 = 2 * b + 1  # Second column of this block
        
        # Strassen's 7 multiplications for 2x2x2 block
        # Indices in U (flattened A, shape 2x2):
        u_a = 0 * m + 0  # A[0,0]
        u_b = 0 * m + 1  # A[0,1]
        u_c = 1 * m + 0  # A[1,0]
        u_d = 1 * m + 1  # A[1,1]
        
        # Indices in V (flattened B, shape 2x2 for this block):
        v_e = 0 * p + k0  # B[0,k0]
        v_f = 0 * p + k1  # B[0,k1]
        v_g = 1 * p + k0  # B[1,k0]
        v_h = 1 * p + k1  # B[1,k1]
        
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # W indices (flattened C, shape 2x2 for this block):
        w_00 = k0 * n + 0  # C[0,k0]
        w_01 = k1 * n + 0  # C[0,k1]
        w_10 = k0 * n + 1  # C[1,k0]
        w_11 = k1 * n + 1  # C[1,k1]
        
        # C[0,k0] = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C[0,k1] = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C[1,k0] = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1,k1] = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # Last block: 2x2x1 (k=8)
    k8 = 8
    v_e = 0 * p + k8  # B[0,8]
    v_g = 1 * p + k8  # B[1,8]
    
    u_a = 0 * m + 0  # A[0,0]
    u_b = 0 * m + 1  # A[0,1]
    u_c = 1 * m + 0  # A[1,0]
    u_d = 1 * m + 1  # A[1,1]
    
    # P1 = a*e
    U[u_a][col] += 1
    V[v_e][col] += 1
    
    # P2 = b*g
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    
    # P3 = c*e
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    
    # P4 = d*g
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0  # C[0,8]
    w_18 = k8 * n + 1  # C[1,8]
    
    # C[0,8] = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C[1,8] = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # === Part 2: 2x1x9 ===
    # A[0:2,2] is a 2x1 vector, B[2,:] is a 1x9 vector
    # C[0:2,:] = A[0:2,2] * B[2,:] (outer product)
    
    u_ar0 = 0 * m + 2  # A[0,2]
    u_ar1 = 1 * m + 2  # A[1,2]
    
    for k in range(9):
        v_b = 2 * p + k  # B[2,k]
        
        # Q_k = A[0,2] * B[2,k]
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A[1,2] * B[2,k]
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    # Verify col == rank
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
