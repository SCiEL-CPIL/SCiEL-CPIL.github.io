"""Strassen-based rank-28 decomposition for 2x4 by 4x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is p x n
    # n=2, m=4, p=4
    
    # Strassen rank-7 coefficients for 2x2x2
    # Local A indices: 0=a11, 1=a12, 2=a21, 3=a22
    # Local B indices: 0=b11, 1=b12, 2=b21, 3=b22
    # M1 = (a11+a22) * (b11+b22)
    # M2 = (a21+a22) * b11
    # M3 = a11 * (b12-b22)
    # M4 = a22 * (b21-b11)
    # M5 = (a11+a12) * b22
    # M6 = (a21-a11) * (b11+b12)
    # M7 = (a12-a22) * (b21+b22)
    
    # U coefficients (A part): list of (index, value)
    SU = [
        [(0, 1), (3, 1)],    # M1
        [(2, 1), (3, 1)],    # M2
        [(0, 1)],            # M3
        [(3, 1)],            # M4
        [(0, 1), (1, 1)],    # M5
        [(2, 1), (0, -1)],   # M6
        [(1, 1), (3, -1)],   # M7
    ]
    
    # V coefficients (B part)
    SV = [
        [(0, 1), (3, 1)],    # M1
        [(0, 1)],            # M2
        [(1, 1), (3, -1)],   # M3
        [(2, 1), (0, -1)],   # M4
        [(3, 1)],            # M5
        [(0, 1), (1, 1)],    # M6
        [(2, 1), (3, 1)],    # M7
    ]
    
    # W coefficients (C part)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M5 + M6
    # C21 = M2 + M3
    # C22 = M1 - M3 + M4 + M6
    SW = [
        [(0, 1), (3, -1), (4, 1), (6, -1)],  # C11 (local idx 0)
        [(4, 1), (5, 1)],                    # C12 (local idx 1)
        [(1, 1), (2, 1)],                    # C21 (local idx 2)
        [(0, 1), (2, -1), (3, 1), (5, 1)]   # C22 (local idx 3)
    ]
    
    # Build the rank-28 decomposition
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Mapping of local 2x2 indices to global flat indices
    # A is 2x4. We partition A into two 2x2 blocks horizontally: A_L (cols 0,1) and A_R (cols 2,3)
    # A_L:
    #   a11 -> A[0,0] -> flat 0
    #   a12 -> A[0,1] -> flat 1
    #   a21 -> A[1,0] -> flat 4
    #   a22 -> A[1,1] -> flat 5
    # A_R:
    #   a11 -> A[0,2] -> flat 2
    #   a12 -> A[0,3] -> flat 3
    #   a21 -> A[1,2] -> flat 6
    #   a22 -> A[1,3] -> flat 7
    
    # B is 4x4. Partition into four 2x2 blocks:
    # B_TL (rows 0,1; cols 0,1)
    # B_TR (rows 0,1; cols 2,3)
    # B_BL (rows 2,3; cols 0,1)
    # B_BR (rows 2,3; cols 2,3)
    
    # B_TL:
    #   b11 -> B[0,0] -> flat 0
    #   b12 -> B[0,1] -> flat 1
    #   b21 -> B[1,0] -> flat 4
    #   b22 -> B[1,1] -> flat 5
    # B_TR:
    #   b11 -> B[0,2] -> flat 2
    #   b12 -> B[0,3] -> flat 3
    #   b21 -> B[1,2] -> flat 6
    #   b22 -> B[1,3] -> flat 7
    # B_BL:
    #   b11 -> B[2,0] -> flat 8
    #   b12 -> B[2,1] -> flat 9
    #   b21 -> B[3,0] -> flat 12
    #   b22 -> B[3,1] -> flat 13
    # B_BR:
    #   b11 -> B[2,2] -> flat 10
    #   b12 -> B[2,3] -> flat 11
    #   b21 -> B[3,2] -> flat 14
    #   b22 -> B[3,3] -> flat 15
    
    # C is 2x4. Output indices in W are k*n + i (trace convention W[k*n+i, r])
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,j] is the output.
    # Tensor T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
    # Here i is index into A (size n*m), j is index into B (size m*p), k is index into C (size p*n).
    # Wait, standard matrix mult tensor is T_{i,j,k} where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The indices in the problem statement:
    # U: (n*m, rank) -> index i
    # V: (m*p, rank) -> index j
    # W: (p*n, rank) -> index k
    # The tensor identity is: sum_c U[i,c] V[j,c] W[k,c] = delta_{i,j,k} for the multiplication tensor.
    # The multiplication tensor has 1s at (i, j, k) such that A_{i_row, i_col} * B_{j_row, j_col} contributes to C_{k_row, k_col}.
    # Specifically, if A is n x m, B is m x p, C is n x p.
    # But here W is p x n. This implies the output C is treated as p x n?
    # "Matrix multiplication 2,4,4" -> A(2x4), B(4x4), C(2x4).
    # If W is (p*n, rank) = (8, rank), and the convention is W[k*n+i, r], then the index k in W corresponds to the column of C?
    # Let's check the seed program:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Loop: for i in range(n), for j in range(m), for k in range(p)
    # U[i*m+j] = 1
    # V[j*p+k] = 1
    # W[k*n+i] = 1
    # This implies:
    # i is row of A (0..n-1)
    # j is col of A / row of B (0..m-1)
    # k is col of B / col of C (0..p-1)
    # W index is k*n + i.
    # This means W index corresponds to C element at (i, k) but flattened as k*n+i.
    # So C is n x p, but flattened column-major? Or just index mapping.
    # C[i, k] is at W[k*n + i].
    
    # Let's map the 2x2 blocks of C to W indices.
    # C is 2x4.
    # C[0, 0] -> W[0*2 + 0] = 0
    # C[0, 1] -> W[1*2 + 0] = 2
    # C[0, 2] -> W[2*2 + 0] = 4
    # C[0, 3] -> W[3*2 + 0] = 6
    # C[1, 0] -> W[0*2 + 1] = 1
    # C[1, 1] -> W[1*2 + 1] = 3
    # C[1, 2] -> W[2*2 + 1] = 5
    # C[1, 3] -> W[3*2 + 1] = 7
    
    # Local C 2x2 block mapping:
    # Local (0,0) -> Global (0,0) -> W[0]
    # Local (0,1) -> Global (0,1) -> W[2]
    # Local (1,0) -> Global (1,0) -> W[1]
    # Local (1,1) -> Global (1,1) -> W[3]
    # So for a C block at column offset col_off (0 or 2):
    # Local 0 (C11) -> W[col_off*2 + 0]
    # Local 1 (C12) -> W[col_off*2 + 2]
    # Local 2 (C21) -> W[col_off*2 + 1]
    # Local 3 (C22) -> W[col_off*2 + 3]
    
    # We have 4 blocks of multiplication:
    # 1. A_L * B_TL -> C[0:2, 0:2] (col_off 0)
    # 2. A_L * B_TR -> C[0:2, 2:4] (col_off 2)
    # 3. A_R * B_BL -> C[0:2, 0:2] (col_off 0)
    # 4. A_R * B_BR -> C[0:2, 2:4] (col_off 2)
    
    # Block 1: A_L * B_TL
    u_map_1 = [0, 1, 4, 5]  # A_L flat indices
    v_map_1 = [0, 1, 4, 5]  # B_TL flat indices
    w_map_1 = [0, 2, 1, 3]  # C[0:2, 0:2] W indices (Local 0->0, 1->2, 2->1, 3->3)
    
    # Block 2: A_L * B_TR
    u_map_2 = [0, 1, 4, 5]  # A_L
    v_map_2 = [2, 3, 6, 7]  # B_TR
    w_map_2 = [4, 6, 5, 7]  # C[0:2, 2:4] W indices (Local 0->4, 1->6, 2->5, 3->7)
    
    # Block 3: A_R * B_BL
    u_map_3 = [2, 3, 6, 7]  # A_R
    v_map_3 = [8, 9, 12, 13] # B_BL
    w_map_3 = [0, 2, 1, 3]  # C[0:2, 0:2] (Same as block 1)
    
    # Block 4: A_R * B_BR
    u_map_4 = [2, 3, 6, 7]  # A_R
    v_map_4 = [10, 11, 14, 15] # B_BR
    w_map_4 = [4, 6, 5, 7]  # C[0:2, 2:4] (Same as block 2)
    
    blocks = [
        (u_map_1, v_map_1, w_map_1, 0),
        (u_map_2, v_map_2, w_map_2, 7),
        (u_map_3, v_map_3, w_map_3, 14),
        (u_map_4, v_map_4, w_map_4, 21)
    ]
    
    for u_map, v_map, w_map, start_col in blocks:
        for k in range(7):
            col = start_col + k
            # Fill U
            for idx, val in SU[k]:
                U[u_map[idx], col] += val
            # Fill V
            for idx, val in SV[k]:
                V[v_map[idx], col] += val
            # Fill W
            for idx, val in SW[k]:
                W[w_map[idx], col] += val
                
    # Validation
    # Check if U, V, W decompose the tensor correctly
    # T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
    # Expected: 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise.
    # i is A index (n*m), j is B index (m*p), k is C index (p*n)
    # A index i -> row i//m, col i%m
    # B index j -> row j//p, col j%p
    # C index k -> row k%n, col k//n (based on W[k*n+i] -> i is row, k is col? No, W index is k*n+i. So row=i, col=k? 
    # Wait. W index = k*n + i. 
    # In seed: W[k*n+i] = 1 for A[i,j] B[j,k].
    # So W index corresponds to (i, k) where i is row of C, k is col of C.
    # So C row = W_idx % n, C col = W_idx // n.
    
    valid = True
    # Sample some entries to verify
    # Full check is 8*16*8 = 1024 entries.
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                # Calculate expected value
                a_row = i // m
                a_col = i % m
                b_row = j // p
                b_col = j % p
                c_row = k % n
                c_col = k // n
                
                # A[a_row, a_col] * B[b_row, b_col] contributes to C[c_row, c_col]
                # This is non-zero if a_row == c_row and a_col == b_row and b_col == c_col
                expected = 1.0 if (a_row == c_row and a_col == b_row and b_col == c_col) else 0.0
                
                # Calculate sum
                s = 0.0
                # Use dot product for speed
                s = np.dot(U[i], V[j] * W[k])
                
                if not np.isclose(s, expected, atol=1e-8):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to rank 32
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
