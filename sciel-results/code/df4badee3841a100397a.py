import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We use a 2+2+1 partition for each dimension.
    # Indices 0,1 are group A; 2,3 are group B; 4 is group C.
    # Groups: 0 (size 2), 1 (size 2), 2 (size 1)
    
    # Total rank is sum of ranks of all 27 blocks.
    # Block (i, j, k) where i,j,k in {0, 1, 2}
    # Size of block is s_i * s_j * s_k where s = [2, 2, 1]
    # Rank of block:
    # If sizes are (2,2,2): Rank 7 (Strassen)
    # If sizes are (2,2,1) perms: Rank 4
    # If sizes are (2,1,1) perms: Rank 2
    # If sizes are (1,1,1): Rank 1
    
    # We construct U, V, W matrices of size 25 x Rank
    # Rank = 8*7 + 12*4 + 6*2 + 1*1 = 56 + 48 + 12 + 1 = 117
    
    rank = 117
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Define the groups and their indices
    # Group 0: indices 0, 1
    # Group 1: indices 2, 3
    # Group 2: index 4
    
    groups = [
        [0, 1],
        [2, 3],
        [4]
    ]
    
    # Strassen decomposition for 2x2x2
    # U, V, W for 2x2x2 Strassen
    # Standard Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We need to map these to the global U, V, W.
    # For a specific block (gi, gj, gk) of size 2x2x2:
    # The local U (2x2) has entries for the 7 terms.
    # The local V (2x2) has entries for the 7 terms.
    # The local W (2x2) has entries for the 7 terms.
    
    # Let's define the Strassen coefficients for U, V, W.
    # U[i, term], V[j, term], W[k, term]
    # i, j, k in 0, 1
    
    # Term 1: (A11+A22)(B11+B22)
    # U: 1, 1 in col 0? No, U is for A.
    # A11 is at (0,0), A22 at (1,1).
    # So U[0,0]=1, U[1,1]=1.
    # B11 is at (0,0), B22 at (1,1).
    # So V[0,0]=1, V[1,1]=1.
    # W[0,0]=1, W[1,1]=1.
    
    # This is getting complex to hardcode for all 27 blocks with correct indexing.
    # I will generate the Strassen tensors for 2x2x2 and then embed them.
    
    # Strassen 2x2x2 tensors
    # U_s, V_s, W_s of shape (4, 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Indices in 2x2: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    # Let's use standard 1-based for Strassen formulas, convert to 0-based.
    # A11=0, A12=1, A21=2, A22=3
    
    # M1 = (A11+A22)(B11+B22)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1
    
    # M3 = A11(B12-B22)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = -1
    
    # M4 = A22(B21-B11)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[2, 3] = 1; W_s[0, 3] = -1
    
    # M5 = (A11+A12)B22
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = 1; W_s[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[2, 5] = 1; W_s[1, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[1, 6] = 1; W_s[3, 6] = 1
    
    # Now iterate over blocks
    col = 0
    for gi in range(3):
        for gj in range(3):
            for gk in range(3):
                si = len(groups[gi])
                sj = len(groups[gj])
                sk = len(groups[gk])
                
                # Determine rank of this block
                if si == 2 and sj == 2 and sk == 2:
                    r = 7
                elif si == 2 and sj == 2 and sk == 1:
                    r = 4
                elif si == 2 and sj == 1 and sk == 2:
                    r = 4
                elif si == 1 and sj == 2 and sk == 2:
                    r = 4
                elif si == 2 and sj == 1 and sk == 1:
                    r = 2
                elif si == 1 and sj == 2 and sk == 1:
                    r = 2
                elif si == 1 and sj == 1 and sk == 2:
                    r = 2
                else: # 1,1,1
                    r = 1
                
                # Map local indices to global indices
                # U index: i_global * m + j_global
                # V index: j_global * p + k_global
                # W index: k_global * n + i_global
                
                # For simplicity, we handle each block type separately.
                
                if r == 7:
                    # Embed Strassen
                    for local_i in range(2):
                        for local_j in range(2):
                            pass # U is built from local_i, local_j? No, U is A.
                            # A index is (i, j).
                            # U[row, col] where row = i*m + j.
                    
                    # We need to fill U, V, W for this block.
                    # Local A indices: (li, lj) in 0..1
                    # Local B indices: (lj, lk) in 0..1
                    # Local C indices: (lk, li) in 0..1
                    
                    # Global indices:
                    # i_g = groups[gi][li]
                    # j_g = groups[gj][lj]
                    # k_g = groups[gk][lk]
                    
                    for t in range(7):
                        # U
                        for li in range(2):
                            for lj in range(2):
                                # U_s index is li*2 + lj
                                val = U_s[li*2 + lj, t]
                                if val != 0:
                                    i_g = groups[gi][li]
                                    j_g = groups[gj][lj]
                                    U[i_g * m + j_g, col + t] = val
                                    
                        # V
                        for lj in range(2):
                            for lk in range(2):
                                val = V_s[lj*2 + lk, t]
                                if val != 0:
                                    j_g = groups[gj][lj]
                                    k_g = groups[gk][lk]
                                    V[j_g * p + k_g, col + t] = val
                                    
                        # W
                        for lk in range(2):
                            for li in range(2):
                                val = W_s[lk*2 + li, t]
                                if val != 0:
                                    k_g = groups[gk][lk]
                                    i_g = groups[gi][li]
                                    W[k_g * n + i_g, col + t] = val
                                    
                elif r == 4:
                    # Block is 2x2x1 or permutations.
                    # We can use schoolbook for these, or a specific 4-rank decomp.
                    # Schoolbook for 2x2x1:
                    # A(2,2) * B(2,1) = C(2,1)
                    # C0 = A00 B0 + A01 B1
                    # C1 = A10 B0 + A11 B1
                    # Rank 4.
                    
                    # Identify which dimension is size 1.
                    # Case 1: sk == 1 (2x2x1)
                    # A is 2x2, B is 2x1, C is 2x1.
                    # U (A): 2x2. V (B): 2x1. W (C): 2x1.
                    # Terms:
                    # 1: A00 * B0 -> C0
                    # 2: A01 * B1 -> C0
                    # 3: A10 * B0 -> C1
                    # 4: A11 * B1 -> C1
                    
                    if sk == 1:
                        for t in range(4):
                            if t == 0:
                                U[groups[gi][0]*m + groups[gj][0], col+t] = 1
                                V[groups[gj][0]*p + groups[gk][0], col+t] = 1
                                W[groups[gk][0]*n + groups[gi][0], col+t] = 1
                            elif t == 1:
                                U[groups[gi][0]*m + groups[gj][1], col+t] = 1
                                V[groups[gj][1]*p + groups[gk][0], col+t] = 1
                                W[groups[gk][0]*n + groups[gi][0], col+t] = 1
                            elif t == 2:
                                U[groups[gi][1]*m + groups[gj][0], col+t] = 1
                                V[groups[gj][0]*p + groups[gk][0], col+t] = 1
                                W[groups[gk][0]*n + groups[gi][1], col+t] = 1
                            elif t == 3:
                                U[groups[gi][1]*m + groups[gj][1], col+t] = 1
                                V[groups[gj][1]*p + groups[gk][0], col+t] = 1
                                W[groups[gk][0]*n + groups[gi][1], col+t] = 1
                    # Case 2: sj == 1 (2x1x2)
                    # A is 2x1, B is 1x2, C is 2x2.
                    # C00 = A0 B0, C01 = A0 B1, C10 = A1 B0, C11 = A1 B1
                    elif sj == 1:
                        for t in range(4):
                            if t == 0:
                                U[groups[gi][0]*m + groups[gj][0], col+t] = 1
                                V[groups[gj][0]*p + groups[gk][0], col+t] = 1
                                W[groups[gk][0]*n + groups[gi][0], col+t] = 1
                            elif t == 1:
                                U[groups[gi][0]*m + groups[gj][0], col+t] = 1
                                V[groups[gj][0]*p + groups[gk][1], col+t] = 1
                                W[groups[gk][1]*n + groups[gi][0], col+t] = 1
                            elif t == 2:
                                U[groups[gi][1]*m + groups[gj][0], col+t] = 1
                                V[groups[gj][0]*p + groups[gk][0], col+t] = 1
                                W[groups[gk][0]*n + groups[gi][1], col+t] = 1
                            elif t == 3:
                                U[groups[gi][1]*m + groups[gj][0], col+t] = 1
                                V[groups[gj][0]*p + groups[gk][1], col+t] = 1
                                W[groups[gk][1]*n + groups[gi][1], col+t] = 1
                    # Case 3: si == 1 (1x2x2)
                    # A is 1x2, B is 2x2, C is 2x2.
                    # C00 = A0 B00, C01 = A0 B01, C10 = A1 B10, C11 = A1 B11
                    elif si == 1:
                        for t in range(4):
                            if t == 0:
                                U[groups[gi][0]*m + groups[gj][0], col+t] = 1
                                V[groups[gj][0]*p + groups[gk][0], col+t] = 1
                                W[groups[gk][0]*n + groups[gi][0], col+t] = 1
                            elif t == 1:
                                U[groups[gi][0]*m + groups[gj][1], col+t] = 1
                                V[groups[gj][1]*p + groups[gk][0], col+t] = 1
                                W[groups[gk][0]*n + groups[gi][0], col+t] = 1
                            elif t == 2:
                                U[groups[gi][0]*m + groups[gj][0], col+t] = 1
                                V[groups[gj][0]*p + groups[gk][1], col+t] = 1
                                W[groups[gk][1]*n + groups[gi][0], col+t] = 1
                            elif t == 3:
                                U[groups[gi][0]*m + groups[gj][1], col+t] = 1
                                V[groups[gj][1]*p + groups[gk][1], col+t] = 1
                                W[groups[gk][1]*n + groups[gi][0], col+t] = 1
                elif r == 2:
                    # Block is 2x1x1 or perms.
                    # A(2,1) * B(1,1) = C(2,1)
                    # C0 = A0 B, C1 = A1 B
                    if si == 2:
                        U[groups[gi][0]*m + groups[gj][0], col+0] = 1
                        V[groups[gj][0]*p + groups[gk][0], col+0] = 1
                        W[groups[gk][0]*n + groups[gi][0], col+0] = 1
                        
                        U[groups[gi][1]*m + groups[gj][0], col+1] = 1
                        V[groups[gj][0]*p + groups[gk][0], col+1] = 1
                        W[groups[gk][0]*n + groups[gi][1], col+1] = 1
                    elif sj == 2:
                        U[groups[gi][0]*m + groups[gj][0], col+0] = 1
                        V[groups[gj][0]*p + groups[gk][0], col+0] = 1
                        W[groups[gk][0]*n + groups[gi][0], col+0] = 1
                        
                        U[groups[gi][0]*m + groups[gj][1], col+1] = 1
                        V[groups[gj][1]*p + groups[gk][0], col+1] = 1
                        W[groups[gk][0]*n + groups[gi][0], col+1] = 1
                    elif sk == 2:
                        U[groups[gi][0]*m + groups[gj][0], col+0] = 1
                        V[groups[gj][0]*p + groups[gk][0], col+0] = 1
                        W[groups[gk][0]*n + groups[gi][0], col+0] = 1
                        
                        U[groups[gi][0]*m + groups[gj][0], col+1] = 1
                        V[groups[gj][0]*p + groups[gk][1], col+1] = 1
                        W[groups[gk][1]*n + groups[gi][0], col+1] = 1
                else: # r == 1
                    U[groups[gi][0]*m + groups[gj][0], col+0] = 1
                    V[groups[gj][0]*p + groups[gk][0], col+0] = 1
                    W[groups[gk][0]*n + groups[gi][0], col+0] = 1
                    
                col += r
                
    # Convert to lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
