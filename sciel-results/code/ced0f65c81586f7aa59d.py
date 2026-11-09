import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Strategy: Partition 4x5x5 into four disjoint 2x2x2 blocks + one 2x3x3 block
    # - 4 Strassen 2x2x2 blocks: rank 7 each = 28
    # - 1 rank-17 2x3x3 decomposition: rank 17
    # Total: 45
    
    # Partition:
    #   Block A: i in {0,1}, j in {0,1}, k in {0,1}  (2x2x2)
    #   Block B: i in {2,3}, j in {0,1}, k in {0,1}  (2x2x2)
    #   Block C: i in {0,1}, j in {2,3}, k in {0,1}  (2x2x2)
    #   Block D: i in {2,3}, j in {2,3}, k in {0,1}  (2x2x2)
    #   Block E: i in {0,1}, j in {0,1,2,3,4}, k in {2,3,4}  (2x5x3 -> needs 2x3x3 sub-blocks)
    #   Wait, that's not right. Let me reconsider.
    #
    # Better partition:
    #   Four 2x2x2 blocks covering i in {0,1},{2,3}, j in {0,1},{2,3}, k in {0,1}
    #   Remaining: all (i,j,k) with k in {2,3,4} -> 4x5x3 = 60 entries
    #   That's too many. Let me think differently.
    #
    # Alternative: Use 2x3x3 blocks.
    #   Block 1: i in {0,1}, j in {0,1,2}, k in {0,1,2}  (2x3x3, rank 17)
    #   Block 2: i in {0,1}, j in {3,4}, k in {0,1,2}  (2x2x3)
    #   Block 3: i in {2,3}, j in {0,1,2}, k in {0,1,2}  (2x3x3, rank 17)
    #   Block 4: i in {2,3}, j in {3,4}, k in {0,1,2}  (2x2x3)
    #   Block 5: all i, j, k in {3,4}  (4x5x2 = 40 entries)
    #
    # This is getting complicated. Let me use a simpler approach:
    # Just apply Strassen to 8 disjoint 2x2x2 blocks (as before) = rank 92
    # Then try to find a 2x3x3 rank-17 decomposition for the remaining part.
    #
    # Actually, let me try a cleaner partition:
    #   4x5x5 = 4x(3+2) x (3+2)
    #   = 4x3x3 + 4x3x2 + 4x2x3 + 4x2x2
    #
    # For the 4x3x3 part, I can use two 2x3x3 blocks (each rank 17) = 34
    # For the 4x3x2 part, I can use two 2x3x2 blocks
    # For the 4x2x3 part, I can use two 2x2x3 blocks
    # For the 4x2x2 part, I can use two 2x2x2 blocks (each rank 7) = 14
    #
    # But I need rank-17 decompositions for 2x3x3, and rank decompositions for 2x3x2, 2x2x3.
    #
    # Known: 2x3x3 has rank 17 (or 18?). Let me verify.
    # Actually, the best known rank for 2x3x3 is 17.
    #
    # Let me just implement the 4x2x2x2 Strassen approach (rank 92) and see if I can
    # find a better partition with known decompositions.
    
    # For now, let's use the proven rank-92 Strassen decomposition
    # and try to improve it.
    
    def build_strassen_decomposition():
        rank = 92
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        blocks = []
        for i_base in [0, 2]:
            for j_base in [0, 2]:
                for k_base in [0, 2]:
                    blocks.append((
                        [i_base, i_base + 1],
                        [j_base, j_base + 1],
                        [k_base, k_base + 1]
                    ))
        
        skip_set = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        skip_set.add((i, j, k))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in skip_set:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        for I, J, K in blocks:
            i1, i2 = I
            j1, j2 = J
            k1, k2 = K
            
            # M1 = (a11 + a22)(b11 + b22)
            U[i1 * m + j1][col] += 1
            U[i2 * m + j2][col] += 1
            V[j1 * p + k1][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[i2 * m + j1][col] += 1
            U[i2 * m + j2][col] += 1
            V[j1 * p + k1][col] += 1
            W[k1 * n + i2][col] += 1
            W[k2 * n + i2][col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[i1 * m + j1][col] += 1
            V[j1 * p + k2][col] += 1
            V[j2 * p + k2][col] += -1
            W[k2 * n + i1][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[i2 * m + j2][col] += 1
            V[j2 * p + k1][col] += 1
            V[j1 * p + k1][col] += -1
            W[k1 * n + i1][col] += 1
            W[k1 * n + i2][col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[i1 * m + j1][col] += 1
            U[i1 * m + j2][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += -1
            W[k2 * n + i1][col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[i2 * m + j1][col] += 1
            U[i1 * m + j1][col] += -1
            V[j1 * p + k1][col] += 1
            V[j1 * p + k2][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[i1 * m + j2][col] += 1
            U[i2 * m + j2][col] += -1
            V[j2 * p + k1][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += 1
            col += 1
        
        return U, V, W, rank
    
    # Try to build a rank-45 decomposition using 2x3x3 blocks
    # Partition: i in {0,1},{2,3}; j in {0,1,2},{3,4}; k in {0,1,2},{3,4}
    # 4 blocks of 2x3x3 (rank 17 each) = 68, plus remaining 2x2x2 blocks
    # This doesn't work out cleanly.
    
    # Let me try: 4x5x5 = (2x2 + 2x3) x (2x2 + 2x3) x (2x2 + 2x3)
    # No, that's not how tensor partitioning works.
    
    # Simpler: Just use the rank-92 decomposition for now.
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is correct."""
        nm, mp, pn = n*m, m*p, p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to standard decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_rank = rank
    
    # Check for all-zero columns
    for r in range(current_rank):
        if all(U_np[i][r] == 0 for i in range(nm)) and \
           all(V_np[i][r] == 0 for i in range(mp)) and \
           all(W_np[i][r] == 0 for i in range(pn)):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            current_rank -= 1
            r -= 1
    
    # Convert back to lists with integer/half-integer entries
    U_final = []
    for i in range(nm):
        row = []
        for r in range(current_rank):
            val = U_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        U_final.append(row)
    
    V_final = []
    for i in range(mp):
        row = []
        for r in range(current_rank):
            val = V_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        V_final.append(row)
    
    W_final = []
    for i in range(pn):
        row = []
        for r in range(current_rank):
            val = W_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        W_final.append(row)
    
    return {"U": U_final, "V": V_final, "W": W_final}
