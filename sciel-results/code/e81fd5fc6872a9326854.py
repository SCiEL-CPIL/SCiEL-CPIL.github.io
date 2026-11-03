import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def build_strassen_56():
        """Build the standard 56-rank Strassen decomposition."""
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1
        
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[0,6] = 1
        
        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
                if s == 0:
                    b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1]
                else:
                    b_map = [2*8+2*j, 2*8+2*j+1, 3*8+2*j, 3*8+2*j+1]
                c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val
        
        return U, V, W

    def build_direct_32():
        """Try to build a rank-32 decomposition using 2x4*4x4 blocks of rank 16."""
        # Split 8-column output into two 4-column blocks
        # Each 2x4 * 4x4 can potentially be done with rank 16
        # Using a Strassen-like decomposition for 2x4 * 4x4
        
        # For 2x4 * 4x4: view as computing C[i,k] = sum_j A[i,j]*B[j,k]
        # Split A into columns [A1, A2] (each 2x2), B into rows [[B1],[B2]] (each 2x4)
        # C = A1*B1 + A2*B2
        # Each 2x2 * 2x4: split 4 cols into two 2-col blocks, each 2x2*2x2 rank 7
        # So each 2x2*2x4 is rank 14, total 28. Not 16.
        
        # Alternative: use the known rank-16 for 2x4*4x2 (transpose case)
        # Actually let's try a different construction
        
        # Direct rank-32: use 4 independent 2x2*2x4 multiplications
        # 2x4 * 4x8: split middle dim into two 2s
        # A = [A1 | A2], B = [B1; B2] where A1,A2 are 2x2, B1,B2 are 2x8
        # C = A1*B1 + A2*B2
        # Each 2x2*2x8: split 8 cols into four 2-col blocks, each 2x2*2x2 rank 7
        # Each 2x2*2x8 is rank 28. Total 56. Same.
        
        # Let me try a completely different approach:
        # Use the fact that we can share some Strassen products
        
        # Actually, let's just return the 56-rank for now and try to verify
        return None
    
    def verify_decomposition(U, V, W):
        """Verify tensor equation: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False, (i, j, k, s)
        return True, None
    
    U56, V56, W56 = build_strassen_56()
    valid, err = verify_decomposition(U56, V56, W56)
    
    if not valid:
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition by searching
    best_U, best_V, best_W = U56, V56, W56
    best_rank = 56
    
    if time.time() < deadline:
        # Try rank 48, 40, 32 with random search
        for target_rank in [48, 40, 36, 32]:
            if time.time() >= deadline:
                break
            found = False
            for attempt in range(50):
                if time.time() >= deadline:
                    break
                U_rand = np.zeros((8, target_rank), dtype=int)
                V_rand = np.zeros((32, target_rank), dtype=int)
                W_rand = np.zeros((16, target_rank), dtype=int)
                
                # Initialize with random ±1 patterns
                for r in range(target_rank):
                    # Randomly assign nonzero entries
                    for i in range(8):
                        if np.random.random() < 0.5:
                            U_rand[i, r] = np.random.choice([-1, 1])
                    for j in range(32):
                        if np.random.random() < 0.3:
                            V_rand[j, r] = np.random.choice([-1, 1])
                    for k in range(16):
                        if np.random.random() < 0.5:
                            W_rand[k, r] = np.random.choice([-1, 1])
                
                valid_r, _ = verify_decomposition(U_rand, V_rand, W_rand)
                if valid_r:
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
