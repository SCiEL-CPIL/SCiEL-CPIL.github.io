"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
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
    
    # Verify a decomposition is exact for a block (n, m, p_block)
    def verify_block(U, V, W, n, m, p_block, tol=1e-9):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p_block+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Search for a decomposition of M_{2,3,5}
    def find_block_decomposition(n, m, p_block, target_rank, max_attempts=1000):
        # Try to find a rank target_rank decomposition for n x m x p_block
        # Using half-integers {-1, -0.5, 0, 0.5, 1}
        # We can also try integers {-2, -1, 0, 1, 2}
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            # Scale factors to try: 0.5, 1.0
            scale = 0.5 if attempt % 2 == 0 else 1.0
            max_val = int(2 / scale) # If scale 0.5, max 4 -> -2..2 * 0.5 = -1..1
                                     # If scale 1.0, max 2 -> -2..2
            
            U_rand = np.random.randint(-max_val, max_val + 1, (n*m, target_rank)) * scale
            V_rand = np.random.randint(-max_val, max_val + 1, (m*p_block, target_rank)) * scale
            W_rand = np.random.randint(-max_val, max_val + 1, (p_block*n, target_rank)) * scale
            
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if verify_block(U_list, V_list, W_list, n, m, p_block):
                return {"U": U_list, "V": V_list, "W": W_list}
                
        return None

    # Strategy:
    # 1. Try to find rank 14 for 2x3x5. If found, construct rank 28 for 2x3x10.
    # 2. Try rank 15, 16, 17, 18 for 2x3x5.
    # 3. If none, return schoolbook.
    
    block_n, block_m, block_p = 2, 3, 5
    
    best_full_result = None
    best_rank = n * m * p
    
    # Try ranks for the block
    # Known lower bound for 2x3x5 is 14.
    target_block_ranks = [14, 15, 16, 17, 18, 19, 20]
    
    for r_block in target_block_ranks:
        if time.time() > deadline - 10:
            break
            
        # Budget for search: more attempts for lower ranks (harder) or fewer?
        # Usually lower rank is harder. But 14 is the min.
        # Let's give 200 attempts for 14, 100 for others.
        attempts = 200 if r_block == 14 else 100
        
        block_decomp = find_block_decomposition(block_n, block_m, block_p, r_block, attempts)
        
        if block_decomp is not None:
            # Construct full 2x3x10 decomposition
            # U_full: 6 x (2 * r_block)
            # V_full: 30 x (2 * r_block)
            # W_full: 20 x (2 * r_block)
            
            U_block = block_decomp["U"]
            V_block = block_decomp["V"]
            W_block = block_decomp["W"]
            
            U_full = [[0] * (2 * r_block) for _ in range(block_n * block_m)]
            V_full = [[0] * (2 * r_block) for _ in range(block_m * 10)]
            W_full = [[0] * (2 * r_block) for _ in range(10 * block_n)]
            
            # Block 1: k in 0..4
            # U: same for both blocks in terms of indices i,j, so we put U_block in first r_block cols
            # V: rows j*5+k for k=0..4 -> j*5+k. Put V_block in first r_block cols
            # W: rows k*2+i for k=0..4 -> k*2+i. Put W_block in first r_block cols
            
            for i in range(block_n * block_m):
                for r in range(r_block):
                    U_full[i][r] = U_block[i][r]
                    U_full[i][r + r_block] = U_block[i][r] # Wait, U is same?
                    # Actually, U depends on i,j. The tensor is T[i,j,k].
                    # For block 1, T[i,j,k] = sum U[i,j,r] V[j,k,r] W[k,i,r].
                    # For block 2, T[i,j,k+5] = sum U'[i,j,r'] V'[j,k+5,r'] W'[k+5,i,r'].
                    # We can set U' = U, V' = V (shifted k), W' = W (shifted k).
                    # So U_full columns 0..r-1 are U_block, columns r..2r-1 are U_block.
            
            for j in range(block_m):
                for k in range(block_p):
                    # V index: j*10 + k
                    # Block 1 V index: j*5 + k
                    V_full[j * 10 + k][r_block] = 0 # Placeholder, will fill below
                    # Actually, let's just copy V_block to the corresponding rows in V_full
                    # V_block row index: j*5 + k
                    # V_full row index: j*10 + k
                    pass
            
            # Let's do this more carefully
            # U_full
            for i in range(block_n * block_m):
                for r in range(r_block):
                    U_full[i][r] = U_block[i][r]
                    U_full[i][r + r_block] = U_block[i][r]
            
            # V_full
            for j in range(block_m):
                for k in range(block_p):
                    v_idx_block = j * block_p + k
                    v_idx_full_1 = j * 10 + k
                    v_idx_full_2 = j * 10 + (k + 5)
                    
                    for r in range(r_block):
                        V_full[v_idx_full_1][r] = V_block[v_idx_block][r]
                        V_full[v_idx_full_2][r + r_block] = V_block[v_idx_block][r]
            
            # W_full
            for k in range(block_p):
                for i in range(block_n):
                    w_idx_block = k * block_n + i
                    w_idx_full_1 = k * block_n + i
                    w_idx_full_2 = (k + 5) * block_n + i
                    
                    for r in range(r_block):
                        W_full[w_idx_full_1][r] = W_block[w_idx_block][r]
                        W_full[w_idx_full_2][r + r_block] = W_block[w_idx_block][r]
            
            # Verify full decomposition
            # Re-verify just in case
            if verify_block(U_full, V_full, W_full, n, m, p, tol=1e-9):
                best_full_result = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = 2 * r_block
                break # Found a valid decomposition, move to next lower rank? No, we started at lowest.
            else:
                # Construction error? Should not happen if block was valid.
                print("Construction verification failed")
                continue
        
        # If we found a rank, we break the loop because we iterate from lowest rank up.
        # Wait, the loop is `for r_block in target_block_ranks`.
        # If we find rank 14, we break.
        # If we don't find 14, we try 15, etc.
        # So if best_full_result is not None, we should break.
        if best_full_result is not None:
            break

    if best_full_result is not None:
        return best_full_result
    else:
        return make_schoolbook()
