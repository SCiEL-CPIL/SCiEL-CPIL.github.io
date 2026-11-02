import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition for n x m x p matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def find_3x2x2_decomposition(max_time=100):
    """
    Find a rank-11 decomposition for 3x2x2 matrix multiplication.
    Returns (U, V, W) or None if not found.
    """
    n, m, p = 3, 2, 2
    rank = 11
    # Indices
    # U: 6x11
    # V: 4x11
    # W: 6x11
    
    # Initialize with random integers
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Precompute indices for speed
    # We need to check 12 equations.
    # Eq (i,j,k): sum_r U[i*2+j, r] V[j*2+k, r] W[k*3+i, r] = 1
    
    while time.time() < deadline:
        # Random restart
        U = np.random.randint(-2, 3, size=(6, rank))
        V = np.random.randint(-2, 3, size=(4, rank))
        W = np.random.randint(-2, 3, size=(6, rank))
        
        # Local search
        for step in range(1000):
            if time.time() > deadline:
                break
            
            # Calculate loss
            loss = 0
            for i in range(3):
                for j in range(2):
                    for k in range(2):
                        s = 0
                        for r in range(rank):
                            s += U[i*2+j, r] * V[j*2+k, r] * W[k*3+i, r]
                        loss += (s - 1)**2
            
            if loss < 1e-9:
                return U, V, W
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            # Pick a random variable to change
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, 5)
                col = random.randint(0, rank-1)
                old_val = U[row, col]
                # Try changing to a nearby integer
                new_val = old_val + random.choice([-2, -1, 1, 2])
                if new_val < -5 or new_val > 5:
                    continue
                U[row, col] = new_val
                # Calculate new loss
                new_loss = 0
                for i in range(3):
                    for j in range(2):
                        for k in range(2):
                            s = 0
                            for r in range(rank):
                                s += U[i*2+j, r] * V[j*2+k, r] * W[k*3+i, r]
                            new_loss += (s - 1)**2
                if new_loss < loss:
                    loss = new_loss
                else:
                    U[row, col] = old_val
                    
            elif mat == 'V':
                row = random.randint(0, 3)
                col = random.randint(0, rank-1)
                old_val = V[row, col]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                if new_val < -5 or new_val > 5:
                    continue
                V[row, col] = new_val
                new_loss = 0
                for i in range(3):
                    for j in range(2):
                        for k in range(2):
                            s = 0
                            for r in range(rank):
                                s += U[i*2+j, r] * V[j*2+k, r] * W[k*3+i, r]
                            new_loss += (s - 1)**2
                if new_loss < loss:
                    loss = new_loss
                else:
                    V[row, col] = old_val
            else: # W
                row = random.randint(0, 5)
                col = random.randint(0, rank-1)
                old_val = W[row, col]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                if new_val < -5 or new_val > 5:
                    continue
                W[row, col] = new_val
                new_loss = 0
                for i in range(3):
                    for j in range(2):
                        for k in range(2):
                            s = 0
                            for r in range(rank):
                                s += U[i*2+j, r] * V[j*2+k, r] * W[k*3+i, r]
                            new_loss += (s - 1)**2
                if new_loss < loss:
                    loss = new_loss
                else:
                    W[row, col] = old_val

    # If best loss is 0, return best
    if best_loss < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find improved decomposition
    # 3x4x5 = 2 * (3x2x5)
    # 3x2x5 = 3x2x2 (cols 0-1) + 3x2x2 (cols 2-3) + 3x2x1 (col 4)
    
    # Time budget for search
    start_time = time.time()
    search_time_limit = 120 # Leave 30s for construction and safety
    
    dec_3x2x2 = None
    if time.time() - start_time < search_time_limit:
        dec_3x2x2 = find_3x2x2_decomposition(max_time=search_time_limit - (time.time() - start_time))
        
    if dec_3x2x2 is not None:
        U_322, V_322, W_322 = dec_3x2x2
        rank_322 = 11
        rank_321 = 6 # Schoolbook for 3x2x1
        rank_325 = rank_322 + rank_322 + rank_321 # 11 + 11 + 6 = 28
        total_rank = 2 * rank_325 # 56
        
        U = [[0] * total_rank for _ in range(n * m)] # 12 x 56
        V = [[0] * total_rank for _ in range(m * p)] # 20 x 56
        W = [[0] * total_rank for _ in range(p * n)] # 15 x 56
        
        # Block 1: Rows 0-1 of A (indices 0-1 in A's row dim? No, A is 3x4.
        # A = [A1 | A2] where A1 is 3x2 (cols 0-1), A2 is 3x2 (cols 2-3).
        # B = [B1; B2] where B1 is 2x5 (rows 0-1), B2 is 2x5 (rows 2-3).
        # C = A1 B1 + A2 B2.
        
        # A1 B1:
        # A1 is 3x2. B1 is 2x5.
        # B1 = [B1_0 | B1_1 | B1_2] where B1_0 is 2x2 (cols 0-1), B1_1 is 2x2 (cols 2-3), B1_2 is 2x1 (col 4).
        # We use dec_322 for A1 B1_0 and A1 B1_1.
        # And schoolbook for A1 B1_2.
        
        # Indices mapping:
        # For A1 B1_0 (cols 0-1 of C):
        # U indices: i*2 + j (i in 0..2, j in 0..1) -> 0..5
        # V indices: j*2 + k (j in 0..1, k in 0..1) -> 0..3
        # W indices: k*3 + i (k in 0..1, i in 0..2) -> 0..5
        # These correspond to global indices:
        # U: i*4 + j (j in 0..1) -> 0..7? No, A is 3x4. U is 12xRank.
        # Global U index for A1 (cols 0-1): i*4 + j.
        # Global V index for B1 (rows 0-1): j*5 + k.
        # Global W index for C (cols 0-1): k*3 + i.
        
        # Let's build the rank 56 decomposition.
        # Rank 0-10: A1 B1_0
        # Rank 11-21: A1 B1_1
        # Rank 22-27: A1 B1_2
        # Rank 28-38: A2 B2_0
        # Rank 39-49: A2 B2_1
        # Rank 50-55: A2 B2_2
        
        # Helper to add decomposition
        def add_3x2x2(U, V, W, U_offset, V_offset, W_offset, k_offset, dec):
            # U_offset: offset in global U (i*4 + j)
            # V_offset: offset in global V (j*5 + k)
            # W_offset: offset in global W (k*3 + i)
            # k_offset: offset for k index in local 3x2x2 (0 or 1)
            # dec: (U_loc, V_loc, W_loc)
            
            for r in range(11):
                for i in range(3):
                    for j in range(2):
                        U[i*4 + j][U_offset + r] += dec[0][i*2 + j][r]
                for j in range(2):
                    for k in range(2):
                        V[j*5 + (k + k_offset)][V_offset + r] += dec[1][j*2 + k][r]
                for k in range(2):
                    for i in range(3):
                        W[(k + k_offset)*3 + i][W_offset + r] += dec[2][k*3 + i][r]
                        
        # Add A1 B1_0 (k_offset=0, U_offset=0, V_offset=0, W_offset=0)
        add_3x2x2(U, V, W, 0, 0, 0, 0, dec_322)
        
        # Add A1 B1_1 (k_offset=2, U_offset=0, V_offset=0, W_offset=0)
        # Note: V index j*5 + k. k is 2,3.
        # W index k*3 + i. k is 2,3.
        # U index i*4 + j. j is 0,1.
        add_3x2x2(U, V, W, 0, 0, 0, 2, dec_322)
        
        # Add A1 B1_2 (k_offset=4, schoolbook)
        # A1 (3x2) * B1_2 (2x1).
        # i in 0..2, j in 0..1, k=4.
        # C[i, 4] = A[i, 0] B[0, 4] + A[i, 1] B[1, 4].
        # Terms: (i, 0, 4) and (i, 1, 4).
        for r in range(6):
            i = r // 2
            j = r % 2
            k = 4
            U[i*4 + j][22 + r] = 1
            V[j*5 + k][22 + r] = 1
            W[k*3 + i][22 + r] = 1
            
        # Add A2 B2_0 (k_offset=0, U_offset=2, V_offset=10, W_offset=0)
        # A2 is cols 2-3 of A. j in 2,3.
        # B2 is rows 2-3 of B. j in 2,3.
        # Local j is 0,1. Global j = local_j + 2.
        # U index: i*4 + (local_j + 2).
        # V index: (local_j + 2)*5 + k.
        # W index: k*3 + i.
        
        # We need to adjust the offsets in add_3x2x2 or create a more flexible one.
        # Let's just write it out or modify add function.
        
        def add_3x2x2_shifted(U, V, W, U_j_offset, V_j_offset, V_k_offset, W_k_offset, dec):
            for r in range(11):
                for i in range(3):
                    for j_local in range(2):
                        j_global = j_local + U_j_offset
                        U[i*4 + j_global][r] += dec[0][i*2 + j_local][r]
                for j_local in range(2):
                    j_global = j_local + V_j_offset
                    for k_local in range(2):
                        k_global = k_local + V_k_offset
                        V[j_global*5 + k_global][r] += dec[1][j_local*2 + k_local][r]
                for k_local in range(2):
                    k_global = k_local + W_k_offset
                    for i in range(3):
                        W[k_global*3 + i][r] += dec[2][k_local*3 + i][r]
                        
        # A2 B2_0: U_j_offset=2, V_j_offset=2, V_k_offset=0, W_k_offset=0
        # Rank offset 28
        # We need to add to U, V, W with rank offset 28.
        # The helper functions above assumed rank offset 0.
        # Let's refactor to pass rank offset.
        
        # Re-do with rank offset
        def add_3x2x2_final(U, V, W, U_j_offset, V_j_offset, V_k_offset, W_k_offset, rank_offset, dec):
            for r in range(11):
                for i in range(3):
                    for j_local in range(2):
                        j_global = j_local + U_j_offset
                        U[i*4 + j_global][rank_offset + r] += dec[0][i*2 + j_local][r]
                for j_local in range(2):
                    j_global = j_local + V_j_offset
                    for k_local in range(2):
                        k_global = k_local + V_k_offset
                        V[j_global*5 + k_global][rank_offset + r] += dec[1][j_local*2 + k_local][r]
                for k_local in range(2):
                    k_global = k_local + W_k_offset
                    for i in range(3):
                        W[k_global*3 + i][rank_offset + r] += dec[2][k_local*3 + i][r]
                        
        # Clear U, V, W and refill
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # 1. A1 B1_0: U_j=0, V_j=0, V_k=0, W_k=0, rank=0
        add_3x2x2_final(U, V, W, 0, 0, 0, 0, 0, dec_322)
        
        # 2. A1 B1_1: U_j=0, V_j=0, V_k=2, W_k=2, rank=11
        add_3x2x2_final(U, V, W, 0, 0, 2, 2, 11, dec_322)
        
        # 3. A1 B1_2: U_j=0, V_j=0, V_k=4, W_k=4, rank=22 (Schoolbook)
        for r in range(6):
            i = r // 2
            j = r % 2
            k = 4
            U[i*4 + j][22 + r] = 1
            V[j*5 + k][22 + r] = 1
            W[k*3 + i][22 + r] = 1
            
        # 4. A2 B2_0: U_j=2, V_j=2, V_k=0, W_k=0, rank=28
        add_3x2x2_final(U, V, W, 2, 2, 0, 0, 28, dec_322)
        
        # 5. A2 B2_1: U_j=2, V_j=2, V_k=2, W_k=2, rank=39
        add_3x2x2_final(U, V, W, 2, 2, 2, 2, 39, dec_322)
        
        # 6. A2 B2_2: U_j=2, V_j=2, V_k=4, W_k=4, rank=50 (Schoolbook)
        for r in range(6):
            i = r // 2
            j = 2 + r % 2 # j is 2 or 3
            k = 4
            U[i*4 + j][50 + r] = 1
            V[j*5 + k][50 + r] = 1
            W[k*3 + i][50 + r] = 1
            
        # Verify
        if verify_decomposition(np.array(U), np.array(V), np.array(W), n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            pass
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
