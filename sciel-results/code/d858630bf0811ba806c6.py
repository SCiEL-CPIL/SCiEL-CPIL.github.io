import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook (Rank 60)
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

    # Fallback: Rank 50 (10 x Rank 5 for 2x3x1)
    def make_rank50():
        # Rank 5 decomposition for 2x3x1
        # A = [[a11, a12, a13], [a21, a22, a23]], B = [b1, b2, b3]
        # m1 = (a11+a21)b1
        # m2 = (a12+a22)b2
        # m3 = (a13+a23)b3
        # m4 = (a11-a12)(b1-b2)
        # m5 = (a12-a13)(b2-b3)
        # This needs verification. If it's not exact rank 5, we use rank 6.
        # Let's use rank 6 to be safe if rank 5 is complex.
        # Actually, let's just use schoolbook if rank 50 is hard to verify quickly.
        # But rank 50 is easy to construct if we have a valid 2x3x1 decomposition.
        
        # Let's try to verify a simple rank 5 scheme.
        # If it fails, we return schoolbook.
        
        # For now, return schoolbook to ensure correctness.
        # The goal is to beat 60.
        pass

    # Strategy: Search for Rank 7 decomposition of 2x3x5
    # If found, combine two such blocks to get Rank 14 for 2x3x10.
    
    def verify_block(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def search_2x3x5_rank7():
        n, m, p = 2, 3, 5
        target_rank = 7
        # Random search with half-integers
        # Entries in U, V, W
        # U: 6x7, V: 15x7, W: 10x7
        
        # To increase success rate, we can try to fix some structure.
        # But random is safer for generic search.
        
        for _ in range(2000):
            if time.time() > deadline - 10:
                break
                
            # Generate random half-integers in [-2, 2]
            # Using integers * 0.5
            U_rand = np.random.randint(-4, 5, (n*m, target_rank)) * 0.5
            V_rand = np.random.randint(-4, 5, (m*p, target_rank)) * 0.5
            W_rand = np.random.randint(-4, 5, (p*n, target_rank)) * 0.5
            
            # Convert to lists for verify
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if verify_block(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
                
        return None

    # Try to find the block
    block = search_2x3x5_rank7()
    
    if block:
        U_block, V_block, W_block = block
        # Construct full 2x3x10 decomposition
        # Two blocks
        rank = 14
        U_full = [[0] * rank for _ in range(n * m)]
        V_full = [[0] * rank for _ in range(m * p)]
        W_full = [[0] * rank for _ in range(p * n)]
        
        # Block 1: k=0..4
        # Block 2: k=5..9
        
        for r in range(7):
            # U is same for both blocks (A factors are shared)
            # Wait, in tensor decomposition, U is independent of k.
            # So U_full[:, r] = U_block[:, r]
            # U_full[:, r+7] = U_block[:, r]
            
            for i in range(n*m):
                U_full[i][r] = U_block[i][r]
                U_full[i][r+7] = U_block[i][r]
                
            # V depends on k.
            # V_block has rows for j=0..2, k=0..4.
            # For full V, rows are j=0..2, k=0..9.
            # For r (first block), V_full[j, k, r] = V_block[j, k, r] for k=0..4.
            # For r (second block), V_full[j, k, r+7] = V_block[j, k-5, r] for k=5..9.
            
            for j in range(m):
                for k in range(5):
                    V_full[j*p + k][r] = V_block[j*p + k][r]
                    V_full[j*p + (k+5)][r+7] = V_block[j*p + k][r]
                    
            # W depends on k.
            # W_block has rows for k=0..4, i=0..1.
            # For full W, rows are k=0..9, i=0..1.
            
            for k in range(5):
                for i in range(n):
                    W_full[k*n + i][r] = W_block[k*n + i][r]
                    W_full[(k+5)*n + i][r+7] = W_block[k*n + i][r]
                    
        # Verify full decomposition
        if verify_block(U_full, V_full, W_full, n, m, p):
            return {"U": U_full, "V": V_full, "W": W_full}
            
    # If rank 14 not found, return schoolbook
    return make_schoolbook()
