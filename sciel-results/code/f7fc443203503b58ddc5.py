import numpy as np
import time
import random

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s
    
    # Start with schoolbook decomposition (rank 64)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    # Try to find a better decomposition using random search + optimization
    # Strategy: Start from Strassen-inspired block structure and try to reduce rank
    
    best_rank = rank_schoolbook
    best_decomp = {
        'U': U_school.copy(),
        'V': V_school.copy(), 
        'W': W_school.copy()
    }
    
    # First, try the 8-block Strassen approach (rank 56)
    # Divide the 2x16 matrix into 8 blocks of 2x2
    # Each 2x2 multiplication uses rank 7 (Strassen)
    # Total rank: 8 * 7 = 56
    
    def strassen_2x2():
        """Return Strassen decomposition for 2x2 * 2x2 -> 2x2"""
        # U: 4x7 (rows: a,b,c,d of A)
        # V: 4x7 (rows: e,f,g,h of B) 
        # W: 4x7 (rows: c00,c01,c10,c11 of C)
        
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e  
        # m3 = (a+b)h
        # m4 = d(f-g)
        # m5 = a(e-f)
        # m6 = c(g-h)
        # m7 = bf
        # 
        # c00 = m1 + m5 - m3
        # c01 = m1 + m7 - m5 - m2
        # c10 = m1 + m2 - m4 - m6
        # c11 = m1 + m4 - m6
        
        U_s = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # a
            [0, 1, 1, 0, 0, 0, 1],  # b
            [0, 1, 0, 0, 0, 1, 0],  # c
            [1, 1, 0, 1, 0, 0, 0]   # d
        ], dtype=float)
        
        V_s = np.array([
            [1, 1, 0, 0, 1, 0, 0],  # e
            [0, 0, 0, 1, -1, 0, 1], # f
            [0, 0, 0, -1, 0, 1, 0], # g
            [1, 0, 1, 0, 0, -1, 0]  # h
        ], dtype=float)
        
        W_s = np.array([
            [1, 0, -1, 0, 1, 0, 0],   # c00 = m1 - m3 + m5
            [1, -1, 0, 0, -1, 0, 1],  # c01 = m1 - m2 - m5 + m7
            [1, 1, 0, -1, 0, -1, 0],  # c10 = m1 + m2 - m4 - m6
            [1, 0, 0, 1, 0, -1, 0]    # c11 = m1 + m4 - m6
        ], dtype=float)
        
        return U_s, V_s, W_s
    
    # Build 8-block Strassen decomposition
    num_blocks = p // 2  # 8 blocks of size 2
    rank_strassen = num_blocks * 7  # 56
    
    U_str = np.zeros((n*m, rank_strassen), dtype=float)
    V_str = np.zeros((m*p, rank_strassen), dtype=float)
    W_str = np.zeros((p*n, rank_strassen), dtype=float)
    
    U_s, V_s, W_s = strassen_2x2()
    
    for b in range(num_blocks):
        block_offset = b * 7
        for r in range(7):
            U_str[:, block_offset + r] = U_s[:, r]
            V_str[:, block_offset + r] = V_s[:, r]
            W_str[:, block_offset + r] = W_s[:, r]
        
        # Map block indices
        # V rows: for block b, columns 2*b and 2*b+1 of B
        # V_str row j*p + k should get V_s[j*2 + (k - 2*b)] if k in [2*b, 2*b+1]
        # Actually, let me rebuild this more carefully
        
    # Let me redo the 8-block construction properly
    U_str = np.zeros((n*m, rank_strassen), dtype=float)
    V_str = np.zeros((m*p, rank_strassen), dtype=float)
    W_str = np.zeros((p*n, rank_strassen), dtype=float)
    
    for b in range(num_blocks):
        block_offset = b * 7
        for r in range(7):
            U_str[:, block_offset + r] = U_s[:, r]
            
            # For V: block b uses B columns 2*b and 2*b+1
            # V_s has rows for e,f,g,h which correspond to B[0,2b], B[0,2b+1], B[1,2b], B[1,2b+1]
            # V_str rows: j*p + k where j=0,1 and k=0..15
            # For this block, we need to put V_s values at the right positions
            # V_s[0] = e component -> V_str[0*p + 2*b] += V_s[0, r]
            # V_s[1] = f component -> V_str[0*p + 2*b + 1] += V_s[1, r]
            # V_s[2] = g component -> V_str[1*p + 2*b] += V_s[2, r]
            # V_s[3] = h component -> V_str[1*p + 2*b + 1] += V_s[3, r]
            
            V_str[0*p + 2*b, block_offset + r] = V_s[0, r]
            V_str[0*p + 2*b + 1, block_offset + r] = V_s[1, r]
            V_str[1*p + 2*b, block_offset + r] = V_s[2, r]
            V_str[1*p + 2*b + 1, block_offset + r] = V_s[3, r]
            
            # For W: block b produces C columns 2*b and 2*b+1
            # W_s rows: c00, c01, c10, c11
            # W_str rows: k*n + i where k=0..15, i=0,1
            # For this block:
            # W_s[0] (c00) -> W_str[2*b * n + 0] += W_s[0, r]
            # W_s[1] (c01) -> W_str[2*b * n + 1] += W_s[1, r]
            # W_s[2] (c10) -> W_str[(2*b+1) * n + 0] += W_s[2, r]
            # W_s[3] (c11) -> W_str[(2*b+1) * n + 1] += W_s[3, r]
            
            W_str[2*b * n + 0, block_offset + r] = W_s[0, r]
            W_str[2*b * n + 1, block_offset + r] = W_s[1, r]
            W_str[(2*b+1) * n + 0, block_offset + r] = W_s[2, r]
            W_str[(2*b+1) * n + 1, block_offset + r] = W_s[3, r]
    
    # Verify the Strassen decomposition
    def verify_decomp(U, V, W, n, m, p, random_seed=42):
        """Verify that the decomposition is correct"""
        rng = np.random.RandomState(random_seed)
        for _ in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            C_actual = W @ (V @ (U.T @ A.T)).T
            if not np.allclose(C_actual, C_expected, atol=1e-10):
                return False
        return True
    
    if verify_decomp(U_str, V_str, W_str, n, m, p):
        best_rank = rank_strassen
        best_decomp = {
            'U': U_str.copy(),
            'V': V_str.copy(),
            'W': W_str.copy()
        }
    
    # Now try to reduce the rank further using linear algebra
    # If the columns of [U, V, W] are linearly dependent, we can reduce rank
    
    def try_reduce_rank(U, V, W, n, m, p, max_iter=100):
        """Try to find linear dependencies in the decomposition to reduce rank"""
        rank = U.shape[1]
        
        # Stack all matrices to find linear dependencies
        # The decomposition is valid if U * V * W.T gives the correct tensor
        # Actually, we need: for each term r, U[:,r] * V[:,r] * W[:,r] contributes to the tensor
        # The full tensor is sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # To check if we can eliminate a term, we need to see if U[:,r] is in the span
        # of other U columns, etc. This is complex.
        
        # Simpler approach: use random linear combinations to see if we can merge terms
        # This is a heuristic search
        
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        current_rank = rank
        
        for attempt in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick two random columns to try to merge
            r1 = rng.randint(0, current_rank - 1)
            r2 = rng.randint(0, current_rank - 1)
            if r1 == r2:
                continue
            
            # Try to express column r2 as a linear combination of column r1 and others
            # This is expensive, so let's use a simpler heuristic:
            # Randomly perturb and check if rank can be reduced
            
            # Alternative: use SVD to find the minimum rank decomposition
            # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
            # We want to find the minimum number of terms
            
            # For now, just return the current decomposition
            pass
        
        return current_U, current_V, current_W, current_rank
    
    # Use a more sophisticated approach: 
    # The key insight is that in the 8-block Strassen, there might be shared terms
    # across blocks that can be combined
    
    # Let's try a different approach: use the fact that we're multiplying 2x2 by 2x16
    # We can think of this as 16 independent 2x2 multiplications, each with rank 7
    # But we might be able to do better by exploiting the structure
    
    # For now, return the best we have (Strassen rank 56)
    return best_decomp

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {result['U'].shape[1]}")
