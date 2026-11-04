import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100
    
    # Deadline: 150s limit, use 140s for safety
    deadline = time.time() + 140
    
    # Build schoolbook as baseline
    def make_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    
    # Verify function using vectorized operations for speed
    def verify(U, V, W, rank):
        # T_r = U[:, r] @ V[:, r] -> shape (n*m, m*p) ? No.
        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Let's check for each (i,j,k)
        # This is a 3-tensor contraction.
        # Efficient way: for each r, compute outer product of U_col and V_col, multiply by W_col?
        # Actually, just loop over r and accumulate.
        # Since rank is small (<=100), a loop is fine.
        # But let's use numpy for the inner sum.
        
        # Initialize a 3D array or just check sums
        # S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can reshape U, V, W to make this easier?
        # U is (n*m, R), V is (m*p, R), W is (p*n, R)
        
        # Let's just do a fast loop. 100 * 100 = 10000 operations. Very fast.
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # This doesn't directly give the sum for all (i,j,k) easily without a 3D array
            # But we can check the specific constraints.
            
        # Alternative: Check each (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Search for rank 99
    # Strategy: Try to eliminate one column.
    # If we eliminate column c, we need to adjust others.
    # This is a non-linear problem. 
    # Given the difficulty, we perform a limited random search with a hard time cutoff.
    
    import random
    random.seed(42)
    
    # We only try a few attempts to avoid wasting time
    # If we find a valid rank 99, we return it.
    # Otherwise, we return rank 100.
    
    # Note: The probability of finding a valid exact integer/half-integer solution
    # for rank 99 by random perturbation is extremely low.
    # However, we must try to be "stronger" than the seed.
    # The seed does 20 attempts for rank 99, 98, 97.
    # We can try a more structured perturbation: 
    # Pick a column c to remove. 
    # Add 0.5 to two other columns c1, c2 that share indices with c.
    # This is what the seed does.
    
    # Let's just run the seed's logic but with a tighter time check and numpy arrays.
    
    for target_rank in [99]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(50): # Increased attempts slightly, but fast
            if time.time() > deadline - 5:
                break
            
            U = U_sb.copy()
            V = V_sb.copy()
            W = W_sb.copy()
            
            col_to_remove = random.randint(0, schoolbook_rank - 1)
            
            i0 = col_to_remove // (m*p)
            j0 = (col_to_remove // p) % m
            k0 = col_to_remove % p
            
            # Zero out this column
            U[:, col_to_remove] = 0
            V[:, col_to_remove] = 0
            W[:, col_to_remove] = 0
            
            # Find candidates
            candidates = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i,j,k) != (i0,j0,k0):
                            shares = (i == i0) + (j == j0) + (k == k0)
                            if shares > 0:
                                candidates.append((i,j,k))
            
            if len(candidates) >= 2:
                c1, c2 = random.sample(candidates, 2)
                col1 = c1[0]*m*p + c1[1]*p + c1[2]
                col2 = c2[0]*m*p + c2[1]*p + c2[2]
                
                # Add 0.5
                U[:, col1] += 0.5
                V[:, col1] += 0.5
                W[:, col1] += 0.5
                U[:, col2] += 0.5
                V[:, col2] += 0.5
                W[:, col2] += 0.5
                
                # Check validity
                # We only need to check if the removed term is compensated and others aren't broken.
                # But checking all is safer.
                if verify(U, V, W, schoolbook_rank):
                    # Remove the zero column
                    mask = np.ones(schoolbook_rank, dtype=bool)
                    mask[col_to_remove] = False
                    U3 = U[:, mask]
                    V3 = V[:, mask]
                    W3 = W[:, mask]
                    
                    if verify(U3, V3, W3, schoolbook_rank - 1):
                        best_rank = schoolbook_rank - 1
                        best_U, best_V, best_W = U3, V3, W3
                        break
        
        if best_rank < schoolbook_rank:
            break

    # Convert back to lists of lists if required? 
    # The interface says "factor matrices of shapes...". 
    # The seed returns lists of lists. 
    # "integer/half-integer entries"
    
    # The seed returns U_sb which is a list of lists.
    # My U_sb is a numpy array. 
    # I should return lists of lists to be safe, or check if numpy arrays are accepted.
    # The prompt says "factor matrices". Usually numpy arrays are fine, but let's stick to the seed's format if possible.
    # However, numpy is faster for verification.
    # Let's convert back to lists of lists for the return value to match the seed's interface exactly.
    
    def to_list(arr):
        return arr.tolist()
        
    if best_rank == schoolbook_rank:
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    else:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
