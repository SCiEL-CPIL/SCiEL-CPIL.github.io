"""Matrix multiplication 2,3,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Build the target tensor T as a 3D array
    # T[x, y, z] = 1 if x = i*m+j, y = j*p+k, z = k*n+i for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Verification function
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            if np.max(np.abs(T_recon - T)) < tol:
                return True
        except:
            pass
        return False
    
    # Convert to list of lists with integer/half-integer values
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    clean_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-9:
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    # Strategy 1: Exploit block structure - 3x6 = two independent 3x3 blocks
    # For each block, solve 2x3 x 3x3 multiplication
    # Then combine: total rank = 2 * rank(2x3x3)
    
    n2, m2, p2 = 2, 3, 3
    T_sub = np.zeros((n2*m2, m2*p2, p2*n2), dtype=float)
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T_sub[i*m2+j, j*p2+k, k*n2+i] = 1.0
    
    def verify_sub(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            if np.max(np.abs(T_recon - T_sub)) < tol:
                return True
        except:
            pass
        return False
    
    def try_structured_decomp(target_rank):
        """
        Try to find a rank-target_rank decomposition for 2x3 x 3x3.
        
        Key insight: The tensor T_sub[i*3+j, j*3+k, k*2+i] = 1
        For fixed (i,k), the support is over j=0,1,2.
        
        We can try decompositions where each term has a specific (i,k) pattern
        but allows mixing in the j dimension.
        """
        import random
        random.seed(42 + target_rank)
        
        # Approach: Use the fact that for each (i,k) pair, we need to cover
        # the j-dimension. There are n2*p2 = 6 such (i,k) pairs.
        # Each needs at least 1 term, but we might be able to share.
        
        # Try various structured patterns
        for pattern_idx in range(20):
            if time.time() > deadline - 15:
                return None
            
            U_try = np.zeros((n2*m2, target_rank))
            V_try = np.zeros((m2*p2, target_rank))
            W_try = np.zeros((p2*n2, target_rank))
            
            # Assign (i,k) pairs to rank terms
            # There are 6 (i,k) pairs: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
            ik_pairs = [(i,k) for i in range(n2) for k in range(p2)]
            
            if target_rank >= 6:
                # Each term handles one (i,k) pair, with possible j-mixing
                for r in range(target_rank):
                    if r < 6:
                        i, k = ik_pairs[r]
                        # Base: cover all j for this (i,k)
                        for j in range(m2):
                            U_try[i*m2+j, r] = 1.0
                            V_try[j*p2+k, r] = 1.0
                            W_try[k*n2+i, r] = 1.0
                    else:
                        # Extra terms: try to create cancellations
                        # Random sparse with small values
                        for _ in range(4):
                            idx_u = random.randint(0, n2*m2-1)
                            U_try[idx_u, r] = random.choice([-1.0, 0.5, 1.0])
                        for _ in range(4):
                            idx_v = random.randint(0, m2*p2-1)
                            V_try[idx_v, r] = random.choice([-1.0, 0.5, 1.0])
                        for _ in range(4):
                            idx_w = random.randint(0, p2*n2-1)
                            W_try[idx_w, r] = random.choice([-1.0, 0.5, 1.0])
            else:
                # target_rank < 6: need to share (i,k) pairs
                # This is harder - try random assignment
                for r in range(target_rank):
                    # Pick 2 (i,k) pairs to handle
                    for _ in range(2):
                        i, k = random.choice(ik_pairs)
                        for j in range(m2):
                            U_try[i*m2+j, r] += 1.0
                            V_try[j*p2+k, r] += 1.0
                            W_try[k*n2+i, r] += 1.0
                    # Subtract to create cancellations
                    for _ in range(3):
                        i, k = random.choice(ik_pairs)
                        for j in range(m2):
                            U_try[i*m2+j, r] -= 1.0
                            V_try[j*p2+k, r] -= 1.0
                            W_try[k*n2+i, r] -= 1.0
            
            if verify_sub(U_try, V_try, W_try):
                return U_try, V_try, W_try
        
        return None
    
    # Try to find low-rank decomposition for 2x3 x 3x3
    # Known: schoolbook is 18, Strassen-like might give less
    # Try ranks from 13 up to 17
    found_sub = False
    for target_rank in range(13, 18):
        if time.time() > deadline - 20:
            break
        result = try_structured_decomp(target_rank)
        if result is not None:
            U_sub, V_sub, W_sub = result
            # Now combine for full 2x3 x 3x6
            U_full = np.zeros((n*m, target_rank * 2))
            V_full = np.zeros((m*p, target_rank * 2))
            W_full = np.zeros((p*n, target_rank * 2))
            
            # Block 1: k=0,1,2 (first 3 columns of B, first 3 rows of W output)
            for r in range(target_rank):
                U_full[:, r] = U_sub[:, r]
                V_full[:, r] = V_sub[:, r]
                W_full[:, r] = W_sub[:, r]
            
            # Block 2: k=3,4,5
            for r in range(target_rank):
                U_full[:, target_rank + r] = U_sub[:, r]
                # V: shift k by 3
                for row in range(m2*p2):
                    j = row // p2
                    k = row % p2
                    V_full[j*p + (k+3), target_rank + r] = V_sub[row, r]
                # W: shift k by 3 in output
                for row in range(p2*n2):
                    k = row // n2
                    i = row % n2
                    W_full[(k+3)*n + i, target_rank + r] = W_sub[row, r]
            
            if verify(U_full, V_full, W_full):
                best_rank = target_rank * 2
                best_U, best_V, best_W = U_full, V_full, W_full
                found_sub = True
                break
    
    # Strategy 2: Direct structured search on full tensor
    # Look for patterns where terms can be shared across k values
    if not found_sub and time.time() < deadline - 10:
        import random
        random.seed(999)
        
        # Try to find a decomposition where for each (i,j), the k-dimension
        # is handled with fewer than 6 terms
        # Idea: use polynomial/FFT-like structure in k
        
        for target_rank in range(24, 36):
            if time.time() > deadline - 5:
                break
            if target_rank >= best_rank:
                continue
            
            # Structured: group by (i,j) and try to compress k
            U_try = np.zeros((n*m, target_rank))
            V_try = np.zeros((m*p, target_rank))
            W_try = np.zeros((p*n, target_rank))
            
            # For each (i,j), assign target_rank//(n*m) terms to handle k
            terms_per_ij = target_rank // (n*m)
            if terms_per_ij < 1:
                continue
                
            col = 0
            for i in range(n):
                for j in range(m):
                    for t in range(terms_per_ij):
                        if col >= target_rank:
                            break
                        # This term handles (i,j) with some k pattern
                        # Use a basis vector in k-space
                        # Simple: one-hot in k
                        k_val = t % p
                        U_try[i*m+j, col] = 1.0
                        V_try[j*p+k_val, col] = 1.0
                        W_try[k_val*n+i, col] = 1.0
                        col += 1
            
            # Fill remaining columns with random
            while col < target_rank:
                for _ in range(3):
                    idx_u = random.randint(0, n*m-1)
                    U_try[idx_u, col] = random.choice([-1.0, 1.0])
                for _ in range(3):
                    idx_v = random.randint(0, m*p-1)
                    V_try[idx_v, col] = random.choice([-1.0, 1.0])
                for _ in range(3):
                    idx_w = random.randint(0, p*n-1)
                    W_try[idx_w, col] = random.choice([-1.0, 1.0])
                col += 1
            
            if verify(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Strategy 3: Try specific known small-rank patterns
    # For 2x2 x 2x2, Strassen gives rank 7
    # For our problem, try to embed Strassen-like patterns
    
    if time.time() < deadline - 8:
        # Try a hand-crafted approach:
        # The tensor T[i*3+j, j*3+k, k*2+i] = 1
        # Consider grouping: for fixed i, we have a 3x6 x 6x2 tensor
        # This is like multiplying 1x3 by 3x6 to get 1x6, done for 2 rows
        
        # Try rank 24: 4 terms per (i,j) pair, covering k with a 4-dim basis
        # Use DFT-like structure on k
        
        U_try = np.zeros((n*m, 24))
        V_try = np.zeros((m*p, 24))
        W_try = np.zeros((p*n, 24))
        
        # For each (i,j), use 4 terms to cover all 6 k values
        # This requires the 4 basis vectors to span the needed patterns
        # Use: k=0,1,2 in one term pattern, k=3,4,5 in another, plus 2 more
        
        col = 0
        for i in range(n):
            for j in range(m):
                # Term 1: k=0,1,2
                U_try[i*m+j, col] = 1.0
                V_try[j*p+0, col] = 1.0
                V_try[j*p+1, col] = 1.0
                V_try[j*p+2, col] = 1.0
                W_try[0*n+i, col] = 1.0
                W_try[1*n+i, col] = 1.0
                W_try[2*n+i, col] = 1.0
                col += 1
                # Term 2: k=3,4,5
                U_try[i*m+j, col] = 1.0
                V_try[j*p+3, col] = 1.0
                V_try[j*p+4, col] = 1.0
                V_try[j*p+5, col] = 1.0
                W_try[3*n+i, col] = 1.0
                W_try[4*n+i, col] = 1.0
                W_try[5*n+i, col] = 1.0
                col += 1
        
        # This gives rank 12, but it's not correct because each term
        # contributes to multiple (i,k) pairs simultaneously
        # Let's verify
        if verify(U_try, V_try, W_try):
            best_rank = 24
            best_U, best_V, best_W = U_try, V_try, W_try
    
    # Final: ensure we have a valid solution
    if not verify(best_U, best_V, best_W):
        U, V, W = schoolbook()
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    # Return the best solution found
    U_out = to_clean_list(best_U)
    V_out = to_clean_list(best_V)
    W_out = to_clean_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
