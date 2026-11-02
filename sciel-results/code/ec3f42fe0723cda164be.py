import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Vectorized version.
    """
    # Reshape U (n*m, r) to (n, m, r)
    # Reshape V (m*p, r) to (m, p, r)
    # Reshape W (p*n, r) to (p, n, r)
    
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # Compute S[i, j, k] = sum_t U[i, j, t] * V[j, k, t] * W[k, i, t]
    # einsum: 'ijr,jkr,ki r->ijk'
    # Note: W is indexed by k*n+i, so in reshaped W it is W_reshaped[k, i, :]
    S = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if all entries are 1
    if not np.all(S == 1):
        return False
    return True

def _solve_w_over_f2(U, V, n, m, p, r):
    """
    Given binary U (nm x r) and V (mp x r), solve for W (pn x r) over F2.
    Uses bitmask representation for speed.
    Returns W as a numpy array of 0/1, or None if no solution.
    """
    W = np.zeros((p * n, r), dtype=np.int64)
    
    # Precompute bitmasks for rows of U and V
    # U_bit[i*m+j] is a bitmask of the row U[i*m+j, :]
    # V_bit[j*p+k] is a bitmask of the row V[j*p+k, :]
    
    U_bits = [0] * (n * m)
    for idx in range(n * m):
        mask = 0
        for t in range(r):
            if U[idx, t] % 2 == 1:
                mask |= (1 << t)
        U_bits[idx] = mask
        
    V_bits = [0] * (m * p)
    for idx in range(m * p):
        mask = 0
        for t in range(r):
            if V[idx, t] % 2 == 1:
                mask |= (1 << t)
        V_bits[idx] = mask
        
    for k in range(p):
        for i in range(n):
            # Build m x r system over F2
            # A[j, t] = U[i*m+j, t] * V[j*p+k, t]
            # In bitmask: A_mask[j] = U_bits[i*m+j] & V_bits[j*p+k]
            
            A_masks = []
            for j in range(m):
                A_masks.append(U_bits[i * m + j] & V_bits[j * p + k])
            
            # We want to find x (bitmask) such that:
            # sum_t (A_masks[j] & (1<<t)) * x_t = 1 for each j
            # This is equivalent to:
            # The linear system A x = b (mod 2)
            # where A is m x r matrix defined by A_masks.
            # We perform Gaussian elimination on the rows A_masks.
            # The augmented matrix rows are (A_mask[j], 1).
            
            # Represent augmented row as (mask, bit)
            # We want to eliminate to find x.
            # Since m is small (3), we can just try to solve.
            # Actually, we can use the fact that we are solving for x.
            # Let's do Gaussian elimination on the transpose? No.
            
            # Standard GE on m x (r+1) matrix.
            # Rows are bitmasks of length r+1?
            # The last bit is the RHS (1).
            
            aug_rows = []
            for j in range(m):
                # Mask for A part
                a_mask = A_masks[j]
                # RHS is 1
                # We store as (a_mask, rhs_bit)
                aug_rows.append([a_mask, 1])
            
            # Gaussian Elimination
            # We want to transform to row echelon form.
            # Pivot search in column c (bit position).
            # If we find a row with bit c set, swap to top, eliminate others.
            
            # Since r can be up to 30, and m is 3, this is fast.
            row = 0
            for col in range(r):
                # Find pivot in rows [row, m) with bit 'col' set
                pivot = -1
                for rr in range(row, m):
                    if aug_rows[rr][0] & (1 << col):
                        pivot = rr
                        break
                
                if pivot == -1:
                    continue
                
                # Swap
                aug_rows[row], aug_rows[pivot] = aug_rows[pivot], aug_rows[row]
                
                # Eliminate other rows
                for rr in range(m):
                    if rr != row and (aug_rows[rr][0] & (1 << col)):
                        aug_rows[rr][0] ^= aug_rows[row][0]
                        aug_rows[rr][1] ^= aug_rows[row][1]
                
                row += 1
                if row == m:
                    break
            
            # Check for inconsistency:
            # If any row in [row, m) has 0 in A part and 1 in RHS
            for rr in range(row, m):
                if aug_rows[rr][0] == 0 and aug_rows[rr][1] == 1:
                    return None
            
            # Back-substitution
            # The solution x is determined by the pivot rows.
            # Free variables (columns where no pivot was found) are set to 0.
            x_mask = 0
            for rr in range(row):
                # The pivot column for this row is the first set bit in aug_rows[rr][0]
                # Actually, in reduced row echelon form (which we are creating by eliminating ALL other rows),
                # the pivot column is the only set bit in the A part?
                # No, we only eliminated below and above?
                # My loop eliminates ALL other rows (rr != row). So it is Gauss-Jordan.
                # So each pivot row has exactly one set bit in the A part?
                # Not necessarily, if we didn't eliminate higher bits?
                # The loop `for col in range(r)` ensures we process columns left to right.
                # When we pick a pivot at col, we eliminate that bit from ALL other rows.
                # So yes, in the final state, each pivot row has exactly one set bit in the A part.
                
                # Find the set bit
                a_part = aug_rows[rr][0]
                # The bit position is the index of the set bit
                # Since it's a power of 2 (only one bit set), we can find it.
                if a_part == 0:
                    continue # Should not happen for pivot rows
                # Get the bit index
                # Python int bit_length - 1
                bit_idx = a_part.bit_length() - 1
                if aug_rows[rr][1] == 1:
                    x_mask |= (1 << bit_idx)
            
            # Convert x_mask to array
            for t in range(r):
                if x_mask & (1 << t):
                    W[k * n + i, t] = 1
    
    # Verify over integers using numpy
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return W

def _try_rank_with_search(rank, n, m, p, max_attempts, deadline, seed=42):
    """
    Try to find a valid decomposition of given rank using random search.
    Returns (U, V, W) or None.
    """
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random sparse patterns with entries in {-1, 0, 1}
        # U: 9 x rank, V: 12 x rank
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        
        # For U: each row has 2-4 nonzeros from {-1, 1}
        for i in range(n * m):
            num_nonzero = rng.randint(2, min(4, rank))
            cols = rng.sample(range(rank), num_nonzero)
            for c in cols:
                U[i, c] = 1 if rng.random() < 0.5 else -1
        
        # For V: each row has 2-4 nonzeros from {-1, 1}
        for i in range(m * p):
            num_nonzero = rng.randint(2, min(4, rank))
            cols = rng.sample(range(rank), num_nonzero)
            for c in cols:
                V[i, c] = 1 if rng.random() < 0.5 else -1
        
        # Try to solve for W over F2 (use absolute values mod 2)
        U_mod2 = (U + 2) % 2  # Convert -1,0,1 to 1,0,1
        V_mod2 = (V + 2) % 2
        
        W = _solve_w_over_f2(U_mod2, V_mod2, n, m, p, rank)
        if W is not None:
            # The binary solution works over F2, which implies it works over Integers (0/1)
            # Verify with the binary U, V, W
            if _verify_decomposition(U_mod2, V_mod2, W, n, m, p):
                return U_mod2, V_mod2, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, t] = 1
                V_trivial[j * p + k, t] = 1
                W_trivial[k * n + i, t] = 1
                t += 1
    
    best_result = (rank_trivial, U_trivial, V_trivial, W_trivial)
    
    # Try ranks from 23 upward
    for rank in range(23, min(rank_trivial + 1, 30)):
        if time.time() > deadline:
            break
        
        # Use more attempts for lower ranks
        base_attempts = 5000 if rank <= 24 else 2000
        num_seeds = 5
        
        for seed_offset in range(num_seeds):
            if time.time() > deadline:
                break
            
            result = _try_rank_with_search(
                rank, n, m, p, 
                base_attempts // num_seeds, 
                deadline, 
                seed=42 + seed_offset * 1000
            )
            
            if result is not None:
                U, V, W = result
                best_result = (rank, U, V, W)
                break
        
        if best_result[0] == rank:
            break
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
