"""Combined program: rank-45 via 3x(3x3x2) with random search, fallback to rank-46 and schoolbook-54."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are within tol of half-integers. Return rounded half-integers if valid."""
    doubled = arr * 2.0
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _search_3x3x2(rank, max_attempts=5000, deadline=None):
    """Search for a rank-R decomposition of 3x3x2 matmul tensor.
    U: 9 x R, V: 6 x R, W: 6 x R
    Returns (U, V, W) as lists, or None.
    """
    n, m, p = 3, 3, 2
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        # Random sparse U, V with entries in {-1, 0, 1}
        U_rand = np.random.randint(-1, 2, size=(n * m, rank))
        V_rand = np.random.randint(-1, 2, size=(m * p, rank))
        
        W_sol = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Build 3 x rank matrix M where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                
                # Solve M @ w = ones(m)
                try:
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                # Check half-integer
                w_check = _check_half_integer(w_vec)
                if w_check is None:
                    valid = False
                    break
                W_sol[k * n + i, :] = w_check
        
        if valid:
            return U_rand.tolist(), V_rand.tolist(), W_sol.tolist()
    
    return None

def _search_3x3x3(rank, max_attempts=3000, deadline=None):
    """Search for a rank-R decomposition of 3x3x3 matmul tensor.
    U: 9 x R, V: 9 x R, W: 9 x R
    Returns (U, V, W) as lists, or None.
    """
    n, m, p = 3, 3, 3
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        U_rand = np.random.randint(-1, 2, size=(n * m, rank))
        V_rand = np.random.randint(-1, 2, size=(m * p, rank))
        
        W_sol = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                
                try:
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                w_check = _check_half_integer(w_vec)
                if w_check is None:
                    valid = False
                    break
                W_sol[k * n + i, :] = w_check
        
        if valid:
            return U_rand.tolist(), V_rand.tolist(), W_sol.tolist()
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the n x m x p matmul tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    and for all other (a, b, c) triples where indices don't align, the sum == 0.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all valid (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that no spurious contributions exist:
    # For any (a, b, c) where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # The tensor entry T[a,b,c] should be 1 only if j==j', i==i', k==k'
    # We need to verify that for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] equals the correct delta.
    # This is expensive but necessary for correctness.
    # T[a,b,c] = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i (all consistent)
    # Otherwise 0.
    
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                # Expected value
                expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                val = 0.0
                for r in range(rank):
                    val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                if abs(val - expected) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Strategy 1: Rank 45 via 3 parallel 3x3x2 blocks
    # B is 3x6, split into B1, B2, B3 each 3x2.
    # C is 3x6, split into C1, C2, C3 each 3x2.
    # Ci = A * Bi, each a 3x3 * 3x2 multiplication.
    # If we have a rank-15 decomposition for 3x3x2, we use 3*15 = 45.
    
    # The 3x3x2 decomposition: U_shared (9x15), V_block (6x15), W_block (6x15)
    # For block t (t=0,1,2), V uses columns k=t*2..t*2+1 of B.
    # W uses rows k=t*2..t*2+1 of C.
    # U is shared across all blocks (same A).
    
    # But wait: if U is shared, the total rank is not 3*15 but rather we need to
    # combine them. Actually, the direct sum means:
    # For block t: term r contributes U[i,j,r] * V_t[j,k_t,r] * W_t[k_t,i,r]
    # where k_t = t*2 + k_local.
    # If U is the same for all blocks, we can write:
    # Total rank = 15 (shared U) + ... no, that's not how it works.
    # 
    # Actually, the standard direct sum: each block has its own U, V, W of rank 15.
    # But U must be the same for all blocks because A is shared.
    # So we need a single U (9x15) that works for all three 3x3x2 blocks.
    # This means we need a rank-15 decomposition of 3x3x2 that uses the same U
    # for all three k-splits. But the three blocks are independent 3x3x2 problems
    # with the same A but different B columns.
    #
    # The correct approach: the 3x3x6 tensor is not a direct sum of three 3x3x2 tensors
    # in the simple sense because U is shared. Instead, we can view it as:
    # We need U (9xR), V (18xR), W (18xR) such that the full tensor is decomposed.
    # If we set R=45, we can partition the 45 columns into three groups of 15.
    # Group t (t=0,1,2): columns 15*t..15*t+14.
    # For group t, U[:, 15*t:15*t+15] = U_t (same for all t? No, they can differ).
    # Actually, U can be different for each group since the columns are independent.
    # So U[:, 15*t:15*t+15] can be any 9x15 matrix.
    # V[:, 15*t:15*t+15] should only be nonzero for rows j*6 + k where k in {2*t, 2*t+1}.
    # W[:, 15*t:15*t+15] should only be nonzero for rows k*3 + i where k in {2*t, 2*t+1}.
    
    # So the construction is:
    # Find three rank-15 decompositions (U_t, V_t, W_t) for 3x3x2.
    # Then:
    # U_total[:, 15*t:15*t+15] = U_t
    # V_total[j*6 + 2*t + k_local, 15*t:15*t+15] = V_t[j*2 + k_local, r]
    # W_total[(2*t + k_local)*3 + i, 15*t:15*t+15] = W_t[k_local*3 + i, r]
    
    # This gives a valid rank-45 decomposition if each (U_t, V_t, W_t) is valid for 3x3x2.
    
    # Search for three rank-15 3x3x2 decompositions
    blocks = []
    for t in range(3):
        if time.time() > deadline:
            break
        result = _search_3x3x2(15, max_attempts=3000, deadline=deadline)
        if result is None:
            break
        blocks.append(result)
    
    if len(blocks) == 3:
        # Construct rank-45 solution
        R = 45
        U_total = [[0] * R for _ in range(n * m)]
        V_total = [[0] * R for _ in range(m * p)]
        W_total = [[0] * R for _ in range(p * n)]
        
        for t in range(3):
            U_t, V_t, W_t = blocks[t]
            offset = 15 * t
            for i in range(3):
                for j in range(3):
                    for r in range(15):
                        U_total[i * 3 + j][offset + r] = U_t[i * 3 + j][r]
            for j in range(3):
                for k_local in range(2):
                    for r in range(15):
                        k_global = 2 * t + k_local
                        V_total[j * 6 + k_global][offset + r] = V_t[j * 2 + k_local][r]
            for k_local in range(2):
                for i in range(3):
                    for r in range(15):
                        k_global = 2 * t + k_local
                        W_total[k_global * 3 + i][offset + r] = W_t[k_local * 3 + i][r]
        
        # Verify
        if _verify_decomposition(U_total, V_total, W_total, n, m, p):
            return {"U": U_total, "V": V_total, "W": W_total}
    
    # Strategy 2: Rank 46 via 2 parallel 3x3x3 blocks
    # B is 3x6, split into B1, B2 each 3x3.
    # Each 3x3x3 block needs rank 23. Total rank 46.
    cores = []
    for t in range(2):
        if time.time() > deadline:
            break
        result = _search_3x3x3(23, max_attempts=2000, deadline=deadline)
        if result is None:
            break
        cores.append(result)
    
    if len(cores) == 2:
        R = 46
        U_total = [[0] * R for _ in range(n * m)]
        V_total = [[0] * R for _ in range(m * p)]
        W_total = [[0] * R for _ in range(p * n)]
        
        for t in range(2):
            U_t, V_t, W_t = cores[t]
            offset = 23 * t
            for i in range(3):
                for j in range(3):
                    for r in range(23):
                        U_total[i * 3 + j][offset + r] = U_t[i * 3 + j][r]
            for j in range(3):
                for k_local in range(3):
                    for r in range(23):
                        k_global = 3 * t + k_local
                        V_total[j * 6 + k_global][offset + r] = V_t[j * 3 + k_local][r]
            for k_local in range(3):
                for i in range(3):
                    for r in range(23):
                        k_global = 3 * t + k_local
                        W_total[k_global * 3 + i][offset + r] = W_t[k_local * 3 + i][r]
        
        if _verify_decomposition(U_total, V_total, W_total, n, m, p):
            return {"U": U_total, "V": V_total, "W": W_total}
    
    # Strategy 3: Try rank 48, 50, 52, 53 via 2 parallel 3x3x3 with higher rank per core
    for core_rank in [24, 25, 26, 27]:
        if time.time() > deadline:
            break
        cores = []
        for t in range(2):
            result = _search_3x3x3(core_rank, max_attempts=1000, deadline=deadline)
            if result is None:
                cores = None
                break
            cores.append(result)
        
        if cores is not None and len(cores) == 2:
            R = 2 * core_rank
            U_total = [[0] * R for _ in range(n * m)]
            V_total = [[0] * R for _ in range(m * p)]
            W_total = [[0] * R for _ in range(p * n)]
            
            for t in range(2):
                U_t, V_t, W_t = cores[t]
                offset = core_rank * t
                for i in range(3):
                    for j in range(3):
                        for r in range(core_rank):
                            U_total[i * 3 + j][offset + r] = U_t[i * 3 + j][r]
                for j in range(3):
                    for k_local in range(3):
                        for r in range(core_rank):
                            k_global = 3 * t + k_local
                            V_total[j * 6 + k_global][offset + r] = V_t[j * 3 + k_local][r]
                for k_local in range(3):
                    for i in range(3):
                        for r in range(core_rank):
                            k_global = 3 * t + k_local
                            W_total[k_global * 3 + i][offset + r] = W_t[k_local * 3 + i][r]
            
            if _verify_decomposition(U_total, V_total, W_total, n, m, p):
                return {"U": U_total, "V": V_total, "W": W_total}
    
    # Fallback: Schoolbook rank 54
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
