import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5 seconds buffer before 120s limit
    
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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

    def verify_decomposition_exact(U, V, W, n, m, p):
        """
        Verify exact decomposition.
        U, V are integers. W is list of floats (half-integers).
        We convert W to integers by multiplying by 2.
        Condition: sum_r U[a][r] * V[b][r] * W[c][r] == 1
        If W[c][r] = w_int[c][r] / 2, then sum_r U*V*w_int / 2 == 1 => sum_r U*V*w_int == 2
        """
        num_u = len(U)
        num_v = len(V)
        num_w = len(W)
        if num_u != n * m or num_v != m * p or num_w != p * n:
            return False
        
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check if W entries are half-integers
        for c in range(num_w):
            for r in range(rank):
                if abs(W[c][r] * 2 - round(W[c][r] * 2)) > 1e-9:
                    return False
        
        # Convert W to integers (x2)
        W_int = [[int(round(W[c][r] * 2)) for r in range(rank)] for c in range(num_w)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W_int[c][r]
                    if s != 2:
                        return False
        return True

    def find_decomposition_for_rank(r, n, m, p, time_limit):
        """
        Attempt to find a rank-r decomposition.
        Strategy:
        1. Randomly initialize U and V with small integers {-1, 0, 1}.
        2. Construct the linear system for W: A * W_flat = b, where A depends on U, V.
           A[i,j,k, c*n+r] = U[i*m+j, r] * V[j*p+k, r]
           b[i,j,k] = 1
        3. Solve for W_flat using least squares.
        4. Check if the solution is exact (residual ~ 0).
        5. Check if W entries are half-integers.
        6. If so, verify exactly and return.
        
        Returns dict if found, else None.
        """
        num_eq = n * m * p
        num_unk = p * n * r
        
        # Precompute indices for equations
        eqs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eqs.append((i, j, k))
        
        # Precompute column mapping for W
        # W index is c * r + r_idx, where c = k*n+i
        # A row (i,j,k), col (c, r_idx) = U[i*m+j, r_idx] * V[j*p+k, r_idx]
        
        # We will iterate over random U, V
        max_attempts = 200
        attempts_per_iter = 0
        last_check = time.time()
        
        while time.time() < time_limit:
            # Random U, V
            # Using sparse random matrices might help? Or dense?
            # Let's try dense with {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            
            # Build A
            # A is (num_eq, num_unk)
            # We can build it efficiently
            A = np.zeros((num_eq, num_unk), dtype=np.float64)
            
            for idx, (i, j, k) in enumerate(eqs):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                c_base = k * n + i
                for r_idx in range(r):
                    val = u_row[r_idx] * v_row[r_idx]
                    if val != 0:
                        A[idx, c_base * r + r_idx] = val
            
            b = np.ones(num_eq, dtype=np.float64)
            
            # Solve
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            # Check if W is close to half-integers
            W_float = W_flat.reshape(p * n, r)
            W_half = np.round(W_float * 2) / 2.0
            
            # Check error
            if np.max(np.abs(W_float - W_half)) > 1e-6:
                continue
            
            # Convert to list of lists for verification
            U_list = [row[:] for row in U]
            V_list = [row[:] for row in V]
            W_list = [[float(x) for x in row] for row in W_half]
            
            # Verify exactly
            if verify_decomposition_exact(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
            
            attempts_per_iter += 1
            if time.time() - last_check > 1.0:
                last_check = time.time()
        
        return None

    def find_decomposition_strategies(r, n, m, p, time_limit):
        """
        Try multiple strategies for a given rank.
        """
        # Strategy 1: Random {-1, 0, 1}
        # Strategy 2: Random small integers {-2, -1, 0, 1, 2}
        # Strategy 3: Random with specific structure (e.g., U has 1s on diagonal?)
        
        # For now, just use the random method which is general.
        # We can add more heuristics if needed.
        
        return find_decomposition_for_rank(r, n, m, p, time_limit)

    # Search ranks
    # Schoolbook is 42.
    # Known lower bounds?
    # For 2x2x2, rank 7.
    # For 2x3x7, let's search from 21 to 41.
    
    # To maximize chance, we should allocate time wisely.
    # Lower ranks are harder.
    # Let's try a few random ranks first? Or sequential?
    # Sequential from low to high ensures we find the minimum if we find any.
    # But if rank 21 doesn't exist, we waste time.
    # Let's try ranks in a specific order: 21, 22, ..., 41.
    # But maybe some ranks are more likely?
    # Usually, the rank is close to n*m*p.
    # However, Strassen's algorithm shows reduction is possible.
    # For 2x3x7, I don't have specific knowledge of the minimal rank.
    
    # Let's allocate a time slice for each rank.
    # Total time available ~ 110s.
    # 21 ranks to try.
    # ~ 5s per rank.
    
    ranks_to_try = list(range(21, 42))
    
    # Shuffle? No, we want minimal rank.
    # But if we assume minimal rank is not 21, we might want to try mid-ranks first?
    # No, the task asks for "rank-minimal". If we find 30, we should keep searching for 29, 28...
    # So we must search from low to high.
    # If we find a decomposition for rank r, we return it immediately?
    # No, we should ensure it's the MINIMAL.
    # But finding a decomposition for rank r doesn't prove r is minimal.
    # However, the task says "Find a rank-minimal exact tensor decomposition".
    # And "score is the rank".
    # If I return rank 30, but rank 29 exists, my score is 30.
    # If I return rank 29, my score is 29.
    # So I should search from low to high, and return the FIRST one I find.
    # This guarantees I return the lowest rank I found.
    # It does NOT guarantee it is the true minimal rank, but it's the best I can do within time.
    
    # Wait, if I find rank 30 first, and then find rank 29 later, I should return 29.
    # So I should search 21, 22, ... 41.
    # If I find one at 25, I return it.
    # I don't need to check 26..41.
    
    # But what if 21-24 don't exist? I waste time.
    # Is there a way to skip?
    # Not easily.
    
    # Let's try to be smarter.
    # Maybe try a few random ranks first to see if any exist?
    # If I find rank 35, I know the answer is <= 35.
    # Then I search 21..34.
    # This might be faster if the minimal rank is high.
    
    # Let's do a hybrid:
    # 1. Try a few random ranks (e.g., 30, 35, 40) to get an upper bound.
    # 2. Search from 21 up to (upper_bound - 1).
    
    # But if the minimal rank is 21, step 1 is wasted.
    
    # Let's stick to sequential 21..41 but with a time limit per rank.
    # If a rank takes too long, move to next.
    
    # Actually, the linear system size is (42, 2*r).
    # For r=21, 42x42.
    # For r=41, 42x82.
    # Solving 42x42 is fast.
    # The bottleneck is the number of attempts.
    
    # Let's give each rank a budget of 3 seconds.
    # 21 ranks * 3s = 63s.
    # Remaining 47s for... ?
    # Maybe increase budget for lower ranks?
    
    # Let's try:
    # r=21..30: 5s each (50s)
    # r=31..41: 2s each (22s)
    # Total 72s.
    
    # Or just uniform 4s each. 21 * 4 = 84s.
    
    # Let's use a dynamic approach.
    # Start with 2s per rank.
    # If we find nothing, we just stop when time is up.
    
    current_rank = 21
    while current_rank <= 41 and time.time() < deadline:
        rank_budget = deadline - time.time()
        if rank_budget < 1.0:
            break
        
        # Allocate at most 5s per rank, or remaining time if less
        alloc = min(5.0, rank_budget)
        
        res = find_decomposition_strategies(current_rank, n, m, p, start_time + (time.time() - start_time) + alloc)
        # The time_limit in the function is absolute.
        # Let's fix the call.
        
        res = find_decomposition_for_rank(current_rank, n, m, p, time.time() + alloc)
        
        if res is not None:
            return res
        
        current_rank += 1
    
    # Fallback
    return _schoolbook_decomposition(n, m, p)

# Test locally
if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Found rank: {rank}")
    # Verify
    n, m, p = DIMENSIONS
    if verify_decomposition_exact(res['U'], res['V'], res['W'], n, m, p):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
